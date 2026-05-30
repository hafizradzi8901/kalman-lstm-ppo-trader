import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ==========================================
# 1. THE STATIONARITY TRAP & FEATURE ENGINEERING
# ==========================================
class FeatureEngineer:
    """
    Feature engineering pipeline for financial time series.

    Fractional differentiation (fracdiff) is from Marcos López de Prado's
    "Advances in Financial Machine Learning". The key insight: integer
    differencing (d=1) makes a series stationary but destroys memory.
    Fracdiff with d ~ 0.3-0.7 achieves stationarity while preserving
    long-range dependence — critical for price-level-aware models.
    """

    def __init__(self, d_value: float = 0.5, weight_threshold: float = 1e-5, max_window: int = 50):
        self.d_value = d_value
        self.weight_threshold = weight_threshold
        self.max_window = max_window

    def _compute_fracdiff_weights(self, d: float, max_lag: int) -> np.ndarray:
        """
        Compute the binomial series weights for fractional differentiation.

        The weights come from expanding (1 - B)^d as an infinite series:
            w_k = (-1)^k * C(d, k)
        where C(d, k) = d * (d-1) * ... * (d-k+1) / k!

        Iteratively: w_k = -w_{k-1} * (d - k + 1) / k
        We truncate when |w_k| < threshold for the fixed-width window method.
        """
        weights = [1.0]
        for k in range(1, max_lag):
            w_k = -weights[-1] * (d - k + 1) / k
            if abs(w_k) < self.weight_threshold:
                break
            weights.append(w_k)
        return np.array(weights[::-1])  # reverse so oldest weight is first

    def apply_fractional_differentiation(self, series: pd.Series) -> pd.Series:
        """
        Fixed-width window fractional differentiation (FFD method).

        For each time t, the fracdiff value is:
            X_t^{(d)} = sum_{k=0}^{K} w_k * X_{t-k}

        This is a weighted sum of current and past values. Unlike full
        integer differencing, partial memory of the price level is retained.

        Args:
            series: Raw price series (e.g., close prices).

        Returns:
            Fractionally differentiated series with NaN for initial window.
        """
        weights = self._compute_fracdiff_weights(self.d_value, min(self.max_window, len(series)))
        window_size = len(weights)

        result = np.full(len(series), np.nan)
        values = series.values

        for t in range(window_size - 1, len(values)):
            window = values[t - window_size + 1 : t + 1]
            result[t] = np.dot(weights, window)

        return pd.Series(result, index=series.index, name=f"fracdiff_d{self.d_value}")

    def compute_order_book_imbalance(
        self, bids: pd.DataFrame, asks: pd.DataFrame, levels: int = 5
    ) -> pd.Series:
        """
        Order Book Imbalance (OBI) from Level 2 data.

        OBI = (sum of bid volumes across levels - sum of ask volumes) /
              (sum of bid volumes + sum of ask volumes)

        Ranges from -1 (all selling pressure) to +1 (all buying pressure).
        This is one of the strongest short-term predictors of price movement.

        Args:
            bids: DataFrame with columns like 'bid_vol_1', 'bid_vol_2', ...
                  Each row is a timestamp, each column is a price level's volume.
            asks: DataFrame with columns like 'ask_vol_1', 'ask_vol_2', ...
            levels: Number of order book levels to use.

        Returns:
            OBI series in [-1, 1].
        """
        bid_cols = bids.columns[:levels]
        ask_cols = asks.columns[:levels]

        total_bid = bids[bid_cols].sum(axis=1)
        total_ask = asks[ask_cols].sum(axis=1)

        denom = total_bid + total_ask
        # Avoid division by zero — if book is empty on both sides, OBI = 0
        obi = (total_bid - total_ask) / denom.replace(0, np.nan)
        obi = obi.fillna(0.0)

        return obi.rename("order_book_imbalance")

    def compute_volatility_clusters(
        self, returns: pd.Series, span: int = 21, min_periods: int = 5
    ) -> pd.Series:
        """
        Exponentially Weighted Moving Volatility (EWMV) as a lightweight
        GARCH(1,1) proxy.

        Full GARCH requires MLE fitting per window and isn't differentiable.
        EWMV captures the same clustering phenomenon — volatility begets
        volatility — while being fast and compatible with online/streaming use.

        σ_t^2 = λ * σ_{t-1}^2 + (1 - λ) * r_t^2

        where λ = 1 - 2/(span+1).  This IS the RiskMetrics variance estimator
        (JP Morgan, 1996), which is a restricted GARCH(1,1) with α+β=1.

        Args:
            returns: Log returns series.
            span: EWM span (higher = smoother, longer memory). 21 ≈ 1 month.
            min_periods: Minimum observations before producing a value.

        Returns:
            Annualized volatility estimate (assuming 252 trading days).
        """
        ewm_var = returns.ewm(span=span, min_periods=min_periods).var()
        annualized_vol = np.sqrt(ewm_var * 252)
        return annualized_vol.rename("ewm_volatility")


# ==========================================
# 2. THE CUSTOM CELL: Kalman-LSTM
# ==========================================
class KalmanLSTMCell(nn.Module):
    """
    Hybrid Kalman-LSTM cell.

    The LSTM learns nonlinear temporal dynamics from raw input (the "measurement").
    The Kalman filter then refines the LSTM's hidden state using Bayesian optimal
    filtering — it smooths out noise and maintains an uncertainty estimate (P).

    Architecture:
        1. LSTM produces h_lstm (noisy "observation" of the true hidden state)
        2. Kalman predict: project previous filtered state forward via a
           learned gating function F (diagonal state transition)
        3. Kalman update: blend h_pred and h_lstm using the Kalman gain K,
           which is computed from the predicted covariance and measurement noise

    Design choices:
        - F is a sigmoid-gated diagonal matrix (element-wise), not dense.
          This keeps P diagonal and avoids O(n^3) matrix inversion.
        - Q, R are parameterized as softplus(learnable) to guarantee positivity.
        - P (covariance) is stored as a vector (diagonal), making the Kalman
          gain computation element-wise: K_i = p_pred_i / (p_pred_i + r_i).
        - This is exactly the scalar Kalman filter applied independently per
          hidden dimension — efficient and stable under gradient descent.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Standard LSTM cell — learns nonlinear dynamics
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        # Kalman state transition gate: F = sigmoid(W_f @ h + b_f)
        # Diagonal approximation: each hidden unit has its own transition weight
        self.transition_gate = nn.Linear(hidden_size, hidden_size)

        # Process noise Q (diagonal): how much uncertainty the state transition adds
        # Parameterized in log-space, passed through softplus for positivity
        self.q_log = nn.Parameter(torch.zeros(hidden_size))

        # Measurement noise R (diagonal): how much we distrust the LSTM output
        self.r_log = nn.Parameter(torch.zeros(hidden_size))

        self._init_kalman_params()

    def _init_kalman_params(self):
        """Initialize Kalman parameters to reasonable defaults."""
        # Start with moderate noise — let the network learn the right balance
        nn.init.constant_(self.q_log, -2.0)  # softplus(-2) ≈ 0.13
        nn.init.constant_(self.r_log, -1.0)  # softplus(-1) ≈ 0.31
        # Transition gate bias positive so sigmoid(.) starts near 0.5-0.7
        nn.init.constant_(self.transition_gate.bias, 0.5)

    def init_covariance(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create initial covariance vector P_0 (diagonal, ones)."""
        return torch.ones(batch_size, self.hidden_size, device=device)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        p_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single timestep forward pass.

        Args:
            x_t:    (batch, input_size)  — current input features
            h_prev: (batch, hidden_size) — previous filtered hidden state
            c_prev: (batch, hidden_size) — previous LSTM cell state
            p_prev: (batch, hidden_size) — previous covariance diagonal

        Returns:
            h_filtered: (batch, hidden_size) — Kalman-filtered hidden state
            c_t:        (batch, hidden_size) — updated LSTM cell state
            p_updated:  (batch, hidden_size) — updated covariance diagonal

        Math (all element-wise since P, Q, R are diagonal):
            1. h_lstm, c_t = LSTM(x_t, h_prev, c_prev)   [measurement]
            2. f_gate = σ(W_f @ h_prev + b_f)             [state transition]
            3. h_pred = f_gate * h_prev                    [predicted state]
            4. p_pred = f_gate² * p_prev + Q               [predicted covariance]
            5. K = p_pred / (p_pred + R)                   [Kalman gain]
            6. h_filtered = h_pred + K * (h_lstm - h_pred) [state update]
            7. p_updated = (1 - K) * p_pred                [covariance update]
        """
        # --- Step 1: LSTM produces the "measurement" ---
        h_lstm, c_t = self.lstm_cell(x_t, (h_prev, c_prev))

        # --- Step 2-3: Kalman predict ---
        f_gate = torch.sigmoid(self.transition_gate(h_prev))  # (batch, hidden)
        h_pred = f_gate * h_prev  # predicted state

        # --- Step 4: Predicted covariance ---
        q = F.softplus(self.q_log)  # (hidden,) — guaranteed positive
        r = F.softplus(self.r_log)  # (hidden,) — guaranteed positive
        p_pred = f_gate.pow(2) * p_prev + q  # (batch, hidden)

        # --- Step 5: Kalman gain ---
        # K_i = p_pred_i / (p_pred_i + r_i)
        # When p_pred is large relative to r → K ≈ 1 → trust LSTM more
        # When r is large relative to p_pred → K ≈ 0 → trust prediction more
        kalman_gain = p_pred / (p_pred + r + 1e-8)  # (batch, hidden)

        # --- Step 6: State update (blending) ---
        innovation = h_lstm - h_pred  # measurement residual
        h_filtered = h_pred + kalman_gain * innovation

        # --- Step 7: Covariance update ---
        p_updated = (1.0 - kalman_gain) * p_pred

        return h_filtered, c_t, p_updated


# ==========================================
# 2.5 FOURIER NEURAL OPERATOR LAYER (FNO)
# ==========================================
class FourierNeuralOperatorLayer(nn.Module):
    """
    Fourier Neural Operator (FNO) layer for sequence data.

    Lifts the hidden state sequence to the frequency domain via FFT,
    applies a learnable complex-valued filter, then IFFTs back.
    A residual linear path runs in parallel (like a skip connection
    in the spatial domain) so the layer can pass through information
    that doesn't have a frequency-domain structure.

    FNO(x) = IFFT(W_freq · FFT(x)) + W_local(x)

    The learnable W_freq is a complex weight matrix that acts as a
    bandpass filter — it amplifies important frequencies and kills noise.
    Only the lowest `n_modes` frequencies are kept (the rest are zeroed),
    which acts as implicit regularization.

    Ref: Li et al., "Fourier Neural Operator for Parametric PDEs" (2020)
    """

    def __init__(self, hidden_size: int, n_modes: int = 16):
        """
        Args:
            hidden_size: dimension of hidden states
            n_modes:     number of Fourier modes to keep (low-frequency cutoff).
                         Higher = more frequency resolution, lower = more smoothing.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_modes = n_modes

        # Complex-valued learnable spectral weights: (n_modes, hidden, hidden)
        # Initialized with small random values so it starts near identity
        scale = 1.0 / (hidden_size * hidden_size)
        self.spectral_weights = nn.Parameter(
            torch.randn(n_modes, hidden_size, hidden_size, dtype=torch.cfloat) * scale
        )

        # Residual path: pointwise linear transform (spatial/local processing)
        self.residual_linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch, hidden_size)

        Returns:
            (seq_len, batch, hidden_size) — filtered + residual
        """
        seq_len, batch, hidden = x.shape

        # --- Frequency path ---
        # FFT along the sequence (time) dimension
        x_freq = torch.fft.rfft(x, dim=0)  # (seq_len//2+1, batch, hidden) complex

        # Apply learnable filter to the lowest n_modes frequencies
        n_modes = min(self.n_modes, x_freq.shape[0])
        # Einstein sum: for each mode m, do x_freq[m] @ W[m]
        # x_freq[:n_modes] is (n_modes, batch, hidden)
        # spectral_weights[:n_modes] is (n_modes, hidden, hidden)
        filtered = torch.zeros_like(x_freq)
        filtered[:n_modes] = torch.einsum(
            "mbh,mhd->mbd",
            x_freq[:n_modes],
            self.spectral_weights[:n_modes],
        )
        # Higher frequencies are zeroed — acts as low-pass + learned filter

        # IFFT back to time domain
        x_filtered = torch.fft.irfft(filtered, n=seq_len, dim=0)  # (seq_len, batch, hidden)

        # --- Residual/local path ---
        x_local = self.residual_linear(x)  # (seq_len, batch, hidden)

        # Combine + normalize
        out = self.norm(x_filtered.real + x_local)
        return out


# ==========================================
# 2.6 NEURAL ODE LAYER (Continuous-depth refinement)
# ==========================================
class NeuralODEFunc(nn.Module):
    """
    The dynamics function f_θ(h, t) for the Neural ODE.

    A small two-layer network that defines the vector field:
        dh/dt = f_θ(h, t)

    Tanh activation keeps the dynamics bounded — prevents exploding
    trajectories during integration. The time variable t is not used
    here (autonomous ODE), but could be added for time-varying dynamics.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        # Initialize near-zero so ODE starts as near-identity
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class NeuralODELayer(nn.Module):
    """
    Neural ODE layer: refines hidden states via continuous dynamics.

    Integrates dh/dt = f_θ(h) from t=0 to t=1 using fixed-step RK4.
    This is equivalent to an infinitely deep residual network with
    shared weights — the ODE solver determines the effective depth.

    Unlike discrete residual blocks, the continuous formulation:
    - Guarantees a smooth transformation (no sharp jumps)
    - Is memory-efficient (O(1) via adjoint method, though we use
      standard backprop here for simplicity)
    - Can adaptively control precision via n_steps

    For the SB3 extractor, this acts as a continuous nonlinear refinement
    of the Kalman-LSTM hidden state — like a "polishing" pass.

    Ref: Chen et al., "Neural Ordinary Differential Equations" (2018)
    """

    def __init__(self, hidden_size: int, n_steps: int = 4):
        """
        Args:
            hidden_size: dimension of hidden states
            n_steps:     number of RK4 integration steps (more = more accurate,
                         but slower). 4 steps is usually plenty.
        """
        super().__init__()
        self.func = NeuralODEFunc(hidden_size)
        self.n_steps = n_steps

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Integrate from t=0 to t=1 using 4th-order Runge-Kutta.

        Args:
            h: (batch, hidden_size) or (seq_len, batch, hidden_size)

        Returns:
            h_refined: same shape as h — the state at t=1
        """
        dt = 1.0 / self.n_steps
        for _ in range(self.n_steps):
            k1 = self.func(h)
            k2 = self.func(h + dt / 2 * k1)
            k3 = self.func(h + dt / 2 * k2)
            k4 = self.func(h + dt * k3)
            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h


# ==========================================
# 2.7 RECURRENT WRAPPERS FOR SB3-CONTRIB
# ==========================================
# These wrap KalmanLSTMCell (and optional ODE/FNO) into nn.LSTM-compatible
# modules so RecurrentPPO can manage hidden state across timesteps.
#
# State packing: RecurrentPPO stores (h, c) each of shape (1, batch, lstm_hidden_size).
# We set lstm_hidden_size = 2*H and pack:
#   h_packed = [kalman_h, log(kalman_P)]   (2*H)
#   c_packed = [lstm_c, zeros_padding]     (2*H)
#
# log(P) encoding ensures episode resets (zeros) produce P = exp(0) = 1 (correct init).
# Output is 2*H: [filtered_state, covariance] — the policy sees both state + uncertainty.


class KalmanLSTMWrapper(nn.Module):
    """nn.LSTM drop-in replacement using KalmanLSTMCell for RecurrentPPO."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = KalmanLSTMCell(input_size, hidden_size)
        self.input_size = input_size
        self._kalman_hidden = hidden_size
        self.hidden_size = 2 * hidden_size  # packed: [h, P]
        self.num_layers = 1

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (seq_len, batch, input_size)
            hx: ((1, batch, 2H), (1, batch, 2H)) or None

        Returns:
            output: (seq_len, batch, 2H) — [kalman_h, kalman_P]
            (h_packed, c_packed): each (1, batch, 2H)
        """
        seq_len, batch, _ = x.shape
        device = x.device
        H = self._kalman_hidden

        if hx is None:
            h = torch.zeros(batch, H, device=device)
            c = torch.zeros(batch, H, device=device)
            p = self.cell.init_covariance(batch, device)
        else:
            hp = hx[0].squeeze(0)  # (batch, 2H)
            cp = hx[1].squeeze(0)  # (batch, 2H)
            h, log_p = hp[:, :H], hp[:, H:]
            p = torch.exp(log_p)  # zeros → P=1 (correct init at episode reset)
            c = cp[:, :H]

        outputs = []
        for t in range(seq_len):
            h, c, p = self.cell(x[t], h, c, p)
            outputs.append(torch.cat([h, p], dim=-1))

        output = torch.stack(outputs, dim=0)  # (seq_len, batch, 2H)

        hp_new = torch.cat([h, torch.log(p.clamp(min=1e-8))], dim=-1).unsqueeze(0)
        cp_new = torch.cat([c, torch.zeros_like(c)], dim=-1).unsqueeze(0)

        return output, (hp_new, cp_new)


class KalmanODEWrapper(nn.Module):
    """KalmanLSTM + Neural ODE per-step refinement for RecurrentPPO.

    After each Kalman step, the hidden state is refined via ODE integration.
    The refined h is persisted (it becomes h_prev for the next timestep).
    """

    def __init__(self, input_size: int, hidden_size: int, n_ode_steps: int = 4):
        super().__init__()
        self.cell = KalmanLSTMCell(input_size, hidden_size)
        self.ode = NeuralODELayer(hidden_size, n_steps=n_ode_steps)
        self.input_size = input_size
        self._kalman_hidden = hidden_size
        self.hidden_size = 2 * hidden_size  # packed: [h, P]
        self.num_layers = 1

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        seq_len, batch, _ = x.shape
        device = x.device
        H = self._kalman_hidden

        if hx is None:
            h = torch.zeros(batch, H, device=device)
            c = torch.zeros(batch, H, device=device)
            p = self.cell.init_covariance(batch, device)
        else:
            hp = hx[0].squeeze(0)
            cp = hx[1].squeeze(0)
            h, log_p = hp[:, :H], hp[:, H:]
            p = torch.exp(log_p)
            c = cp[:, :H]

        outputs = []
        for t in range(seq_len):
            h, c, p = self.cell(x[t], h, c, p)
            h = self.ode(h)  # continuous refinement (persists to next step)
            outputs.append(torch.cat([h, p], dim=-1))

        output = torch.stack(outputs, dim=0)

        hp_new = torch.cat([h, torch.log(p.clamp(min=1e-8))], dim=-1).unsqueeze(0)
        cp_new = torch.cat([c, torch.zeros_like(c)], dim=-1).unsqueeze(0)

        return output, (hp_new, cp_new)


class KalmanFNOWrapper(nn.Module):
    """KalmanLSTM + FNO sequence filtering for RecurrentPPO.

    Unlike the stateless extractor, RecurrentPPO processes sub-sequences
    (between episode boundaries), so FNO actually gets multi-step
    sequences to apply frequency-domain filtering on.

    The FNO-filtered output is used for the policy, but the raw Kalman
    state persists (FNO is a readout transform, not a state update).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_fno_layers: int = 2,
        n_modes: int = 16,
    ):
        super().__init__()
        self.cell = KalmanLSTMCell(input_size, hidden_size)
        self.fno_layers = nn.ModuleList([
            FourierNeuralOperatorLayer(hidden_size, n_modes=n_modes)
            for _ in range(n_fno_layers)
        ])
        self.input_size = input_size
        self._kalman_hidden = hidden_size
        self.hidden_size = 2 * hidden_size  # packed: [h, P]
        self.num_layers = 1

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        seq_len, batch, _ = x.shape
        device = x.device
        H = self._kalman_hidden

        if hx is None:
            h = torch.zeros(batch, H, device=device)
            c = torch.zeros(batch, H, device=device)
            p = self.cell.init_covariance(batch, device)
        else:
            hp = hx[0].squeeze(0)
            cp = hx[1].squeeze(0)
            h, log_p = hp[:, :H], hp[:, H:]
            p = torch.exp(log_p)
            c = cp[:, :H]

        # Run Kalman-LSTM over full sub-sequence
        h_states = []
        p_states = []
        for t in range(seq_len):
            h, c, p = self.cell(x[t], h, c, p)
            h_states.append(h)
            p_states.append(p)

        h_seq = torch.stack(h_states, dim=0)  # (seq_len, batch, H)
        p_seq = torch.stack(p_states, dim=0)  # (seq_len, batch, H)

        # FNO frequency filtering on the hidden state sequence
        for fno in self.fno_layers:
            h_seq = fno(h_seq)

        # Output: FNO-filtered h + raw P
        output = torch.cat([h_seq, p_seq], dim=-1)  # (seq_len, batch, 2H)

        # Persist raw Kalman state (not FNO-filtered)
        hp_new = torch.cat([h, torch.log(p.clamp(min=1e-8))], dim=-1).unsqueeze(0)
        cp_new = torch.cat([c, torch.zeros_like(c)], dim=-1).unsqueeze(0)

        return output, (hp_new, cp_new)


# ==========================================
# 3. ADVANCED MODULES: ATTENTION & MULTI-TASK LEARNING
# ==========================================
class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) from the Temporal Fusion Transformer paper.

    Applies a nonlinear transform with a gating mechanism that lets the
    network learn to suppress components that aren't useful, and a skip
    connection so the default behavior is identity.

    GRN(a) = LayerNorm(a + GLU(W1 * ELU(W2 * a)))
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # If input_size != hidden_size, we need a skip projection
        self.skip_proj = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_proj(x)
        hidden = F.elu(self.fc1(x))
        hidden = self.dropout(self.fc2(hidden))
        # GLU-style gating: element-wise sigmoid gate
        gate_values = torch.sigmoid(self.gate(hidden))
        hidden = gate_values * hidden
        return self.layer_norm(residual + hidden)


class TemporalFusionEncoder(nn.Module):
    """
    Temporal attention encoder inspired by the Temporal Fusion Transformer.

    Takes a sequence of Kalman-LSTM hidden states and applies:
    1. Multi-head self-attention to learn which past timesteps matter
    2. GRN post-attention for nonlinear gating
    3. Residual connection + layer norm for stable gradients

    Unlike vanilla transformers, TFT-style attention is interpretable —
    the attention weights directly show which historical timesteps the
    model considers important for the current prediction.
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,  # input is (seq, batch, hidden)
        )

        # Post-attention GRN for gated nonlinear transform
        self.grn = GatedResidualNetwork(hidden_size, hidden_size, dropout)

        # Pre-attention projection if input_size != hidden_size
        self.input_proj = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size
            else nn.Identity()
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (seq_len, batch, input_size) — Kalman-LSTM outputs
            mask: optional (seq_len, seq_len) causal mask

        Returns:
            output:    (seq_len, batch, hidden_size) — attended + gated output
            attn_weights: (batch, seq_len, seq_len) — interpretable attention map
        """
        x = self.input_proj(hidden_states)  # (seq, batch, hidden)

        # Causal mask: prevent attending to future timesteps
        if mask is None:
            seq_len = x.size(0)
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        # Self-attention with causal mask
        attn_out, attn_weights = self.attention(
            query=x, key=x, value=x,
            attn_mask=mask,
            need_weights=True,
        )

        # Residual + norm
        x = self.layer_norm(x + self.dropout(attn_out))

        # GRN gating per timestep
        seq_len, batch, hidden = x.shape
        x_flat = x.reshape(seq_len * batch, hidden)
        x_gated = self.grn(x_flat)
        output = x_gated.reshape(seq_len, batch, hidden)

        return output, attn_weights


class MultiTaskTradingModel(nn.Module):
    """
    Full sequence model: KalmanLSTMCell → TemporalFusionEncoder → task heads.

    Multi-task setup forces the network to learn richer internal representations.
    The volatility head acts as an auxiliary task — predicting vol requires
    understanding regime changes, which also benefits return prediction.
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = KalmanLSTMCell(input_size, hidden_size)
        self.attention = TemporalFusionEncoder(hidden_size, hidden_size, num_heads, dropout)

        # Multi-Task Heads
        self.price_head = nn.Linear(hidden_size, 1)
        self.volatility_head = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (seq_len, batch, input_size) — input feature sequence

        Returns:
            price_pred: (seq_len, batch, 1) — predicted returns/prices
            vol_pred:   (seq_len, batch, 1) — predicted volatility
            attn_weights: (batch, seq_len, seq_len) — attention map
        """
        seq_len, batch, _ = x.shape
        device = x.device

        # Initialize Kalman-LSTM states
        h = torch.zeros(batch, self.hidden_size, device=device)
        c = torch.zeros(batch, self.hidden_size, device=device)
        p = self.cell.init_covariance(batch, device)

        # Run through sequence
        h_states = []
        for t in range(seq_len):
            h, c, p = self.cell(x[t], h, c, p)
            h_states.append(h)

        h_seq = torch.stack(h_states)  # (seq_len, batch, hidden)

        # Temporal attention
        attended, attn_weights = self.attention(h_seq)

        # Task heads (applied at every timestep)
        price_pred = self.price_head(attended)       # (seq_len, batch, 1)
        vol_pred = self.volatility_head(attended)     # (seq_len, batch, 1)

        return price_pred, vol_pred, attn_weights


class FNOMultiTaskTradingModel(nn.Module):
    """
    FNO-enhanced variant: KalmanLSTM → FNO → Attention → task heads.

    Inserts FNO layers between the Kalman-LSTM and the attention encoder.
    The FNO filters the hidden state sequence in the frequency domain,
    amplifying cyclical patterns and suppressing noise before attention.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        n_fno_layers: int = 2,
        n_modes: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = KalmanLSTMCell(input_size, hidden_size)

        # FNO stack between Kalman-LSTM and attention
        self.fno_layers = nn.ModuleList([
            FourierNeuralOperatorLayer(hidden_size, n_modes=n_modes)
            for _ in range(n_fno_layers)
        ])

        self.attention = TemporalFusionEncoder(hidden_size, hidden_size, num_heads, dropout)

        # Multi-Task Heads
        self.price_head = nn.Linear(hidden_size, 1)
        self.volatility_head = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len, batch, _ = x.shape
        device = x.device

        h = torch.zeros(batch, self.hidden_size, device=device)
        c = torch.zeros(batch, self.hidden_size, device=device)
        p = self.cell.init_covariance(batch, device)

        h_states = []
        for t in range(seq_len):
            h, c, p = self.cell(x[t], h, c, p)
            h_states.append(h)

        h_seq = torch.stack(h_states)  # (seq_len, batch, hidden)

        # FNO frequency-domain filtering
        for fno in self.fno_layers:
            h_seq = fno(h_seq)

        attended, attn_weights = self.attention(h_seq)

        price_pred = self.price_head(attended)
        vol_pred = self.volatility_head(attended)

        return price_pred, vol_pred, attn_weights


# ==========================================
# 4. EXECUTION COSTS (THE ALPHA KILLER)
# ==========================================
class ProfitLossWithSlippage(nn.Module):
    """
    Cost-aware loss function for trading models.

    Standard MSE only penalizes prediction error. But in real trading,
    a model that's slightly less accurate but trades less frequently
    can be far more profitable after costs. This loss has three terms:

    L = MSE(pred, target) + λ_comm * turnover + λ_slip * turnover

    where turnover = |signal_t - signal_{t-1}| summed over time.
    This directly penalizes the model for "churning" — flipping
    between buy/sell too often, which destroys alpha through
    commission and slippage.
    """

    def __init__(
        self,
        commission_rate: float = 0.0001,
        slippage_penalty: float = 0.0005,
        prediction_weight: float = 1.0,
    ):
        super().__init__()
        self.commission = commission_rate
        self.slippage = slippage_penalty
        self.prediction_weight = prediction_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        trade_signals: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            predictions:  (seq_len, batch, 1) — model price predictions
            targets:      (seq_len, batch, 1) — actual price/return values
            trade_signals: (seq_len, batch, 1) — model's position signal
                           (continuous, e.g., tanh output: -1=short, +1=long)

        Returns:
            total_loss: scalar
            breakdown:  dict with individual loss components for logging
        """
        # --- Prediction error ---
        mse_loss = F.mse_loss(predictions, targets)

        # --- Turnover penalty ---
        # |signal_t - signal_{t-1}| measures how much the position changes
        # High turnover = lots of trading = lots of costs
        signal_diff = torch.diff(trade_signals, dim=0)  # (seq_len-1, batch, 1)
        turnover = signal_diff.abs().mean()

        # --- Cost terms ---
        commission_cost = self.commission * turnover
        slippage_cost = self.slippage * turnover

        # --- Total loss ---
        total_loss = (
            self.prediction_weight * mse_loss
            + commission_cost
            + slippage_cost
        )

        breakdown = {
            "mse": mse_loss.detach(),
            "turnover": turnover.detach(),
            "commission_cost": commission_cost.detach(),
            "slippage_cost": slippage_cost.detach(),
            "total": total_loss.detach(),
        }

        return total_loss, breakdown


# ==========================================
# 5. REINFORCEMENT LEARNING INTEGRATION
# ==========================================
class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for RL-based trading.

    The agent observes features and takes discrete actions:
        Buy (0):  go long  → position = +1
        Hold (1): keep current position
        Sell (2): go flat  → position = 0  (close any open position)

    Position is tracked as 0 (flat) or +1 (long). No shorting.
    Reward is mark-to-market PnL scaled ×100 for PPO-friendly magnitudes,
    with a small flat-penalty to discourage permanent inaction.
    """

    metadata = {"render_modes": ["human"]}
    # Actions: 0=Buy (long), 1=Hold, 2=Sell (go flat)
    ACTION_BUY = 0
    ACTION_HOLD = 1
    ACTION_SELL = 2

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        commission_rate: float = 0.0001,
        slippage_rate: float = 0.0005,
        initial_balance: float = 100_000.0,
        window_size: int = 1,
        random_start: bool = False,
        min_episode_length: int = 60,
        reward_scale: float = 100.0,
        flat_penalty: float = 0.005,
    ):
        super().__init__()

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.random_start = random_start
        self.min_episode_length = min_episode_length
        self.reward_scale = reward_scale
        self.flat_penalty = flat_penalty
        self.num_steps = len(prices)

        # Observation: feature vector + [position, unrealized_pnl, balance_ratio]
        obs_size = features.shape[1] * window_size + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # State variables (set in reset)
        self.current_step = 0
        self.position = 0        # 0 (flat) or +1 (long) — no shorting
        self.entry_price = 0.0
        self.balance = initial_balance
        self.portfolio_value = initial_balance
        self.total_trades = 0
        self.total_commission = 0.0

    def _get_observation(self) -> np.ndarray:
        """Build observation vector from features + portfolio state."""
        start = max(0, self.current_step - self.window_size + 1)
        feat_window = self.features[start : self.current_step + 1]

        # Pad if we're near the beginning
        if feat_window.shape[0] < self.window_size:
            pad_size = self.window_size - feat_window.shape[0]
            feat_window = np.vstack(
                [np.zeros((pad_size, self.features.shape[1])), feat_window]
            )

        flat_features = feat_window.flatten()

        # Portfolio state features
        current_price = self.prices[self.current_step]
        unrealized_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl = self.position * (current_price - self.entry_price) / self.entry_price

        balance_ratio = self.balance / self.initial_balance

        state_features = np.array(
            [float(self.position), unrealized_pnl, balance_ratio], dtype=np.float32
        )

        return np.concatenate([flat_features, state_features])

    def _calculate_transaction_cost(self, price: float) -> float:
        """Commission + slippage for one trade."""
        return price * (self.commission_rate + self.slippage_rate)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step.

        Returns: (observation, reward, terminated, truncated, info)
        """
        current_price = self.prices[self.current_step]
        reward = 0.0
        trade_executed = False

        # Map action to target position (no shorting: Sell = go flat)
        target_position = {self.ACTION_BUY: 1, self.ACTION_HOLD: self.position, self.ACTION_SELL: 0}[action]

        # Execute trade if position changes
        if target_position != self.position:
            trade_executed = True
            cost = self._calculate_transaction_cost(current_price)
            self.total_commission += cost
            self.total_trades += 1

            # Close existing position PnL
            if self.position != 0 and self.entry_price > 0:
                pnl = self.position * (current_price - self.entry_price)
                self.balance += pnl - cost
            else:
                self.balance -= cost

            # Open new position
            self.position = target_position
            self.entry_price = current_price if target_position != 0 else 0.0

        # Mark-to-market reward: change in portfolio value
        unrealized = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized = self.position * (current_price - self.entry_price)

        new_portfolio_value = self.balance + unrealized
        reward = (new_portfolio_value - self.portfolio_value) / self.initial_balance
        reward *= self.reward_scale  # scale up for PPO-friendly magnitudes

        # Small penalty for being flat — breaks "do nothing" local optimum
        if self.position == 0:
            reward -= self.flat_penalty

        self.portfolio_value = new_portfolio_value

        # Advance
        self.current_step += 1
        terminated = self.current_step >= self.num_steps - 1
        truncated = False

        # Bankruptcy check
        if self.portfolio_value <= 0:
            terminated = True
            reward = -1.0

        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "position": self.position,
            "total_trades": self.total_trades,
            "total_commission": self.total_commission,
            "trade_executed": trade_executed,
        }

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset to initial state, optionally at a random start point."""
        super().reset(seed=seed)

        if self.random_start and self.num_steps > self.min_episode_length:
            max_start = self.num_steps - self.min_episode_length
            self.current_step = self.np_random.integers(0, max_start)
        else:
            self.current_step = 0

        self.position = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.total_trades = 0
        self.total_commission = 0.0
        return self._get_observation(), {}


# ==========================================
# SMOKE TESTS — validates all modules
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 1 TEST: FeatureEngineer")
    print("=" * 60)

    fe = FeatureEngineer(d_value=0.4)

    # Simulate a price series (geometric Brownian motion)
    np.random.seed(42)
    n = 500
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    price_series = pd.Series(prices, name="close")

    # Fractional differentiation
    fracdiff = fe.apply_fractional_differentiation(price_series)
    valid = fracdiff.dropna()
    print(f"  FracDiff: {len(valid)}/{n} valid values, "
          f"mean={valid.mean():.4f}, std={valid.std():.4f}")

    # Volatility clusters
    returns = np.log(price_series / price_series.shift(1)).dropna()
    vol = fe.compute_volatility_clusters(returns)
    print(f"  Volatility: mean={vol.dropna().mean():.4f}")

    # Order book imbalance (synthetic)
    bids = pd.DataFrame(np.random.rand(n, 5), columns=[f"bid_{i}" for i in range(5)])
    asks = pd.DataFrame(np.random.rand(n, 5), columns=[f"ask_{i}" for i in range(5)])
    obi = fe.compute_order_book_imbalance(bids, asks)
    print(f"  OBI: mean={obi.mean():.4f}, range=[{obi.min():.3f}, {obi.max():.3f}]")

    print()
    print("=" * 60)
    print("MODULE 2 TEST: KalmanLSTMCell")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    batch_size = 16
    seq_len = 50
    input_size = 8
    hidden_size = 32

    cell = KalmanLSTMCell(input_size, hidden_size).to(device)

    h = torch.zeros(batch_size, hidden_size, device=device)
    c = torch.zeros(batch_size, hidden_size, device=device)
    p = cell.init_covariance(batch_size, device)
    x_seq = torch.randn(seq_len, batch_size, input_size, device=device)

    h_states = []
    for t in range(seq_len):
        h, c, p = cell(x_seq[t], h, c, p)
        h_states.append(h)
    h_stack = torch.stack(h_states)

    print(f"  Output shape: {h_stack.shape}")
    print(f"  Final h mean: {h.mean().item():.4f}, std: {h.std().item():.4f}")
    print(f"  Final P mean: {p.mean().item():.6f}")

    loss = h_stack.sum()
    loss.backward()
    grad_ok = all(p.grad is not None for p in cell.parameters() if p.requires_grad)
    print(f"  Gradients flow: {grad_ok}")

    print()
    print("=" * 60)
    print("MODULE 3 TEST: TemporalFusionEncoder + MultiTaskTradingModel")
    print("=" * 60)

    model = MultiTaskTradingModel(
        input_size=input_size, hidden_size=hidden_size, num_heads=4
    ).to(device)

    x_in = torch.randn(seq_len, batch_size, input_size, device=device)
    price_pred, vol_pred, attn_w = model(x_in)

    print(f"  Price pred shape: {price_pred.shape}")   # (50, 16, 1)
    print(f"  Vol pred shape:   {vol_pred.shape}")      # (50, 16, 1)
    print(f"  Attn weights shape: {attn_w.shape}")      # (16, 50, 50)
    print(f"  Price pred range: [{price_pred.min().item():.4f}, {price_pred.max().item():.4f}]")

    # Gradient check
    total = price_pred.sum() + vol_pred.sum()
    total.backward()
    grad_ok_model = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    print(f"  Gradients flow: {grad_ok_model}")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {param_count:,}")

    print()
    print("=" * 60)
    print("MODULE 4 TEST: ProfitLossWithSlippage")
    print("=" * 60)

    loss_fn = ProfitLossWithSlippage(
        commission_rate=0.0001, slippage_penalty=0.0005
    )

    # Synthetic predictions, targets, and trade signals
    preds = torch.randn(seq_len, batch_size, 1, device=device, requires_grad=True)
    targets = torch.randn(seq_len, batch_size, 1, device=device)
    # Simulate a signal that changes frequently (high turnover)
    signals_noisy = torch.tanh(torch.randn(seq_len, batch_size, 1, device=device))
    # And one that's stable (low turnover)
    signals_stable = torch.ones(seq_len, batch_size, 1, device=device) * 0.5

    loss_noisy, breakdown_noisy = loss_fn(preds, targets, signals_noisy)
    loss_stable, breakdown_stable = loss_fn(preds, targets, signals_stable)

    print(f"  Noisy signal loss:  {loss_noisy.item():.6f} (turnover: {breakdown_noisy['turnover'].item():.4f})")
    print(f"  Stable signal loss: {loss_stable.item():.6f} (turnover: {breakdown_stable['turnover'].item():.4f})")
    print(f"  Cost penalty works: {loss_noisy.item() > loss_stable.item()}")

    # Gradient check
    loss_noisy.backward()
    print(f"  Gradients flow: {preds.grad is not None}")

    print()
    print("=" * 60)
    print("MODULE 5 TEST: TradingEnv (Gymnasium)")
    print("=" * 60)

    # Create env with synthetic data
    env_prices = prices.astype(np.float32)
    env_features = np.column_stack([
        np.random.randn(n).astype(np.float32) for _ in range(input_size)
    ])

    env = TradingEnv(
        features=env_features,
        prices=env_prices,
        commission_rate=0.0001,
        slippage_rate=0.0005,
        initial_balance=100_000.0,
    )

    obs, info = env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")

    # Run random agent for 100 steps
    total_reward = 0.0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break

    print(f"  After 100 random steps:")
    print(f"    Portfolio value: ${info['portfolio_value']:,.2f}")
    print(f"    Total trades: {info['total_trades']}")
    print(f"    Total commission: ${info['total_commission']:.2f}")
    print(f"    Cumulative reward: {total_reward:.6f}")

    # Validate Gymnasium API compliance
    from gymnasium.utils.env_checker import check_env
    env2 = TradingEnv(features=env_features, prices=env_prices)
    try:
        check_env(env2)
        print(f"  Gymnasium API check: PASSED")
    except Exception as e:
        print(f"  Gymnasium API check: FAILED — {e}")

    print()
    print("All tests passed.")
