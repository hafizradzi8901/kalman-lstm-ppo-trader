# Kalman-LSTM + RecurrentPPO — an uncertainty-aware RL trader

A research prototype that treats intraday trading as a **partially-observed reinforcement-learning**
problem, and tackles the two things that usually break naive ML traders: **non-stationarity** and
**uncertainty**. Built from scratch in PyTorch on top of `stable-baselines3` / `sb3-contrib`.

> ⚠️ Research / educational project — **not financial advice** and not a profitable system.
> The interesting part is the architecture and the methodology, not a P&L claim.

---

## The ideas, and why each one is here

**1. Fractional differentiation for the "stationarity trap"**
Integer differencing (returns, `d=1`) makes a price series stationary but **erases its memory**.
Following López de Prado's *Advances in Financial Machine Learning*, the `FeatureEngineer` uses
**fractional differentiation** (`d ≈ 0.3–0.7`) — the minimum differencing that passes a stationarity
test while preserving long-range dependence. Weights come from the binomial expansion of `(1−B)^d`.

**2. A Kalman-LSTM cell — an LSTM that knows what it doesn't know**
A custom recurrent cell that augments the LSTM state `(h, c)` with a **covariance `P`** and a
Bayesian predict/update step. Instead of emitting a point estimate, it carries an explicit estimate
of its own uncertainty forward in time — the right inductive bias for a noisy, low-signal domain.

**3. RL, not regression — with persistent recurrent state**
The Kalman-LSTM is a `BaseFeaturesExtractor` feeding a **RecurrentPPO** agent (`sb3-contrib`), so the
hidden state `(h, c, P)` **persists across timesteps within an episode** — the agent reasons over
sequences, and the reward function bakes in **execution costs** so it can't farm frictionless profits.

**4. Three architectural ablations (all implemented)**
The training pipeline (`train_ppo_v3.py`) runs the same agent with three encoders:

| Variant | What it adds |
|---|---|
| `baseline` | Kalman-LSTM cell with persistent Bayesian state |
| `ode` | + a **Neural ODE** per-step refinement (continuous-depth state evolution) |
| `fno` | + a **Fourier Neural Operator** filter over the sequence (learnable frequency bandpass) |

---

## Architecture

```
OHLCV ──▶ FeatureEngineer ──▶ TradingEnv ──▶ RecurrentPPO agent
        (fractional diff,    (gymnasium,        │
         technical features)  execution costs)  ▼
                                    Kalman-LSTM features extractor
                                    (persistent h, c, P)  ± Neural-ODE / FNO
```

| File | Responsibility |
|---|---|
| `kalman_lstm.py` | `FeatureEngineer`, the `KalmanLSTMCell`, attention/multi-task heads, and the `TradingEnv` |
| `data_pipeline.py` | OHLCV fetch (yfinance) → feature matrix ready for the env |
| `train_ppo_v3.py` | **Current** training + ablation runner (baseline / ode / fno), eval mode |
| `train_ppo.py` | Earlier training script (kept for history) |

## Run it

```bash
pip install -r requirements.txt

python train_ppo_v3.py                                   # baseline
python train_ppo_v3.py --model ode  --total-steps 200000 # Neural-ODE ablation
python train_ppo_v3.py --model fno  --total-steps 200000 # FNO ablation
python train_ppo_v3.py --tickers SPY QQQ IWM             # multi-ticker
python train_ppo_v3.py --eval --model-path runs/<run>/best_model
```

Backtesting pulls free data from Yahoo Finance (`yfinance`). Live/paper trading via Alpaca is
optional — copy `.env.example` to `.env` and add your own keys (the real `.env` is git-ignored).

## License

MIT — see [LICENSE](LICENSE).
