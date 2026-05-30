"""
RecurrentPPO Training Pipeline -- Baseline + Ablation Studies (v3)

Uses sb3-contrib RecurrentPPO for proper LSTM state management.
The Kalman-LSTM hidden state (h, c, P) now persists across timesteps
within each episode — enabling real temporal reasoning.

Ablation variants:
  baseline: KalmanLSTMCell (persistent state, Bayesian filtering)
  ode:      KalmanLSTMCell + Neural ODE per-step refinement
  fno:      KalmanLSTMCell + FNO sequence filtering

Usage:
    python train_ppo.py                                    # baseline
    python train_ppo.py --model ode --total-steps 200000   # ODE ablation
    python train_ppo.py --model fno --total-steps 200000   # FNO ablation
    python train_ppo.py --tickers SPY QQQ IWM              # multi-ticker
    python train_ppo.py --eval --model-path runs/xxx/best_model
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from data_pipeline import fetch_ohlcv, build_features, train_test_split
from kalman_lstm import (
    TradingEnv,
    KalmanLSTMWrapper,
    KalmanODEWrapper,
    KalmanFNOWrapper,
)


# ============================================================
# Custom recurrent policy using Kalman-LSTM wrappers
# ============================================================
class KalmanRecurrentPolicy(RecurrentActorCriticPolicy):
    """RecurrentPPO policy that replaces the vanilla LSTM with a
    Kalman-LSTM wrapper (baseline, ODE, or FNO variant).

    State packing: lstm_hidden_size = 2*H because we pack both the
    Kalman filtered state and covariance into the LSTM state tuple.
    The policy MLP receives [h, P] so it can see both the state
    estimate and the model's uncertainty — useful for risk-aware trading.
    """

    def __init__(self, *args, **kwargs):
        # Pop our custom kwargs before passing to parent
        kalman_wrapper_cls = kwargs.pop("kalman_wrapper_cls", KalmanLSTMWrapper)
        kalman_hidden = kwargs.pop("kalman_hidden_size", 64)
        kalman_wrapper_kwargs = kwargs.pop("kalman_wrapper_kwargs", {})
        # Wrapper.hidden_size is already 2*H (packed), so pass it directly
        kwargs["lstm_hidden_size"] = 2 * kalman_hidden
        kwargs.setdefault("shared_lstm", True)
        kwargs.setdefault("n_lstm_layers", 1)
        # shared_lstm=True requires enable_critic_lstm=False
        if kwargs.get("shared_lstm", False):
            kwargs["enable_critic_lstm"] = False
        # Let parent build everything (including vanilla nn.LSTM)
        super().__init__(*args, **kwargs)
        # NOW swap the nn.LSTM(s) with our Kalman wrapper
        self.lstm_actor = kalman_wrapper_cls(
            self.features_dim, kalman_hidden,
            **kalman_wrapper_kwargs,
        )
        if self.lstm_critic is not None:
            self.lstm_critic = kalman_wrapper_cls(
                self.features_dim, kalman_hidden,
                **kalman_wrapper_kwargs,
            )
        # Re-setup optimizer so it picks up the new Kalman parameters
        lr_schedule = kwargs.get("lr_schedule", None)
        if lr_schedule is None:
            # args[2] is lr_schedule in the positional args
            lr_schedule = args[2] if len(args) > 2 else lambda _: 3e-4
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


# ============================================================
# Trading metrics callback
# ============================================================
class TradingMetricsCallback(BaseCallback):
    def __init__(self, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_trades = []
        self.episode_portfolio_values = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
            if "portfolio_value" in info:
                self.episode_portfolio_values.append(info["portfolio_value"])
            if "total_trades" in info:
                self.episode_trades.append(info["total_trades"])

        if self.num_timesteps % self.log_freq == 0 and len(self.episode_rewards) > 0:
            n = min(20, len(self.episode_rewards))
            mean_reward = np.mean(self.episode_rewards[-n:])
            mean_trades = np.mean(self.episode_trades[-n:]) if self.episode_trades else 0
            mean_pv = np.mean(self.episode_portfolio_values[-n:]) if self.episode_portfolio_values else 0

            self.logger.record("trading/mean_reward_20ep", mean_reward)
            self.logger.record("trading/mean_trades_20ep", mean_trades)
            self.logger.record("trading/mean_portfolio_value_20ep", mean_pv)

            if self.verbose:
                print(
                    f"  Step {self.num_timesteps}: "
                    f"reward={mean_reward:.4f}, "
                    f"trades={mean_trades:.0f}, "
                    f"portfolio=${mean_pv:,.0f}"
                )
        return True


# ============================================================
# Environment factory
# ============================================================
def make_env(features: np.ndarray, prices: np.ndarray, random_start: bool = False, **env_kwargs):
    def _init():
        env = TradingEnv(
            features=features, prices=prices,
            random_start=random_start, **env_kwargs,
        )
        env = Monitor(env)
        return env
    return _init


# ============================================================
# Evaluation — handles recurrent state for RecurrentPPO
# ============================================================
def evaluate_agent(
    model,
    features: np.ndarray,
    prices: np.ndarray,
    vec_normalize=None,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> dict:
    """Run the agent on test data with proper normalization and state management."""
    eval_env = DummyVecEnv([make_env(features, prices, random_start=False)])

    if vec_normalize is not None:
        eval_env = VecNormalize(
            eval_env, norm_obs=True, norm_reward=False,
            clip_obs=10.0, training=False,
        )
        eval_env.obs_rms = vec_normalize.obs_rms
        eval_env.ret_rms = vec_normalize.ret_rms

    all_rewards = []
    all_portfolio_values = []
    all_trades = []
    all_commissions = []
    all_actions = []
    all_step_rewards = []

    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0.0
        episode_actions = []
        ep_step_rewards = []

        # RecurrentPPO state management
        lstm_state = None
        episode_start = np.array([True])

        while not done:
            action, lstm_state = model.predict(
                obs, state=lstm_state, episode_start=episode_start,
                deterministic=deterministic,
            )
            obs, reward, dones, infos = eval_env.step(action)
            episode_start = dones  # reset state on next call if episode ended
            actual_reward = reward[0]
            episode_reward += actual_reward
            ep_step_rewards.append(actual_reward)
            episode_actions.append(int(action[0]))
            done = dones[0]
            info = infos[0]

        all_rewards.append(episode_reward)
        all_portfolio_values.append(info["portfolio_value"])
        all_trades.append(info["total_trades"])
        all_commissions.append(info["total_commission"])
        all_actions.append(episode_actions)
        all_step_rewards.append(ep_step_rewards)

    # Action distribution
    flat_actions = [a for ep in all_actions for a in ep]
    total_actions = max(len(flat_actions), 1)
    action_pct = {
        "buy": flat_actions.count(0) / total_actions * 100,
        "hold": flat_actions.count(1) / total_actions * 100,
        "flat": flat_actions.count(2) / total_actions * 100,
    }

    # Sharpe ratio (annualized, from step rewards)
    all_flat_rewards = [r for ep in all_step_rewards for r in ep]
    if len(all_flat_rewards) > 1 and np.std(all_flat_rewards) > 0:
        sharpe = np.mean(all_flat_rewards) / np.std(all_flat_rewards) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Buy & Hold benchmark
    bh_return = (prices[-1] - prices[0]) / prices[0]
    bh_portfolio = 100_000 * (1 + bh_return)

    results = {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_portfolio_value": float(np.mean(all_portfolio_values)),
        "mean_trades": float(np.mean(all_trades)),
        "mean_commission": float(np.mean(all_commissions)),
        "sharpe_ratio": float(sharpe),
        "action_distribution": action_pct,
        "buy_hold_portfolio": float(bh_portfolio),
        "buy_hold_return_pct": float(bh_return * 100),
        "agent_return_pct": float(
            (np.mean(all_portfolio_values) - 100_000) / 100_000 * 100
        ),
    }
    return results


# ============================================================
# Multi-ticker data loading
# ============================================================
def load_multi_ticker_data(tickers, period, interval, train_ratio):
    all_train_f, all_train_p = [], []
    all_test_f, all_test_p = [], []

    for ticker in tickers:
        print(f"  Loading {ticker}...")
        df = fetch_ohlcv(ticker, period=period, interval=interval)
        features, prices, dates = build_features(df)
        tr_f, tr_p, te_f, te_p = train_test_split(features, prices, train_ratio=train_ratio)
        all_train_f.append(tr_f)
        all_train_p.append(tr_p)
        all_test_f.append(te_f)
        all_test_p.append(te_p)
        print(f"    {ticker}: {len(tr_f)} train, {len(te_f)} test steps")

    return all_train_f, all_train_p, all_test_f, all_test_p


# ============================================================
# Main training routine
# ============================================================
def train(args):
    model_label = args.model
    tickers = args.tickers
    run_name = f"{model_label}_{'_'.join(tickers)}_{int(time.time())}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = run_dir / "tensorboard"
    best_model_dir = run_dir / "best_model"

    print(f"Run: {run_name}")
    print(f"Model variant: {model_label} (RecurrentPPO)")
    print(f"{'='*60}")

    # --- Data ---
    print(f"Loading data ({args.period}, {args.interval})...")
    all_train_f, all_train_p, all_test_f, all_test_p = load_multi_ticker_data(
        tickers, args.period, args.interval, args.train_ratio
    )

    # Training envs: one per ticker, all with random start
    train_env_fns = []
    for tf, tp in zip(all_train_f, all_train_p):
        train_env_fns.append(make_env(tf, tp, random_start=True))

    train_env = DummyVecEnv(train_env_fns)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Eval env: first ticker test set, no random start
    eval_env = DummyVecEnv([make_env(all_test_f[0], all_test_p[0], random_start=False)])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=False,
    )
    eval_env.obs_rms = train_env.obs_rms

    # --- Select wrapper for model variant ---
    if model_label == "baseline":
        wrapper_cls = KalmanLSTMWrapper
        wrapper_kwargs = {}
    elif model_label == "ode":
        wrapper_cls = KalmanODEWrapper
        wrapper_kwargs = dict(n_ode_steps=args.n_ode_steps)
    elif model_label == "fno":
        wrapper_cls = KalmanFNOWrapper
        wrapper_kwargs = dict(n_fno_layers=args.n_fno_layers, n_modes=args.n_modes)
    else:
        raise ValueError(f"Unknown model: {model_label}. Use 'baseline', 'fno', or 'ode'.")

    policy_kwargs = dict(
        kalman_wrapper_cls=wrapper_cls,
        kalman_hidden_size=args.hidden_size,
        kalman_wrapper_kwargs=wrapper_kwargs,
        shared_lstm=True,
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    ppo = RecurrentPPO(
        KalmanRecurrentPolicy,
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(tb_log_dir),
        device="auto",
    )

    total_params = sum(p.numel() for p in ppo.policy.parameters())
    print(f"\n  Model: {model_label} (RecurrentPPO) | Params: {total_params:,}")
    print(f"  Tickers: {tickers} | Envs: {len(train_env_fns)}")
    print(f"  Device: {ppo.device}")
    print(f"  Entropy coeff: {args.ent_coef}")
    print(f"  Training for {args.total_steps:,} steps...")
    print(f"{'='*60}\n")

    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // len(train_env_fns), 1),
        n_eval_episodes=5,
        deterministic=True,
    )
    metrics_callback = TradingMetricsCallback(log_freq=5000, verbose=1)

    # --- Train ---
    t0 = time.time()
    ppo.learn(
        total_timesteps=args.total_steps,
        callback=[eval_callback, metrics_callback],
        progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save
    ppo.save(str(run_dir / "final_model"))
    train_env.save(str(run_dir / "vec_normalize.pkl"))

    # --- Evaluate on all test sets ---
    print(f"\n{'='*60}")
    print("EVALUATION ON TEST SET")
    print(f"{'='*60}")

    all_results = {}
    for i, ticker in enumerate(tickers):
        print(f"\n  --- {ticker} ---")
        results = evaluate_agent(
            ppo, all_test_f[i], all_test_p[i],
            vec_normalize=train_env, n_episodes=10,
        )
        for k, v in results.items():
            if isinstance(v, dict):
                print(f"    {k}:")
                for kk, vv in v.items():
                    print(f"      {kk}: {vv:.1f}%")
            elif isinstance(v, float):
                print(f"    {k}: {v:.4f}")
        all_results[ticker] = results

        with open(run_dir / f"eval_{ticker}.json", "w") as f:
            json.dump(results, f, indent=2)

    # Save config
    config = vars(args)
    config["total_params"] = total_params
    config["training_time_s"] = elapsed
    config["tickers"] = tickers
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to: {run_dir}")
    print(f"TensorBoard: tensorboard --logdir \"{tb_log_dir}\"")
    return ppo, all_results


def eval_only(args):
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    run_dir = model_path.parent if model_path.suffix == ".zip" else model_path.parent.parent
    vec_norm_path = run_dir / "vec_normalize.pkl"

    print(f"Loading model from {model_path}...")
    # RecurrentPPO needs the custom policy class to be importable for loading
    ppo = RecurrentPPO.load(str(model_path))

    vec_normalize = None
    if vec_norm_path.exists():
        print(f"Loading normalization stats from {vec_norm_path}...")
        df = fetch_ohlcv(args.tickers[0], period=args.period, interval=args.interval)
        features, prices, _ = build_features(df)
        _, _, test_f, test_p = train_test_split(features, prices, train_ratio=args.train_ratio)
        dummy_env = DummyVecEnv([make_env(test_f, test_p)])
        vec_normalize = VecNormalize.load(str(vec_norm_path), dummy_env)
        vec_normalize.training = False

    for ticker in args.tickers:
        df = fetch_ohlcv(ticker, period=args.period, interval=args.interval)
        features, prices, _ = build_features(df)
        _, _, test_f, test_p = train_test_split(features, prices, train_ratio=args.train_ratio)

        print(f"\n--- {ticker} ({len(test_f)} test steps) ---")
        results = evaluate_agent(ppo, test_f, test_p, vec_normalize=vec_normalize, n_episodes=10)
        for k, v in results.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(f"  {kk}: {vv:.1f}%")
            elif isinstance(v, float):
                print(f"  {k}: {v:.4f}")


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="RecurrentPPO Trading Agent -- Ablation Study")

    # Data
    parser.add_argument("--tickers", nargs="+", default=["SPY"])
    parser.add_argument("--period", type=str, default="5y")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--train-ratio", type=float, default=0.8)

    # Model variant
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "fno", "ode"],
                        help="Model variant for ablation")

    # Architecture
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--n-fno-layers", type=int, default=2)
    parser.add_argument("--n-modes", type=int, default=16)
    parser.add_argument("--n-ode-steps", type=int, default=4)

    # PPO hyperparameters
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.05)

    # Evaluation
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="runs")

    # Modes
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        if args.model_path is None:
            raise ValueError("--model-path required for --eval mode")
        eval_only(args)
    else:
        train(args)
