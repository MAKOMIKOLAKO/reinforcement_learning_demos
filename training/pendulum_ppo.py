"""
PPO training script for Gymnasium's standard Pendulum-v1 environment.

Trains a small MLP actor (policy) network to swing up and balance an
inverted pendulum, and periodically exports the actor weights to JSON
checkpoints that a browser-based JS demo can load and run forward passes
on (no PyTorch / ML framework needed at inference time).

Implementation notes:
  * Uses the UNMODIFIED Gymnasium Pendulum-v1 env (standard physics:
    g=10.0, m=1.0, l=1.0, dt=0.05, max_speed=8.0, max_torque=2.0) so the
    JS reimplementation of the physics matches exactly.
  * PPO (clipped surrogate objective) is implemented from scratch with
    plain NumPy (no torch/stable-baselines3 dependency needed) -- both
    the actor and a small critic (value function, used only to compute
    advantages during training) are simple 2-hidden-layer MLPs.
  * Only the ACTOR is exported. The critic is training-only scaffolding.
  * Actor architecture: input 3 -> [64, 64] ReLU -> 1, tanh output
    representing the action mean in [-1, 1]. The action_scale (2.0,
    i.e. Pendulum's max_torque) is recorded in the exported JSON and is
    NOT baked into the weights -- the JS demo multiplies the tanh output
    by action_scale at inference time.

Run:
    python training/pendulum_ppo.py
"""

import json
import os
import time

import numpy as np
import gymnasium as gym

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

SEED = 0
ENV_ID = "Pendulum-v1"

HIDDEN_SIZES = [64, 64]
ACTION_SCALE = 2.0  # Pendulum-v1 max_torque

TOTAL_TIMESTEPS = 1_000_000
ROLLOUT_LEN = 2048
PPO_EPOCHS = 10
MINIBATCH_SIZE = 256
CLIP_EPS = 0.2
GAMMA = 0.98
GAE_LAMBDA = 0.95
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
MAX_GRAD_NORM = 0.5
ENTROPY_COEF = 0.0  # exploration handled via explicit std schedule instead

STD_START = 1.2
STD_END = 0.1

CHECKPOINT_PCTS = [0, 5, 10, 25, 50, 75, 100]

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public", "checkpoints", "pendulum", "ppo",
)

rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------
# Small MLP utilities (manual forward + backward, Adam optimizer)
# --------------------------------------------------------------------------

def init_mlp(sizes, rng):
    """sizes: e.g. [3, 64, 64, 1]. Weights stored as (out, in) to match
    the y = W @ x + b convention used both here and in the exported JSON."""
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        limit = np.sqrt(6.0 / (fan_in + fan_out))  # Glorot uniform
        w = rng.uniform(-limit, limit, size=(fan_out, fan_in)).astype(np.float64)
        b = np.zeros(fan_out, dtype=np.float64)
        params.append({"w": w, "b": b})
    return params


def relu(x):
    return np.maximum(x, 0.0)


def mlp_forward(params, x, final_activation):
    """x: (batch, in_dim). Returns final output and cache for backprop."""
    cache = {"x": [x]}
    a = x
    n_layers = len(params)
    for i, layer in enumerate(params):
        z = a @ layer["w"].T + layer["b"]  # (batch, out)
        if i < n_layers - 1:
            a = relu(z)
        else:
            if final_activation == "tanh":
                a = np.tanh(z)
            else:
                a = z
        cache.setdefault("z", []).append(z)
        cache["x"].append(a)
    return a, cache


def mlp_backward(params, cache, d_out, final_activation):
    """d_out: gradient of loss wrt the network's final output (batch, out_dim).
    Returns list of {"w": dW, "b": db} matching params, summed over batch."""
    n_layers = len(params)
    grads = [None] * n_layers
    d_a = d_out
    for i in reversed(range(n_layers)):
        z = cache["z"][i]
        a_prev = cache["x"][i]
        if i == n_layers - 1 and final_activation == "tanh":
            d_z = d_a * (1.0 - np.tanh(z) ** 2)
        elif i == n_layers - 1:
            d_z = d_a
        else:
            d_z = d_a * (z > 0).astype(np.float64)
        dW = d_z.T @ a_prev
        db = d_z.sum(axis=0)
        grads[i] = {"w": dW, "b": db}
        d_a = d_z @ params[i]["w"]
    return grads


class Adam:
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [{"w": np.zeros_like(p["w"]), "b": np.zeros_like(p["b"])} for p in params]
        self.v = [{"w": np.zeros_like(p["w"]), "b": np.zeros_like(p["b"])} for p in params]

    def step(self, params, grads):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            for k in ("w", "b"):
                m[k][:] = self.b1 * m[k] + (1 - self.b1) * g[k]
                v[k][:] = self.b2 * v[k] + (1 - self.b2) * (g[k] ** 2)
                m_hat = m[k] / bc1
                v_hat = v[k] / bc2
                p[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_grad_norm(grads, max_norm):
    total_sq = 0.0
    for g in grads:
        total_sq += np.sum(g["w"] ** 2) + np.sum(g["b"] ** 2)
    total_norm = np.sqrt(total_sq)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for g in grads:
            g["w"] *= scale
            g["b"] *= scale
    return total_norm


# --------------------------------------------------------------------------
# Policy helpers
# --------------------------------------------------------------------------

def actor_mean(actor_params, obs):
    """Returns tanh output in [-1, 1] (NOT yet scaled by ACTION_SCALE)."""
    out, cache = mlp_forward(actor_params, obs, final_activation="tanh")
    return out, cache


def sample_action(mean_raw, std):
    """mean_raw in [-1,1]; scaled mean is mean_raw*ACTION_SCALE. Sample in
    scaled (torque) space, then clip to valid torque range."""
    mean_scaled = mean_raw * ACTION_SCALE
    noise = rng.normal(size=mean_scaled.shape)
    action = mean_scaled + noise * std
    logp = gaussian_logp(action, mean_scaled, std)
    action_clipped = np.clip(action, -ACTION_SCALE, ACTION_SCALE)
    return action, action_clipped, logp


def gaussian_logp(x, mean, std):
    var = std ** 2
    logp = -0.5 * ((x - mean) ** 2) / var - np.log(std) - 0.5 * np.log(2 * np.pi)
    return logp.sum(axis=-1)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def export_checkpoint(actor_params, pct, path):
    layers = []
    for layer in actor_params:
        layers.append({
            "w": layer["w"].tolist(),
            "b": layer["b"].tolist(),
        })
    data = {
        "algorithm": "ppo",
        "env": "pendulum",
        "checkpoint_pct": pct,
        "architecture": {
            "input_dim": 3,
            "hidden_layers": HIDDEN_SIZES,
            "output_dim": 1,
            "activation": "relu",
            "output_activation": "tanh",
            "action_scale": ACTION_SCALE,
        },
        "action_scale": ACTION_SCALE,
        "layers": layers,
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return data


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def evaluate(actor_params, n_episodes=5):
    env = gym.make(ENV_ID)
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED + 1000 + ep)
        done = False
        ep_ret = 0.0
        while not done:
            mean_raw, _ = actor_mean(actor_params, obs[None, :])
            action = np.clip(mean_raw[0] * ACTION_SCALE, -ACTION_SCALE, ACTION_SCALE)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            done = terminated or truncated
        total += ep_ret
    env.close()
    return total / n_episodes


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    sizes = [3] + HIDDEN_SIZES + [1]
    actor_params = init_mlp(sizes, rng)
    critic_params = init_mlp(sizes, rng)  # same shape, separate net (out_dim=1, no squash needed)

    actor_opt = Adam(actor_params, lr=ACTOR_LR)
    critic_opt = Adam(critic_params, lr=CRITIC_LR)

    # Save the 0% (random init) checkpoint immediately.
    saved_pcts = set()
    path0 = os.path.join(OUT_DIR, "checkpoint_000.json")
    export_checkpoint(actor_params, 0, path0)
    saved_pcts.add(0)
    print(f"[0%] saved untrained checkpoint -> {path0}")

    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=SEED)

    n_updates = TOTAL_TIMESTEPS // ROLLOUT_LEN
    global_step = 0
    start_time = time.time()

    for update in range(1, n_updates + 1):
        progress = global_step / TOTAL_TIMESTEPS
        std = STD_START + (STD_END - STD_START) * progress

        obs_buf = np.zeros((ROLLOUT_LEN, 3), dtype=np.float64)
        act_buf = np.zeros((ROLLOUT_LEN, 1), dtype=np.float64)
        logp_buf = np.zeros(ROLLOUT_LEN, dtype=np.float64)
        rew_buf = np.zeros(ROLLOUT_LEN, dtype=np.float64)
        val_buf = np.zeros(ROLLOUT_LEN, dtype=np.float64)
        done_buf = np.zeros(ROLLOUT_LEN, dtype=np.float64)

        for t in range(ROLLOUT_LEN):
            mean_raw, _ = actor_mean(actor_params, obs[None, :])
            value, _ = mlp_forward(critic_params, obs[None, :], final_activation="linear")
            action, action_clipped, logp = sample_action(mean_raw, std)

            obs_buf[t] = obs
            act_buf[t] = action[0]
            logp_buf[t] = logp[0]
            val_buf[t] = value[0, 0]

            next_obs, reward, terminated, truncated, _ = env.step(action_clipped[0])
            rew_buf[t] = reward
            done_buf[t] = float(terminated or truncated)

            obs = next_obs
            global_step += 1
            if terminated or truncated:
                obs, _ = env.reset()

        last_value, _ = mlp_forward(critic_params, obs[None, :], final_activation="linear")
        last_value = last_value[0, 0]

        # GAE-Lambda advantage estimation.
        advantages = np.zeros(ROLLOUT_LEN, dtype=np.float64)
        last_gae = 0.0
        for t in reversed(range(ROLLOUT_LEN)):
            next_nonterminal = 1.0 - done_buf[t]
            next_value = last_value if t == ROLLOUT_LEN - 1 else val_buf[t + 1]
            delta = rew_buf[t] + GAMMA * next_value * next_nonterminal - val_buf[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs over minibatches.
        idx_all = np.arange(ROLLOUT_LEN)
        for epoch in range(PPO_EPOCHS):
            rng.shuffle(idx_all)
            for start in range(0, ROLLOUT_LEN, MINIBATCH_SIZE):
                mb_idx = idx_all[start:start + MINIBATCH_SIZE]
                mb_obs = obs_buf[mb_idx]
                mb_act = act_buf[mb_idx]
                mb_logp_old = logp_buf[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                # ---- Actor forward ----
                mean_raw, cache = mlp_forward(actor_params, mb_obs, final_activation="tanh")
                mean_scaled = mean_raw * ACTION_SCALE
                logp_new = gaussian_logp(mb_act, mean_scaled, std)

                ratio = np.exp(logp_new - mb_logp_old)
                unclipped = ratio * mb_adv
                clipped_ratio = np.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                clipped = clipped_ratio * mb_adv

                use_unclipped = unclipped <= clipped
                in_bounds = (ratio >= 1 - CLIP_EPS) & (ratio <= 1 + CLIP_EPS)
                d_loss_d_ratio = np.where(
                    use_unclipped,
                    -mb_adv,
                    np.where(in_bounds, -mb_adv, 0.0),
                )
                d_loss_d_logp_new = d_loss_d_ratio * ratio  # (batch,)

                # d logp_new / d mean_scaled = (a - mean)/std^2
                d_logp_d_mean_scaled = (mb_act - mean_scaled) / (std ** 2)  # (batch, 1)
                d_loss_d_mean_scaled = d_loss_d_logp_new[:, None] * d_logp_d_mean_scaled
                d_loss_d_mean_raw = d_loss_d_mean_scaled * ACTION_SCALE

                d_out = d_loss_d_mean_raw / mb_obs.shape[0]  # mean over batch
                actor_grads = mlp_backward(actor_params, cache, d_out, final_activation="tanh")
                clip_grad_norm(actor_grads, MAX_GRAD_NORM)
                actor_opt.step(actor_params, actor_grads)

                # ---- Critic forward/backward (MSE) ----
                pred, c_cache = mlp_forward(critic_params, mb_obs, final_activation="linear")
                d_pred = 2.0 * (pred[:, 0] - mb_ret)[:, None] / mb_obs.shape[0]
                critic_grads = mlp_backward(critic_params, c_cache, d_pred, final_activation="linear")
                clip_grad_norm(critic_grads, MAX_GRAD_NORM)
                critic_opt.step(critic_params, critic_grads)

        pct_progress = int(round(100 * global_step / TOTAL_TIMESTEPS))
        for pct in CHECKPOINT_PCTS:
            if pct in saved_pcts:
                continue
            if pct_progress >= pct:
                path = os.path.join(OUT_DIR, f"checkpoint_{pct:03d}.json")
                export_checkpoint(actor_params, pct, path)
                saved_pcts.add(pct)
                avg_ret = evaluate(actor_params)
                elapsed = time.time() - start_time
                print(f"[{pct}%] step={global_step} avg_eval_return={avg_ret:.1f} "
                      f"std={std:.3f} elapsed={elapsed:.1f}s -> {path}")

    # Ensure 100% is always saved even if rounding missed it.
    if 100 not in saved_pcts:
        path = os.path.join(OUT_DIR, "checkpoint_100.json")
        export_checkpoint(actor_params, 100, path)
        avg_ret = evaluate(actor_params)
        print(f"[100%] final avg_eval_return={avg_ret:.1f} -> {path}")

    env.close()

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"checkpoints": CHECKPOINT_PCTS}, f)
    print(f"wrote manifest -> {manifest_path}")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s ({total_time/60:.2f} min)")


if __name__ == "__main__":
    main()
