"""
Vanilla REINFORCE (Monte-Carlo policy gradient) on CartPole-v1.

This is a from-scratch implementation: the policy network (a small MLP) and
its forward/backward pass are implemented directly with numpy -- no
autograd/deep-learning framework, and deliberately NO baseline / value
function / critic of any kind. That is the point of this demo: it exists to
contrast with the actor-critic (PPO) demo elsewhere on the site, and it is
expected to show the high variance that plain REINFORCE is known for.

Usage:
    python cartpole_reinforce.py

Outputs:
    public/checkpoints/cartpole/reinforce/checkpoint_{pct:03d}.json
        for pct in (0, 5, 10, 25, 50, 75, 100)
    public/checkpoints/cartpole/reinforce/manifest.json
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
GAMMA = 0.99
# Plain SGD (gradient ascent) rather than Adam: closer to the textbook
# REINFORCE update, and empirically much less prone to the late-training
# policy collapse that adaptive per-parameter step sizes (Adam) caused here
# once the policy became confident. The learning rate is linearly annealed
# so late-training updates are gentler -- a step-size schedule, not a
# variance-reduction trick.
LEARNING_RATE_START = 0.05
LEARNING_RATE_END = 0.005
GRAD_CLIP_NORM = 2.0  # numerical-stability safeguard only, not a variance-reduction trick
HIDDEN_SIZES = [64, 64]
INPUT_DIM = 4
OUTPUT_DIM = 2
TOTAL_EPISODES = 5000
CHECKPOINT_PCTS = [0, 5, 10, 25, 50, 75, 100]
MAX_STEPS_PER_EPISODE = 500  # CartPole-v1 default cap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "public", "checkpoints", "cartpole", "reinforce")
OUT_DIR = os.path.normpath(OUT_DIR)


# --------------------------------------------------------------------------
# Policy network: 4 -> 64 -> 64 -> 2, ReLU, ReLU, softmax.
# Implemented by hand (forward + backward) with numpy arrays.
# --------------------------------------------------------------------------

class PolicyNetwork:
    def __init__(self, rng):
        sizes = [INPUT_DIM] + HIDDEN_SIZES + [OUTPUT_DIM]
        self.num_layers = len(sizes) - 1
        self.W = []
        self.b = []
        for i in range(self.num_layers):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            # He initialization for ReLU hidden layers, smaller scale for
            # the output (softmax) layer so the initial policy is close to
            # uniform random.
            if i < self.num_layers - 1:
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = 0.01
            self.W.append(rng.standard_normal((fan_out, fan_in)) * scale)
            self.b.append(np.zeros(fan_out))

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def _softmax(x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def forward(self, x):
        """Returns (probs, cache) where cache holds intermediates for backprop."""
        activations = [x]
        zs = []
        a = x
        for i in range(self.num_layers):
            z = self.W[i] @ a + self.b[i]
            zs.append(z)
            if i < self.num_layers - 1:
                a = self._relu(z)
            else:
                a = self._softmax(z)
            activations.append(a)
        probs = activations[-1]
        cache = (activations, zs)
        return probs, cache

    def backward(self, cache, action):
        """
        Gradient of -log(pi(action|state)) w.r.t. all weights/biases.
        For a softmax output with cross-entropy-style loss against the
        chosen action, d(loss)/d(logits) = probs - one_hot(action).
        """
        activations, zs = cache
        probs = activations[-1]

        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        one_hot = np.zeros(OUTPUT_DIM)
        one_hot[action] = 1.0
        delta = probs - one_hot  # dL/dz for the output layer

        for i in reversed(range(self.num_layers)):
            a_prev = activations[i]
            grads_W[i] = np.outer(delta, a_prev)
            grads_b[i] = delta
            if i > 0:
                da_prev = self.W[i].T @ delta
                relu_mask = (zs[i - 1] > 0).astype(np.float64)
                delta = da_prev * relu_mask

        return grads_W, grads_b

    def param_list(self):
        return list(self.W) + list(self.b)

    def to_checkpoint_dict(self, pct):
        layers = []
        for i in range(self.num_layers):
            layers.append({
                "w": self.W[i].tolist(),
                "b": self.b[i].tolist(),
            })
        return {
            "algorithm": "reinforce",
            "env": "cartpole",
            "checkpoint_pct": pct,
            "architecture": {
                "input_dim": INPUT_DIM,
                "hidden_layers": HIDDEN_SIZES,
                "output_dim": OUTPUT_DIM,
                "activation": "relu",
                "output_activation": "softmax",
            },
            "layers": layers,
        }


# --------------------------------------------------------------------------
# Plain SGD "optimizer" (gradient ascent on expected return, implemented as
# gradient descent on the -log(pi) * G loss) with a linearly-annealed
# learning rate. No momentum, no per-parameter adaptive scaling.
# --------------------------------------------------------------------------

class SGD:
    def __init__(self, lr_start, lr_end, total_steps):
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.total_steps = max(1, total_steps)
        self.t = 0

    def current_lr(self):
        frac = min(1.0, self.t / self.total_steps)
        return self.lr_start + frac * (self.lr_end - self.lr_start)

    def step(self, params, grads):
        lr = self.current_lr()
        for p, g in zip(params, grads):
            p -= lr * g
        self.t += 1


def discounted_returns(rewards, gamma):
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def run_episode(env, net, rng):
    state, _ = env.reset(seed=None)
    states, actions, rewards, caches = [], [], [], []
    for _ in range(MAX_STEPS_PER_EPISODE):
        probs, cache = net.forward(state)
        action = rng.choice(OUTPUT_DIM, p=probs)
        next_state, reward, terminated, truncated, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        caches.append(cache)

        state = next_state
        if terminated or truncated:
            break
    return states, actions, rewards, caches


def save_checkpoint(net, pct):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"checkpoint_{pct:03d}.json")
    with open(path, "w") as f:
        json.dump(net.to_checkpoint_dict(pct), f)
    print(f"  saved {path}")


def save_manifest():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "manifest.json")
    with open(path, "w") as f:
        json.dump({"checkpoints": CHECKPOINT_PCTS}, f)
    print(f"  saved {path}")


def main():
    rng = np.random.default_rng(SEED)
    env = gym.make("CartPole-v1")

    net = PolicyNetwork(rng)
    optimizer = SGD(LEARNING_RATE_START, LEARNING_RATE_END, TOTAL_EPISODES)

    checkpoint_episodes = {
        pct: max(0, round(TOTAL_EPISODES * pct / 100) - 1) if pct > 0 else 0
        for pct in CHECKPOINT_PCTS
    }
    # episode index (0-based) at which to snapshot for each pct; last one
    # should correspond to *after* the final training episode.
    checkpoint_episodes[0] = -1  # save before any training happens
    checkpoint_episodes[100] = TOTAL_EPISODES - 1

    saved_pcts = set()

    # Save the untrained (0%) checkpoint first.
    save_checkpoint(net, 0)
    saved_pcts.add(0)

    start_time = time.time()
    reward_history = []

    for episode in range(TOTAL_EPISODES):
        states, actions, rewards, caches = run_episode(env, net, rng)
        episode_return = sum(rewards)
        reward_history.append(episode_return)

        returns = discounted_returns(rewards, GAMMA)

        # Accumulate policy-gradient gradients over the whole trajectory.
        # No baseline is subtracted from `returns` -- this is vanilla
        # REINFORCE, so variance across episodes/updates is expected to be
        # high, especially early in training.
        T = len(rewards)
        grad_W_sum = [np.zeros_like(w) for w in net.W]
        grad_b_sum = [np.zeros_like(b) for b in net.b]
        for t in range(T):
            grads_W, grads_b = net.backward(caches[t], actions[t])
            G = returns[t]
            for i in range(net.num_layers):
                grad_W_sum[i] += grads_W[i] * G
                grad_b_sum[i] += grads_b[i] * G

        # Average over timesteps (a per-episode learning-rate normalization,
        # not a variance-reduction trick like a baseline/critic).
        grads = [g / T for g in grad_W_sum] + [g / T for g in grad_b_sum]

        # Clip the global gradient norm purely to avoid numerical blow-ups
        # from occasional huge returns/long episodes; this does not reduce
        # the variance of the REINFORCE gradient estimator itself.
        total_norm = np.sqrt(sum(np.sum(g * g) for g in grads))
        if total_norm > GRAD_CLIP_NORM:
            scale = GRAD_CLIP_NORM / (total_norm + 1e-8)
            grads = [g * scale for g in grads]

        params = net.param_list()
        optimizer.step(params, grads)

        pct_done = round(100 * (episode + 1) / TOTAL_EPISODES)
        for pct in CHECKPOINT_PCTS:
            if pct == 0:
                continue
            if pct not in saved_pcts and episode == checkpoint_episodes[pct]:
                save_checkpoint(net, pct)
                saved_pcts.add(pct)

        if (episode + 1) % 100 == 0:
            recent = reward_history[-100:]
            print(f"episode {episode + 1}/{TOTAL_EPISODES} "
                  f"return={episode_return:.0f} "
                  f"avg_last_100={np.mean(recent):.1f}")

    # Safety net: make sure every requested pct got saved even if rounding
    # caused an off-by-one.
    for pct in CHECKPOINT_PCTS:
        if pct not in saved_pcts:
            save_checkpoint(net, pct)
            saved_pcts.add(pct)

    save_manifest()

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s "
          f"({TOTAL_EPISODES} episodes).")
    print(f"Final avg return (last 100 episodes): "
          f"{np.mean(reward_history[-100:]):.1f}")

    env.close()


if __name__ == "__main__":
    main()
