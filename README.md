# RL Demos

Interactive, in-browser demos of reinforcement learning agents trained offline in Python. Three classic algorithms, three classic environments — trained once, exported as JSON checkpoints, and replayed forward-pass-only as plain client-side JavaScript. No ML framework ships to the browser.

Once deployed via GitHub Pages, scrub the training-progress slider on each demo to watch a policy evolve from random noise to competent behavior.

| Demo | Algorithm | Environment | What it shows |
|---|---|---|---|
| [Snake](snake/) | DQN (value-only) | 20x20 grid | Learns Q-values and greedily picks the best action to survive and grow. |
| [CartPole](cartpole/) | REINFORCE (policy-only) | `CartPole-v1` | Learns a stochastic policy directly by sampling actions from probabilities — no baseline/critic, so it shows the high variance vanilla policy gradient is known for. |
| [Pendulum](pendulum/) | PPO (actor-critic) | `Pendulum-v1` | Jointly learns a policy and a value function to swing up and balance a continuous-control pendulum. |

A [side-by-side comparison](compare/) page contrasts all three approaches.

## How it works

Training and inference are deliberately split:

- **Training** happens offline, in Python (`training/`), and is never run in the browser.
- **Inference** is a handful of exported checkpoints (`public/checkpoints/`) — plain JSON arrays of weights — replayed with hand-written matrix multiplies in JavaScript (`shared/inference.js`). Moving the training-progress slider just swaps which checkpoint's weights are used for the forward pass.

This keeps the site a static bundle of HTML/CSS/JS with zero runtime dependencies and zero build step — it's served as-is by GitHub Pages.

## Repo layout

```
index.html              Landing page linking to the three demos
snake/                   Snake demo (game.js: env/rendering, main.js: demo wiring)
cartpole/                CartPole demo
pendulum/                Pendulum demo (pendulum.js: env/physics + demo wiring)
compare/                 Side-by-side DQN vs. REINFORCE vs. PPO comparison page
shared/                  Code shared across demos
  style.css              Site-wide styles
  inference.js           JS forward-pass implementations (MLP, DQN/policy heads)
  rl-diagrams.js          Mermaid.js-based architecture/flow diagrams
  glossary.js             Shared RL term glossary/tooltips
training/                Offline Python training scripts (one per demo)
public/checkpoints/      Exported JSON weight checkpoints + manifest.json per demo
.github/workflows/        GitHub Pages deploy workflow
```

## Running locally

The site is fully static — no build step, no bundler, no `node_modules`. Serve the repo root with any static file server and open it in a browser:

```bash
python -m http.server 8000
# then visit http://localhost:8000/
```

(Opening the HTML files directly via `file://` won't work — the demos `fetch()` their JSON checkpoints, which most browsers block under the `file://` origin.)

## Re-training the agents

Each demo has a corresponding training script in `training/`, implemented from scratch with NumPy (no autograd framework) except for Snake's DQN, which uses PyTorch for the network/optimizer.

```bash
pip install numpy gymnasium torch

python training/cartpole_reinforce.py   # -> public/checkpoints/cartpole/reinforce/
python training/pendulum_ppo.py         # -> public/checkpoints/pendulum/ppo/
python training/snake_dqn.py            # -> public/checkpoints/snake/dqn/
```

Each script periodically snapshots the policy (at 0%, 5%, 10%, 25%, 50%, 75%, 100% of training) into `checkpoint_XXX.json` files plus a `manifest.json` listing the available checkpoints — this is exactly what the training-progress slider in each demo reads. Re-running a script overwrites its demo's checkpoints in place.

Environment physics/state representations in the JS demos are written to match their Python/Gymnasium counterparts exactly (see the header comments in each `training/*.py` file for the precise env spec, e.g. Snake's state vector and reward shaping) — if you change an environment's physics or state encoding in the Python trainer, update the matching JS reimplementation as well.

## Deployment

Pushes to `main` trigger [`.github/workflows/`](.github/workflows/) to publish the repo root directly to GitHub Pages — no build step runs in CI either. `.nojekyll` disables Jekyll processing so files/folders starting with `_` (if any) are served as-is.

## Tech stack

- Vanilla HTML/CSS/JS on the frontend, no framework, no bundler.
- [Mermaid.js](https://mermaid.js.org/) (loaded from CDN) for architecture diagrams in `shared/rl-diagrams.js`.
- Python + NumPy (+ Gymnasium for envs, PyTorch for the Snake DQN) for offline training.
