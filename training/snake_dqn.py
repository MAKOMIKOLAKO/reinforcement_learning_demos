"""
Train a small DQN agent to play Snake, and export checkpoints as JSON
weight files that a browser can replay with plain matrix multiplies
(no ML runtime needed client-side).

Environment spec (must match the JS reimplementation exactly):
  - Grid: 20x20 cells.
  - State vector (11 floats), in this exact order:
      [danger_straight, danger_right, danger_left,
       moving_left, moving_right, moving_up, moving_down,
       food_left, food_right, food_up, food_down]
    danger_* is 1.0 if moving that relative direction hits a wall or the
    snake's own body within 1 cell, else 0.0.
    moving_* is a one-hot of the current absolute heading.
    food_* is 1.0 if food is in that absolute direction relative to the head.
  - Actions: 4 discrete absolute headings: up, down, left, right. Reversing
    directly into your own neck is legal input -- it just kills you, same
    as any other self-collision. No special-casing.
  - Reward: +10 for eating food, -10 for dying (wall or self collision),
    -0.01 per step otherwise.
  - Episode ends on collision, or after 500 steps with no food eaten.

Network: MLP, input 11 -> hidden 64 (ReLU) -> hidden 64 (ReLU) -> output 4
(linear, raw Q-values).

Checkpoints are exported at training progress 0/5/10/25/50/75/100 percent
to public/checkpoints/snake/dqn/checkpoint_{pct:03d}.json, plus a
manifest.json listing the available percentages.
"""

import json
import os
import random
import time
from collections import deque

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------------------------------
# Env
# --------------------------------------------------------------------------

GRID_SIZE = 20
MAX_STEPS_NO_FOOD = 500

# Absolute directions as (dx, dy) with y growing downward.
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Action index -> absolute direction. Order matches "up, down, left, right".
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# Clockwise cycle used to compute "turn right" / "turn left" relative to
# whatever the current heading is (used only for the danger_* features --
# the action space itself is absolute, not relative).
CW_ORDER = [UP, RIGHT, DOWN, LEFT]


def turn_right(d):
    i = CW_ORDER.index(d)
    return CW_ORDER[(i + 1) % 4]


def turn_left(d):
    i = CW_ORDER.index(d)
    return CW_ORDER[(i - 1) % 4]


class SnakeEnv:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        c = self.grid_size // 2
        # Snake body: list of (x, y), index 0 is the head.
        self.snake = [(c, c), (c - 1, c), (c - 2, c)]
        self.direction = RIGHT
        self.steps_since_food = 0
        self._place_food()
        self.done = False
        return self._get_state()

    def _place_food(self):
        occupied = set(self.snake)
        free = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]
        self.food = random.choice(free)

    def _is_collision(self, pt):
        x, y = pt
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        # Body collision (tail cell excluded elsewhere by caller logic when
        # relevant -- here we just check against the full current body,
        # which is the right check for danger-sensing one step ahead).
        if pt in self.snake:
            return True
        return False

    def _get_state(self):
        head = self.snake[0]
        d = self.direction

        straight_pt = (head[0] + d[0], head[1] + d[1])
        right_pt = (head[0] + turn_right(d)[0], head[1] + turn_right(d)[1])
        left_pt = (head[0] + turn_left(d)[0], head[1] + turn_left(d)[1])

        danger_straight = 1.0 if self._is_collision(straight_pt) else 0.0
        danger_right = 1.0 if self._is_collision(right_pt) else 0.0
        danger_left = 1.0 if self._is_collision(left_pt) else 0.0

        moving_left = 1.0 if d == LEFT else 0.0
        moving_right = 1.0 if d == RIGHT else 0.0
        moving_up = 1.0 if d == UP else 0.0
        moving_down = 1.0 if d == DOWN else 0.0

        food_left = 1.0 if self.food[0] < head[0] else 0.0
        food_right = 1.0 if self.food[0] > head[0] else 0.0
        food_up = 1.0 if self.food[1] < head[1] else 0.0
        food_down = 1.0 if self.food[1] > head[1] else 0.0

        return np.array(
            [
                danger_straight, danger_right, danger_left,
                moving_left, moving_right, moving_up, moving_down,
                food_left, food_right, food_up, food_down,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        if self.done:
            raise RuntimeError("step() called on finished episode")

        self.direction = ACTIONS[action]
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        reward = -0.01
        self.steps_since_food += 1

        # Wall collision.
        out_of_bounds = (
            new_head[0] < 0
            or new_head[0] >= self.grid_size
            or new_head[1] < 0
            or new_head[1] >= self.grid_size
        )

        ate_food = new_head == self.food

        # Body collision: the tail cell vacates this step unless food is
        # eaten (snake grows and tail stays put).
        body_check = self.snake if ate_food else self.snake[:-1]
        self_collision = new_head in body_check

        if out_of_bounds or self_collision:
            self.done = True
            reward = -10.0
            return self._get_state(), reward, self.done

        self.snake.insert(0, new_head)
        if ate_food:
            reward = 10.0
            self.steps_since_food = 0
            self._place_food()
        else:
            self.snake.pop()

        if self.steps_since_food >= MAX_STEPS_NO_FOOD:
            self.done = True

        return self._get_state(), reward, self.done


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

class QNet(nn.Module):
    def __init__(self, input_dim=11, hidden=(64, 64), output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def export_checkpoint(net: QNet, pct: int, path: str):
    layers = []
    for lin in (net.fc1, net.fc2, net.fc3):
        w = lin.weight.detach().cpu().numpy().tolist()  # [out, in]
        b = lin.bias.detach().cpu().numpy().tolist()  # [out]
        layers.append({"w": w, "b": b})

    checkpoint = {
        "algorithm": "dqn",
        "env": "snake",
        "checkpoint_pct": pct,
        "architecture": {
            "input_dim": 11,
            "hidden_layers": [64, 64],
            "output_dim": 4,
            "activation": "relu",
            "output_activation": "linear",
        },
        "layers": layers,
    }

    with open(path, "w") as f:
        json.dump(checkpoint, f)


# --------------------------------------------------------------------------
# Replay buffer
# --------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cpu")

    total_episodes = 1900
    checkpoint_pcts = [0, 5, 10, 25, 50, 75, 100]
    checkpoint_episodes = {
        pct: max(0, round(total_episodes * pct / 100)) for pct in checkpoint_pcts
    }
    # 100% checkpoint should be taken after the very last episode.
    checkpoint_episodes[100] = total_episodes

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "public",
        "checkpoints", "snake", "dqn",
    )
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    policy_net = QNet().to(device)
    target_net = QNet().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Export the untrained (random-init) network as the 0% checkpoint
    # right away, before any training happens.
    export_checkpoint(policy_net, 0, os.path.join(out_dir, "checkpoint_000.json"))
    print("Exported checkpoint_000.json (random init)")

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=50_000)

    batch_size = 64
    gamma = 0.95
    eps_start = 1.0
    eps_end = 0.01
    eps_decay_episodes = int(total_episodes * 0.7)
    target_update_every = 500  # env steps
    train_every = 1  # env steps
    min_buffer = 1000

    env = SnakeEnv()

    global_step = 0
    recent_scores = deque(maxlen=100)
    start_time = time.time()

    pending_pcts = sorted(p for p in checkpoint_pcts if p != 0)

    for episode in range(1, total_episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        foods_eaten = 0

        eps = eps_end + (eps_start - eps_end) * max(
            0.0, 1.0 - episode / eps_decay_episodes
        )

        while True:
            if random.random() < eps:
                action = random.randrange(4)
            else:
                with torch.no_grad():
                    q = policy_net(torch.from_numpy(state).unsqueeze(0))
                    action = int(torch.argmax(q, dim=1).item())

            next_state, reward, done = env.step(action)
            if reward == 10.0:
                foods_eaten += 1
            ep_reward += reward
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            global_step += 1

            if len(buffer) >= min_buffer and global_step % train_every == 0:
                s, a, r, s2, d = buffer.sample(batch_size)
                s = torch.from_numpy(s)
                a = torch.from_numpy(a)
                r = torch.from_numpy(r)
                s2 = torch.from_numpy(s2)
                d = torch.from_numpy(d)

                q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(s2).max(dim=1)[0]
                    target = r + gamma * next_q * (1.0 - d)

                loss = nn.functional.smooth_l1_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        recent_scores.append(foods_eaten)

        if episode % 100 == 0:
            avg_score = sum(recent_scores) / len(recent_scores)
            elapsed = time.time() - start_time
            print(
                f"episode {episode}/{total_episodes}  eps={eps:.3f}  "
                f"avg_food(last100)={avg_score:.2f}  elapsed={elapsed:.1f}s"
            )

        while pending_pcts and episode >= checkpoint_episodes[pending_pcts[0]]:
            pct = pending_pcts.pop(0)
            path = os.path.join(out_dir, f"checkpoint_{pct:03d}.json")
            export_checkpoint(policy_net, pct, path)
            print(f"Exported checkpoint_{pct:03d}.json at episode {episode}")

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.1f}s")

    manifest = {"checkpoints": checkpoint_pcts}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    print("Wrote manifest.json")


if __name__ == "__main__":
    train()
