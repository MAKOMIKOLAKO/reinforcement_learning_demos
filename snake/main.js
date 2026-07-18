/**
 * snake/main.js
 *
 * Wires the pure Snake env (./game.js) up to a canvas renderer, playback
 * controls, and the shared checkpoint loader/forward-pass engine
 * (../shared/inference.js). Loads a checkpoint for the slider's current
 * training-progress percentage, then continuously rolls out
 * encode -> predict -> argmax -> step, auto-restarting episodes forever so
 * the demo keeps animating.
 */

import { loadCheckpoint, predict } from "../shared/inference.js";
import { GRID_SIZE, createInitialState, encodeState, stepState, argmaxAction } from "./game.js";

const CHECKPOINT_PCTS = [0, 5, 10, 25, 50, 75, 100];
const CHECKPOINT_PATH = (pct) =>
  `../public/checkpoints/snake/dqn/checkpoint_${String(pct).padStart(3, "0")}.json`;

const STEPS_PER_SECOND = 8;
const STEP_INTERVAL_MS = 1000 / STEPS_PER_SECOND;

const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const cellSize = canvas.width / GRID_SIZE;

const playPauseBtn = document.getElementById("playPauseBtn");
const stepBtn = document.getElementById("stepBtn");
const restartBtn = document.getElementById("restartBtn");
const progressSlider = document.getElementById("progressSlider");
const progressOutput = document.getElementById("progressOutput");

const statCheckpoint = document.getElementById("statCheckpoint");
const statScore = document.getElementById("statScore");
const statSteps = document.getElementById("statSteps");
const statStatus = document.getElementById("statStatus");

let checkpoint = null;
let gameState = createInitialState();
let playing = true;
let lastStepTime = 0;
let rafHandle = null;

/**
 * Read-only accessor for the currently loaded checkpoint and live game
 * state, so other modules on this page (e.g. the stepper widget) can run
 * their own forward passes against exactly what's on screen right now,
 * without duplicating the rollout loop above.
 */
export function getCurrentInference() {
  return { checkpoint, gameState };
}

/** Draws the current game state to the canvas. */
function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Food.
  if (gameState.food) {
    ctx.fillStyle = "#e0554d";
    ctx.fillRect(
      gameState.food.x * cellSize + 1,
      gameState.food.y * cellSize + 1,
      cellSize - 2,
      cellSize - 2
    );
  }

  // Snake body.
  gameState.snake.forEach((segment, index) => {
    ctx.fillStyle = index === 0 ? "#7ee787" : "#3fa34d";
    ctx.fillRect(
      segment.x * cellSize + 1,
      segment.y * cellSize + 1,
      cellSize - 2,
      cellSize - 2
    );
  });
}

/** Updates the stats panel text. */
function renderStats() {
  statScore.textContent = String(gameState.score);
  statSteps.textContent = String(gameState.stepCount);
  statStatus.textContent = gameState.alive ? "running" : "restarting…";
}

/** Runs one env step using the loaded checkpoint's greedy action. */
function advanceOneStep() {
  if (!checkpoint) {
    return;
  }
  if (!gameState.alive) {
    gameState = createInitialState();
    return;
  }
  const observation = encodeState(gameState);
  const qValues = predict(checkpoint, observation);
  const action = argmaxAction(qValues);
  stepState(gameState, action);
}

function stepAndRender() {
  advanceOneStep();
  render();
  renderStats();
}

function animate(timestamp) {
  if (playing && timestamp - lastStepTime >= STEP_INTERVAL_MS) {
    lastStepTime = timestamp;
    stepAndRender();
  }
  rafHandle = requestAnimationFrame(animate);
}

function setPlaying(nextPlaying) {
  playing = nextPlaying;
  playPauseBtn.textContent = playing ? "Pause" : "Play";
}

async function loadCheckpointForSliderValue(sliderValue) {
  const pct = CHECKPOINT_PCTS[Number(sliderValue)];
  progressOutput.textContent = `${pct}%`;
  statCheckpoint.textContent = `loading ${pct}%…`;
  try {
    checkpoint = await loadCheckpoint(CHECKPOINT_PATH(pct));
    statCheckpoint.textContent = `${pct}% trained`;
  } catch (err) {
    statCheckpoint.textContent = `failed to load ${pct}%`;
    checkpoint = null;
    console.error(err);
  }
  gameState = createInitialState();
  render();
  renderStats();
  // Let other modules on the page (the stepper widget) know a new
  // checkpoint is live so they can re-run inference against it.
  document.dispatchEvent(new CustomEvent("snake:checkpointchanged"));
}

playPauseBtn.addEventListener("click", () => {
  setPlaying(!playing);
});

stepBtn.addEventListener("click", () => {
  stepAndRender();
});

restartBtn.addEventListener("click", () => {
  gameState = createInitialState();
  render();
  renderStats();
});

progressSlider.addEventListener("input", () => {
  loadCheckpointForSliderValue(progressSlider.value);
});

// Initial load: slider defaults to the last (100%) checkpoint.
loadCheckpointForSliderValue(progressSlider.value);
render();
renderStats();
rafHandle = requestAnimationFrame(animate);
