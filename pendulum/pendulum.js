/**
 * pendulum/pendulum.js
 *
 * Drives the Pendulum/PPO demo page: reimplements Gym's Pendulum-v1 physics,
 * runs a live rollout using a checkpoint loaded via shared/inference.js, and
 * renders the pendulum + applied torque on a canvas.
 */

import { loadCheckpoint, predict } from "../shared/inference.js";

// ---------------------------------------------------------------------------
// Gym Pendulum-v1 physics constants
// ---------------------------------------------------------------------------
const G = 10.0;
const MASS = 1.0;
const LENGTH = 1.0;
const DT = 0.05;
const MAX_SPEED = 8.0;
const MAX_TORQUE = 2.0;
const MAX_STEPS = 200;

const MANIFEST_PATH = "../public/checkpoints/pendulum/ppo/manifest.json";
const checkpointPath = (pct) =>
  `../public/checkpoints/pendulum/ppo/checkpoint_${String(pct).padStart(3, "0")}.json`;

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const canvas = document.getElementById("stage-canvas");
const ctx = canvas.getContext("2d");
const playPauseBtn = document.getElementById("play-pause-btn");
const stepBtn = document.getElementById("step-btn");
const restartBtn = document.getElementById("restart-btn");
const progressSlider = document.getElementById("progress-slider");
const progressOutput = document.getElementById("progress-output");
const checkpointBadge = document.getElementById("checkpoint-badge");
const stepStat = document.getElementById("stat-step");
const angleStat = document.getElementById("stat-angle");
const speedStat = document.getElementById("stat-speed");
const torqueStat = document.getElementById("stat-torque");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let checkpoint = null;
let manifest = null;
let theta = 0;
let thetaDot = 0;
let stepCount = 0;
let lastTorque = 0;
let playing = true;
let rafId = null;
let lastFrameTime = null;
const STEP_INTERVAL_MS = DT * 1000; // real-time-ish playback

function resetState() {
  theta = Math.random() * 2 * Math.PI - Math.PI; // uniform [-pi, pi]
  thetaDot = Math.random() * 2 - 1; // uniform [-1, 1]
  stepCount = 0;
  lastTorque = 0;
}

// ---------------------------------------------------------------------------
// Physics: exact port of Gym's Pendulum-v1 step()
// ---------------------------------------------------------------------------
function clip(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

function angleNormalize(angle) {
  // Wrap to (-pi, pi] purely for display purposes; physics state (theta)
  // itself is left unbounded, matching Gym's internal representation.
  return ((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
}

function stepPhysics(rawTorque) {
  const u = clip(rawTorque, -MAX_TORQUE, MAX_TORQUE);

  const newThetaDotUnclipped =
    thetaDot +
    ((3 * G) / (2 * LENGTH) * Math.sin(theta) +
      (3.0 / (MASS * LENGTH * LENGTH)) * u) *
      DT;
  const newThetaDot = clip(newThetaDotUnclipped, -MAX_SPEED, MAX_SPEED);
  const newTheta = theta + newThetaDot * DT;

  theta = newTheta;
  thetaDot = newThetaDot;
  lastTorque = u;
  stepCount += 1;

  if (stepCount >= MAX_STEPS) {
    resetState();
  }
}

function runInference() {
  if (!checkpoint) return 0;
  const obs = [Math.cos(theta), Math.sin(theta), thetaDot];
  const output = predict(checkpoint, obs);
  return output[0];
}

function advanceOneStep() {
  const torque = runInference();
  stepPhysics(torque);
  updateStats();
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function render() {
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const cx = width / 2;
  const cy = height / 2;
  const rodLength = Math.min(width, height) * 0.35;

  // Rod tip: Gym's convention has theta=0 pointing "up" (rod extends up from
  // pivot), with cos/sin used as observation components. Draw accordingly:
  // x = sin(theta) * L, y = -cos(theta) * L relative to pivot (up = -y).
  const tipX = cx + Math.sin(theta) * rodLength;
  const tipY = cy - Math.cos(theta) * rodLength;

  // Torque indicator: color intensity + a curved arrow scaled by |torque|.
  const torqueFrac = Math.min(1, Math.abs(lastTorque) / MAX_TORQUE);
  const torqueColor = lastTorque >= 0
    ? `rgba(109, 147, 255, ${0.25 + 0.65 * torqueFrac})`
    : `rgba(242, 104, 95, ${0.25 + 0.65 * torqueFrac})`;

  // Torque arc around the pivot indicating direction/magnitude of applied u.
  if (torqueFrac > 0.02) {
    const arcRadius = rodLength * 0.28;
    const direction = lastTorque >= 0 ? 1 : -1; // CCW positive per Gym convention
    const startAngle = -Math.PI / 2;
    const endAngle = startAngle + direction * (0.4 + 1.6 * torqueFrac);

    ctx.beginPath();
    ctx.arc(cx, cy, arcRadius, startAngle, endAngle, direction < 0);
    ctx.strokeStyle = torqueColor;
    ctx.lineWidth = 6;
    ctx.lineCap = "round";
    ctx.stroke();

    // Arrowhead at the end of the arc.
    const headX = cx + arcRadius * Math.cos(endAngle);
    const headY = cy + arcRadius * Math.sin(endAngle);
    const tangent = endAngle + (direction * Math.PI) / 2;
    const headSize = 9;
    ctx.beginPath();
    ctx.moveTo(headX, headY);
    ctx.lineTo(
      headX - headSize * Math.cos(tangent - 0.4),
      headY - headSize * Math.sin(tangent - 0.4)
    );
    ctx.lineTo(
      headX - headSize * Math.cos(tangent + 0.4),
      headY - headSize * Math.sin(tangent + 0.4)
    );
    ctx.closePath();
    ctx.fillStyle = torqueColor;
    ctx.fill();
  }

  // Rod.
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(tipX, tipY);
  ctx.strokeStyle = "#eaeaef";
  ctx.lineWidth = 6;
  ctx.lineCap = "round";
  ctx.stroke();

  // Pivot base.
  ctx.beginPath();
  ctx.arc(cx, cy, 7, 0, 2 * Math.PI);
  ctx.fillStyle = "#8badff";
  ctx.fill();

  // Bob at rod tip, color-coded by torque intensity.
  ctx.beginPath();
  ctx.arc(tipX, tipY, 16, 0, 2 * Math.PI);
  ctx.fillStyle = torqueColor.replace(/[\d.]+\)$/, "1)");
  ctx.fill();
  ctx.strokeStyle = "#0a0a0c";
  ctx.lineWidth = 2;
  ctx.stroke();

  // Upright target line (dashed), for reference.
  ctx.beginPath();
  ctx.setLineDash([4, 6]);
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx, cy - rodLength);
  ctx.strokeStyle = "rgba(161, 161, 170, 0.35)";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.setLineDash([]);
}

function updateStats() {
  stepStat.textContent = `${stepCount} / ${MAX_STEPS}`;
  angleStat.textContent = `${angleNormalize(theta).toFixed(2)} rad`;
  speedStat.textContent = `${thetaDot.toFixed(2)} rad/s`;
  torqueStat.textContent = `${lastTorque.toFixed(2)} N·m`;
}

// ---------------------------------------------------------------------------
// Animation loop
// ---------------------------------------------------------------------------
function frame(timestamp) {
  if (!playing) {
    lastFrameTime = null;
    return;
  }
  if (lastFrameTime === null) {
    lastFrameTime = timestamp;
  }
  const elapsed = timestamp - lastFrameTime;
  if (elapsed >= STEP_INTERVAL_MS) {
    lastFrameTime = timestamp;
    advanceOneStep();
  }
  render();
  rafId = requestAnimationFrame(frame);
}

function startLoop() {
  if (rafId !== null) return;
  playing = true;
  playPauseBtn.textContent = "Pause";
  lastFrameTime = null;
  rafId = requestAnimationFrame(frame);
}

function stopLoop() {
  playing = false;
  playPauseBtn.textContent = "Play";
  if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
}

// ---------------------------------------------------------------------------
// Checkpoint loading
// ---------------------------------------------------------------------------
async function loadCheckpointForPct(pct) {
  checkpointBadge.textContent = "loading…";
  checkpointBadge.classList.remove("error");
  try {
    checkpoint = await loadCheckpoint(checkpointPath(pct));
    checkpointBadge.textContent = `checkpoint ${pct}%`;
  } catch (err) {
    checkpointBadge.textContent = "load failed";
    checkpointBadge.classList.add("error");
    console.error(err);
  }
  resetState();
  render();
  updateStats();
  // Let any listener (e.g. the step-through diagram) know the live checkpoint
  // just changed, so it can re-render against the newly-selected weights.
  document.dispatchEvent(new CustomEvent("pendulum:checkpoint-changed"));
}

// ---------------------------------------------------------------------------
// Accessors for other modules/inline scripts on this page (the flow-stepper
// diagram) that need to read the CURRENT live rollout state/checkpoint
// without duplicating the rollout loop above.
// ---------------------------------------------------------------------------

/**
 * Returns the live observation vector [cos(theta), sin(theta), angular
 * velocity] currently driving the rollout, exactly as fed to predict().
 * @returns {number[]}
 */
export function getLiveObservation() {
  return [Math.cos(theta), Math.sin(theta), thetaDot];
}

/**
 * Returns the currently-loaded checkpoint object (or null if not loaded
 * yet), the same object passed to predict() by the rollout loop.
 * @returns {object|null}
 */
export function getLiveCheckpoint() {
  return checkpoint;
}

async function init() {
  try {
    manifest = await (await fetch(MANIFEST_PATH)).json();
  } catch (err) {
    manifest = { checkpoints: [0, 5, 10, 25, 50, 75, 100] };
    console.error("Failed to load manifest, using default checkpoints.", err);
  }

  const pcts = manifest.checkpoints;
  progressSlider.min = "0";
  progressSlider.max = String(pcts.length - 1);
  progressSlider.step = "1";
  progressSlider.value = String(pcts.length - 1); // default to fully trained

  const applySliderValue = async () => {
    const idx = Number(progressSlider.value);
    const pct = pcts[idx];
    progressOutput.textContent = `${pct}%`;
    await loadCheckpointForPct(pct);
    startLoop();
  };

  progressSlider.addEventListener("input", () => {
    stopLoop();
    applySliderValue();
  });

  await applySliderValue();
  startLoop();
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
playPauseBtn.addEventListener("click", () => {
  if (playing) {
    stopLoop();
  } else {
    startLoop();
  }
});

stepBtn.addEventListener("click", () => {
  stopLoop();
  advanceOneStep();
  render();
});

restartBtn.addEventListener("click", () => {
  resetState();
  render();
  updateStats();
  startLoop();
});

init();
