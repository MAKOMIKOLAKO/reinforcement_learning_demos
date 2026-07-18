/**
 * shared/flow-stepper.js
 *
 * Dependency-free vanilla JS ES module implementing an animated step-through
 * widget: a horizontal row of circular stage nodes connected by a track,
 * with a dot that animates to the current stage, a detail panel below that
 * each stage's render() fills with a live visualization + explanatory prose,
 * and Step / Reset buttons.
 *
 * Reuses the site's existing CSS custom properties (--accent, --border,
 * --bg-elevated, --text-muted, --radius, etc. — see shared/style.css) so the
 * widget matches the rest of the site without inventing new visual language.
 *
 * Usage:
 *   import { createStepper } from '../shared/flow-stepper.js';
 *
 *   const stepper = createStepper('my-stepper', [
 *     { key: 'observe', label: 'Observe', render: (el) => { el.innerHTML = '...'; } },
 *     { key: 'predict', label: 'Predict', render: async (el) => { ... } },
 *     ...
 *   ]);
 *
 *   nextButton.addEventListener('click', () => stepper.next());
 *   resetButton.addEventListener('click', () => stepper.reset());
 *
 * Detail-panel content contract for stage authors (see shared/flow-stepper-test.html
 * for a worked example): each render(detailPanelEl) should populate detailPanelEl
 * with, in order:
 *   1. A short heading (e.g. <h3>) naming the stage.
 *   2. A live visualization of computed values — use the helper classes
 *      .flow-stepper-bars / .flow-stepper-bar (bar chart for value/probability
 *      arrays, winner in the accent color via .flow-stepper-bar.winner) and
 *      .flow-stepper-chips / .flow-stepper-chip (small chips for state vectors).
 *   3. A paragraph of explanatory prose, with unfamiliar terms wrapped via
 *      glossaryTerm() from shared/glossary.js (call initGlossary() after
 *      inserting the markup so tooltips wire up).
 */

const STYLE_ID = "flow-stepper-styles";

function injectStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.flow-stepper-track-wrap {
  position: relative;
  padding: 1.75rem 0.5rem 2.25rem;
}

.flow-stepper-track {
  position: relative;
  height: 2px;
  background: var(--border);
  margin: 0 1.25rem;
}

.flow-stepper-nodes {
  position: absolute;
  top: -1.1rem;
  left: 0;
  right: 0;
  display: flex;
  justify-content: space-between;
  margin: 0 1.25rem;
}

.flow-stepper-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.4rem;
}

.flow-stepper-node-circle {
  width: 2.25rem;
  height: 2.25rem;
  border-radius: 50%;
  background: var(--bg-elevated);
  border: 2px solid var(--border);
  color: var(--text-muted);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.85rem;
  font-weight: 600;
  transition: border-color 0.3s ease, color 0.3s ease, background 0.3s ease;
}

.flow-stepper-node.active .flow-stepper-node-circle,
.flow-stepper-node.done .flow-stepper-node-circle {
  border-color: var(--accent);
  color: var(--accent);
}

.flow-stepper-node.active .flow-stepper-node-circle {
  background: var(--accent);
  color: var(--accent-contrast);
}

.flow-stepper-node-label {
  font-size: 0.78rem;
  color: var(--text-muted);
  text-align: center;
  white-space: nowrap;
}

.flow-stepper-node.active .flow-stepper-node-label {
  color: var(--text);
  font-weight: 600;
}

.flow-stepper-dot {
  position: absolute;
  top: 50%;
  width: 0.7rem;
  height: 0.7rem;
  border-radius: 50%;
  background: var(--accent);
  transform: translate(-50%, -50%);
  transition: left 0.45s ease;
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 25%, transparent);
}

.flow-stepper-panel {
  background: var(--bg-elevated);
  border-radius: var(--radius);
  padding: 1rem;
  border: 1px solid var(--border);
}

.flow-stepper-panel h3 {
  margin-top: 0;
}

.flow-stepper-controls {
  display: flex;
  gap: 0.5rem;
  margin-top: 1rem;
}

/* Helper visualization primitives for stage render() implementations */
.flow-stepper-bars {
  display: flex;
  align-items: flex-end;
  gap: 0.6rem;
  height: 6rem;
  margin: 0.75rem 0;
}

.flow-stepper-bar-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.3rem;
  flex: 1;
  height: 100%;
  justify-content: flex-end;
}

.flow-stepper-bar {
  width: 100%;
  max-width: 2.5rem;
  background: var(--border);
  border-radius: 4px 4px 0 0;
  transition: height 0.3s ease, background 0.3s ease;
}

.flow-stepper-bar.winner {
  background: var(--accent);
}

.flow-stepper-bar-value {
  font-size: 0.72rem;
  color: var(--text-muted);
  font-family: var(--font-mono);
}

.flow-stepper-bar-label {
  font-size: 0.75rem;
  color: var(--text-muted);
  text-align: center;
}

.flow-stepper-chips {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(3.5rem, 1fr));
  gap: 0.4rem;
  margin: 0.75rem 0;
}

.flow-stepper-chip {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.35rem 0.4rem;
  text-align: center;
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: var(--text);
}

.flow-stepper-chip-label {
  display: block;
  font-family: var(--font-sans);
  font-size: 0.65rem;
  color: var(--text-muted);
  margin-bottom: 0.15rem;
}

.flow-stepper-callout {
  margin: 0.75rem 0;
  padding: 0.6rem 0.85rem;
  border-radius: calc(var(--radius) - 2px);
  background: color-mix(in srgb, var(--accent) 14%, transparent);
  border: 1px solid var(--accent);
  color: var(--text);
  font-size: 0.9rem;
}
`;
  document.head.appendChild(style);
}

/**
 * Creates and mounts a step-through widget into the element with id
 * `containerId`.
 *
 * @param {string} containerId - id of the DOM element to render into.
 * @param {Array<{key: string, label: string, render: (detailPanelEl: HTMLElement) => (void|Promise<void>)}>} stages
 * @returns {{ next: () => void, reset: () => void, stepTo: (index: number) => void }}
 */
export function createStepper(containerId, stages) {
  injectStyles();

  const container = document.getElementById(containerId);
  if (!container) {
    throw new Error(`flow-stepper: no element with id "${containerId}"`);
  }
  if (!Array.isArray(stages) || stages.length === 0) {
    throw new Error("flow-stepper: stages must be a non-empty array");
  }

  container.classList.add("flow-stepper");
  container.innerHTML = `
    <div class="flow-stepper-track-wrap">
      <div class="flow-stepper-nodes"></div>
      <div class="flow-stepper-track">
        <div class="flow-stepper-dot"></div>
      </div>
    </div>
    <div class="flow-stepper-panel"></div>
    <div class="flow-stepper-controls">
      <button type="button" data-flow-stepper-next>Step</button>
      <button type="button" data-flow-stepper-reset>Reset</button>
    </div>
  `;

  const nodesEl = container.querySelector(".flow-stepper-nodes");
  const dotEl = container.querySelector(".flow-stepper-dot");
  const panelEl = container.querySelector(".flow-stepper-panel");
  const nextBtn = container.querySelector("[data-flow-stepper-next]");
  const resetBtn = container.querySelector("[data-flow-stepper-reset]");

  const nodeEls = stages.map((stage, index) => {
    const node = document.createElement("div");
    node.className = "flow-stepper-node";
    node.dataset.key = stage.key;
    node.innerHTML = `
      <div class="flow-stepper-node-circle">${index + 1}</div>
      <div class="flow-stepper-node-label">${stage.label}</div>
    `;
    nodesEl.appendChild(node);
    return node;
  });

  let currentIndex = 0;
  // Bumped on every render call so a stale async render() can't clobber a
  // newer one if the user steps quickly.
  let renderToken = 0;

  function positionDot() {
    const pct =
      stages.length === 1 ? 0 : (currentIndex / (stages.length - 1)) * 100;
    dotEl.style.left = `${pct}%`;
  }

  function updateNodes() {
    nodeEls.forEach((node, index) => {
      node.classList.toggle("active", index === currentIndex);
      node.classList.toggle("done", index < currentIndex);
    });
  }

  async function renderCurrentStage() {
    const token = ++renderToken;
    const stage = stages[currentIndex];
    panelEl.innerHTML = "";
    await stage.render(panelEl);
    if (token !== renderToken) return; // superseded by a later step
  }

  function goTo(index) {
    if (index < 0 || index >= stages.length) return;
    currentIndex = index;
    positionDot();
    updateNodes();
    renderCurrentStage();
  }

  function next() {
    if (currentIndex < stages.length - 1) {
      goTo(currentIndex + 1);
    }
  }

  function reset() {
    goTo(0);
  }

  function stepTo(index) {
    goTo(index);
  }

  nextBtn.addEventListener("click", next);
  resetBtn.addEventListener("click", reset);

  // Initial render.
  positionDot();
  updateNodes();
  renderCurrentStage();

  return { next, reset, stepTo };
}
