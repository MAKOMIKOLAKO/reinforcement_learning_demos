/**
 * shared/rl-diagrams.js
 *
 * Renders the RL Demos site's data-flow diagrams (agent-environment loop,
 * DQN, REINFORCE, PPO) using Mermaid.js instead of hand-placed SVG. Mermaid
 * owns layout, so diagrams stay readable as node/edge count grows instead of
 * relying on manually guessed coordinates.
 *
 * Exposed API:
 *   DIAGRAMS                                  -> { loop, dqn, reinforce, ppo }
 *   renderDiagram(containerId, diagramKey)     -> Promise<void>
 */

import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

export const DIAGRAMS = {
  loop: `flowchart LR
    A[Agent] -->|action| E[Environment]
    E -->|state| A
    E -->|reward| A`,

  dqn: `flowchart LR
    S["State<br/>dim = 11"] --> Q[Q-network]
    Q -->|Q-value per action| AM{argmax}
    AM --> ACT[Action]
    ACT -->|env step| TR["Transition (s, a, r, s')"]
    TR --> RB[("Replay Buffer")]
    RB -->|sampled batch| LOSS["Loss (TD error)"]
    Q -.->|copy every N steps| TGT["Target Network<br/>(periodic copy)"]
    TGT -->|target Q-value| LOSS
    LOSS -->|gradient update| Q`,

  reinforce: `flowchart LR
    S[State] --> P[Policy network]
    P -->|action probabilities| SAMPLE[Sample action]
    SAMPLE --> ACT[Action]
    ACT --> ENV[Environment step]
    ENV -->|reward, next state| TRAJ["Trajectory buffer<br/>(full episode)"]
    TRAJ -->|episode ends| RET["Compute returns<br/>(discounted future reward)"]
    RET -->|policy gradient| P`,

  ppo: `flowchart LR
    S[State] --> ACTOR[Actor network]
    S --> CRITIC[Critic network]
    ACTOR -->|action distribution| SAMPLE[Sample action]
    CRITIC -->|"value estimate V(s)"| ADV[Advantage calculation]
    SAMPLE --> ACT[Action]
    ACT --> ENV[Environment step]
    ENV -->|reward, next state| ADV
    ADV --> CLIP["Clipped surrogate objective"]
    CLIP -->|update| ACTOR
    ADV -->|value loss| CRITIC`,
};

let initialized = false;
let renderCount = 0;

/** Reads a CSS custom property off :root, falling back if unset/inaccessible. */
function cssVar(name, fallback) {
  try {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  } catch {
    return fallback;
  }
}

function initMermaid() {
  if (initialized) return;

  mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    securityLevel: "strict",
    themeVariables: {
      fontFamily: cssVar("--font-sans", "-apple-system, BlinkMacSystemFont, sans-serif"),
      background: cssVar("--bg-elevated", "#ffffff"),
      primaryColor: cssVar("--bg", "#eef2ff"),
      primaryTextColor: cssVar("--text", "#1a1a1e"),
      primaryBorderColor: cssVar("--accent", "#3b6ef6"),
      secondaryColor: cssVar("--bg-elevated", "#ffffff"),
      secondaryBorderColor: cssVar("--border", "#e0e0e5"),
      tertiaryColor: cssVar("--bg", "#f7f7f8"),
      tertiaryBorderColor: cssVar("--border", "#e0e0e5"),
      lineColor: cssVar("--text-muted", "#5a5a63"),
      textColor: cssVar("--text", "#1a1a1e"),
      mainBkg: cssVar("--bg", "#eef2ff"),
      nodeBorder: cssVar("--accent", "#3b6ef6"),
      clusterBkg: cssVar("--bg", "#f7f7f8"),
      edgeLabelBackground: cssVar("--bg-elevated", "#ffffff"),
    },
  });

  initialized = true;
}

/**
 * Renders DIAGRAMS[diagramKey] as SVG and injects it into the element with
 * id containerId, replacing any existing content there.
 *
 * @param {string} containerId
 * @param {"loop"|"dqn"|"reinforce"|"ppo"} diagramKey
 * @returns {Promise<void>}
 */
export async function renderDiagram(containerId, diagramKey) {
  const container = document.getElementById(containerId);
  if (!container) {
    throw new Error(`renderDiagram: no element with id "${containerId}"`);
  }

  const definition = DIAGRAMS[diagramKey];
  if (!definition) {
    throw new Error(`renderDiagram: unknown diagram key "${diagramKey}"`);
  }

  initMermaid();

  const renderId = `rl-diagram-${diagramKey}-${renderCount++}`;
  const { svg } = await mermaid.render(renderId, definition);
  container.innerHTML = svg;

  const svgEl = container.querySelector("svg");
  if (svgEl) {
    svgEl.classList.add("rl-diagram");
    svgEl.removeAttribute("width");
    svgEl.removeAttribute("height");
  }
}
