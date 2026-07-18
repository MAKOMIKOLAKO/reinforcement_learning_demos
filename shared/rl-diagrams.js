/**
 * shared/rl-diagrams.js
 *
 * Dependency-free SVG diagram library for the RL Demos site. Each demo page
 * (DQN/Snake, REINFORCE/CartPole, PPO/Pendulum) imports the render function it
 * needs and drops the returned markup into a container element. Diagrams are
 * plain template-string SVG — no canvas, no external charting library, no
 * build step. Every render function accepts a small options object so a
 * consuming page can relabel dimensions/action names without needing to
 * redraw the diagram structure itself.
 *
 * Exposed API:
 *   renderAgentEnvLoop(options)  -> string (SVG markup)
 *   renderDQNDiagram(options)    -> string (SVG markup)
 *   renderREINFORCEDiagram(options) -> string (SVG markup)
 *   renderPPODiagram(options)    -> string (SVG markup)
 *   mountDiagram(container, svgMarkup) -> void   (helper: injects markup)
 *
 * All four render functions return a self-contained <svg>...</svg> string
 * sized with a viewBox, so it scales responsively when the caller sets
 * width: 100% / max-width in CSS.
 */

// ---------------------------------------------------------------------------
// Shared low-level SVG building blocks
// ---------------------------------------------------------------------------

/** Escapes text for safe interpolation into SVG/HTML attribute or text content. */
function escapeText(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Draws a labeled rectangular box centered at (cx, cy) with the given width
 * and height. `variant` selects a style: "default" (solid network/box),
 * "muted" (dashed, duller — used for target-network shadow copies), or
 * "accent" (highlighted — used for the PPO clipped-objective node).
 */
function box(cx, cy, width, height, label, opts = {}) {
  const { variant = "default", subLabel = "", id = "" } = opts;
  const x = cx - width / 2;
  const y = cy - height / 2;

  const styles = {
    default: {
      fill: "var(--diagram-box-fill, #eef2ff)",
      stroke: "var(--diagram-box-stroke, #4338ca)",
      dash: "",
      textFill: "var(--diagram-text, #1e1b4b)",
    },
    muted: {
      fill: "var(--diagram-muted-fill, #f3f4f6)",
      stroke: "var(--diagram-muted-stroke, #9ca3af)",
      dash: "6,4",
      textFill: "var(--diagram-muted-text, #6b7280)",
    },
    accent: {
      fill: "var(--diagram-accent-fill, #fef3c7)",
      stroke: "var(--diagram-accent-stroke, #b45309)",
      dash: "",
      textFill: "var(--diagram-accent-text, #78350f)",
    },
    node: {
      fill: "var(--diagram-node-fill, #ecfdf5)",
      stroke: "var(--diagram-node-stroke, #047857)",
      dash: "",
      textFill: "var(--diagram-node-text, #064e3b)",
    },
  };
  const style = styles[variant] || styles.default;

  const labelLines = String(label).split("\n");
  const lineHeight = 15;
  const firstLineY = cy - ((labelLines.length - 1) * lineHeight) / 2 + (subLabel ? -6 : 0);
  const textLines = labelLines
    .map(
      (line, i) =>
        `<tspan x="${cx}" y="${firstLineY + i * lineHeight}">${escapeText(line)}</tspan>`
    )
    .join("");

  const subLabelMarkup = subLabel
    ? `<text x="${cx}" y="${y + height - 8}" text-anchor="middle" font-size="10" fill="${style.textFill}" opacity="0.8">${escapeText(
        subLabel
      )}</text>`
    : "";

  return `
    <g${id ? ` id="${escapeText(id)}"` : ""}>
      <rect x="${x}" y="${y}" width="${width}" height="${height}" rx="8"
        fill="${style.fill}" stroke="${style.stroke}" stroke-width="2"
        ${style.dash ? `stroke-dasharray="${style.dash}"` : ""} />
      <text x="${cx}" text-anchor="middle" font-size="13" font-weight="600" fill="${style.textFill}">${textLines}</text>
      ${subLabelMarkup}
    </g>
  `;
}

/** Draws a small labeled circular/oval "step" node (e.g. argmax, sample, sum). */
function stepNode(cx, cy, label, opts = {}) {
  const { rx = 34, ry = 20, variant = "node" } = opts;
  return box(cx, cy, rx * 2, ry * 2, label, { variant, id: opts.id });
}

/**
 * Draws an arrow from (x1,y1) to (x2,y2) with an optional text label placed
 * at the midpoint (offset above/below the line via labelDy).
 */
function arrow(x1, y1, x2, y2, label = "", opts = {}) {
  const { labelDy = -6, dashed = false, color = "var(--diagram-arrow, #374151)", curve = null } = opts;
  const markerId = "rl-diagram-arrowhead";
  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2;

  const path = curve
    ? `M ${x1} ${y1} Q ${curve.cx} ${curve.cy} ${x2} ${y2}`
    : `M ${x1} ${y1} L ${x2} ${y2}`;

  const labelX = curve ? curve.cx : midX;
  const labelY = curve ? curve.cy : midY;

  const labelMarkup = label
    ? `<text x="${labelX}" y="${labelY + labelDy}" text-anchor="middle" font-size="11" font-style="italic" fill="${color}">${escapeText(
        label
      )}</text>`
    : "";

  return `
    <path d="${path}" fill="none" stroke="${color}" stroke-width="2"
      ${dashed ? 'stroke-dasharray="5,4"' : ""}
      marker-end="url(#${markerId})" />
    ${labelMarkup}
  `;
}

/** Shared <defs> block with the arrowhead marker, included once per SVG. */
function defsBlock() {
  return `
    <defs>
      <marker id="rl-diagram-arrowhead" viewBox="0 0 10 10" refX="9" refY="5"
        markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" />
      </marker>
    </defs>
  `;
}

/** Wraps inner SVG content in a root <svg> element with the given viewBox. */
function svgRoot(width, height, title, innerMarkup) {
  return `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg"
      role="img" aria-label="${escapeText(title)}" class="rl-diagram">
    <title>${escapeText(title)}</title>
    ${defsBlock()}
    ${innerMarkup}
  </svg>`;
}

// ---------------------------------------------------------------------------
// 1. Base agent-environment loop
// ---------------------------------------------------------------------------

/**
 * Renders the canonical agent-environment interaction loop: Agent and
 * Environment boxes connected by state / action / reward arrows. This is the
 * shared foundation the DQN/REINFORCE/PPO diagrams build outward from.
 *
 * @param {object} [options]
 * @param {string} [options.agentLabel="Agent"]
 * @param {string} [options.envLabel="Environment"]
 * @param {string} [options.stateLabel="state"]
 * @param {string} [options.actionLabel="action"]
 * @param {string} [options.rewardLabel="reward, next state"]
 * @returns {string} SVG markup string.
 */
export function renderAgentEnvLoop(options = {}) {
  const {
    agentLabel = "Agent",
    envLabel = "Environment",
    stateLabel = "state",
    actionLabel = "action",
    rewardLabel = "reward, next state",
  } = options;

  const width = 420;
  const height = 220;
  const agentX = 110;
  const envX = 310;
  const boxY = 110;

  const inner = `
    ${box(agentX, boxY, 140, 70, agentLabel)}
    ${box(envX, boxY, 140, 70, envLabel)}
    ${arrow(envX - 70, boxY - 28, agentX + 70, boxY - 28, stateLabel, { labelDy: -8 })}
    ${arrow(agentX + 70, boxY + 28, envX - 70, boxY + 28, actionLabel, { labelDy: 18 })}
    ${arrow(envX - 70, boxY + 40, agentX + 70, boxY + 40, rewardLabel, {
      labelDy: 32,
      color: "var(--diagram-reward-arrow, #b45309)",
    })}
  `;

  return svgRoot(width, height, "Agent-environment interaction loop", inner);
}

// ---------------------------------------------------------------------------
// 2. DQN (value-based) flow diagram
// ---------------------------------------------------------------------------

/**
 * Renders the DQN flow: state -> Q-network -> per-action Q-values -> argmax
 * -> action; the (s, a, r, s') transition -> Replay Buffer; sampled batches
 * from the buffer plus a Target Network (dashed/duller shadow copy of the
 * Q-network) combine into a loss that updates the Q-network. Exactly one
 * learned network is drawn (plus its target shadow copy) — no separate
 * policy box.
 *
 * @param {object} [options]
 * @param {number} [options.stateDim=4] - Dimensionality of the state vector,
 *   shown as a sub-label on the state input.
 * @param {string[]} [options.actions=["a1","a2","a3"]] - Action names, shown
 *   as the per-action Q-value outputs of the Q-network.
 * @returns {string} SVG markup string.
 */
export function renderDQNDiagram(options = {}) {
  const { stateDim = 4, actions = ["a1", "a2", "a3"] } = options;

  const width = 640;
  const height = 400;

  const stateX = 60;
  const qNetX = 210;
  const argmaxX = 370;
  const actionX = 500;
  const topY = 70;

  const bufferX = 210;
  const bufferY = 220;
  const targetX = 420;
  const targetY = 220;
  const lossX = 320;
  const lossY = 320;

  const qValuesLabel = actions.map((a) => `Q(s, ${a})`).join("\n");

  const inner = `
    ${box(stateX, topY, 90, 56, "state", { subLabel: `dim = ${stateDim}` })}
    ${box(qNetX, topY, 130, 70, "Q-network")}
    ${box(argmaxX, topY, 90, 56, "argmax", { variant: "node" })}
    ${box(actionX, topY, 90, 56, "action")}

    ${arrow(stateX + 45, topY, qNetX - 65, topY)}
    ${arrow(qNetX + 65, topY, argmaxX - 45, topY, qValuesLabel.split("\n")[0] + " ...", { labelDy: -14 })}
    ${arrow(argmaxX + 45, topY, actionX - 45, topY)}

    ${box(bufferX, bufferY, 150, 60, "Replay Buffer", { subLabel: "(s, a, r, s')" })}
    ${arrow(actionX, topY + 28, bufferX + 70, bufferY - 20, "transition", {
      curve: { cx: 420, cy: 150 },
      labelDy: -6,
    })}

    ${box(targetX, targetY, 130, 70, "Target Network\n(periodic copy)", { variant: "muted" })}
    ${arrow(qNetX + 20, topY + 35, targetX - 40, targetY - 35, "copy every N steps", {
      dashed: true,
      color: "var(--diagram-muted-stroke, #9ca3af)",
      curve: { cx: 300, cy: 150 },
      labelDy: -6,
    })}

    ${box(lossX, lossY, 140, 60, "Loss\n(TD error)", { variant: "accent" })}
    ${arrow(bufferX + 40, bufferY + 30, lossX - 50, lossY - 25, "sampled batch", { labelDy: -6 })}
    ${arrow(targetX - 40, targetY + 30, lossX + 50, lossY - 25, "target Q-value", { labelDy: -6 })}
    ${arrow(lossX, lossY - 30, qNetX + 15, topY + 36, "gradient update", {
      color: "var(--diagram-update-arrow, #b91c1c)",
      curve: { cx: 150, cy: 260 },
      labelDy: 14,
    })}
  `;

  return svgRoot(width, height, "DQN value-based learning flow", inner);
}

// ---------------------------------------------------------------------------
// 3. REINFORCE (policy-based) flow diagram
// ---------------------------------------------------------------------------

/**
 * Renders the REINFORCE flow: state -> Policy Network -> action distribution
 * -> explicit sample step -> action. A full episode trajectory (chain of
 * state->action->reward steps) accumulates into a Trajectory Buffer before
 * any learning happens; a Return is then computed per step and used to
 * weight the policy gradient update. Deliberately has NO value network.
 *
 * @param {object} [options]
 * @param {number} [options.stateDim=4]
 * @param {string[]} [options.actions=["a1","a2"]] - Action names shown in the
 *   distribution output.
 * @param {number} [options.episodeSteps=3] - How many state->action->reward
 *   links to draw in the trajectory chain (purely illustrative).
 * @returns {string} SVG markup string.
 */
export function renderREINFORCEDiagram(options = {}) {
  const { stateDim = 4, actions = ["a1", "a2"], episodeSteps = 3 } = options;

  const width = 720;
  const height = 420;

  const stateX = 60;
  const policyX = 210;
  const sampleX = 350;
  const actionX = 470;
  const topY = 60;

  const distLabel = `P(${actions.join(" | s), P(")} | s)`;

  // Trajectory chain: a row of small step nodes below the top row.
  const chainY = 200;
  const chainStartX = 90;
  const chainSpacing = 130;
  let chainMarkup = "";
  for (let i = 0; i < episodeSteps; i++) {
    const cx = chainStartX + i * chainSpacing;
    chainMarkup += box(cx, chainY, 100, 50, `s${i}, a${i}, r${i}`, { variant: "node" });
    if (i > 0) {
      chainMarkup += arrow(cx - chainSpacing + 50, chainY, cx - 50, chainY);
    }
  }
  const lastChainX = chainStartX + (episodeSteps - 1) * chainSpacing;

  const bufferX = lastChainX + 110;
  const bufferY = chainY;
  const returnX = bufferX;
  const returnY = 300;
  const updateY = 380;

  const inner = `
    ${box(stateX, topY, 90, 56, "state", { subLabel: `dim = ${stateDim}` })}
    ${box(policyX, topY, 140, 70, "Policy Network", { subLabel: "outputs distribution" })}
    ${stepNode(sampleX, topY, "sample", { rx: 42, ry: 24 })}
    ${box(actionX, topY, 90, 56, "action")}

    ${arrow(stateX + 45, topY, policyX - 70, topY)}
    ${arrow(policyX + 70, topY, sampleX - 42, topY, distLabel, { labelDy: -14 })}
    ${arrow(sampleX + 42, topY, actionX - 45, topY)}

    ${arrow(actionX, topY + 28, lastChainX, chainY - 28, "episode unfolds", {
      curve: { cx: (actionX + lastChainX) / 2, cy: (topY + chainY) / 2 },
      labelDy: -6,
    })}

    ${chainMarkup}

    ${box(bufferX, bufferY, 150, 60, "Trajectory Buffer", { subLabel: "whole episode" })}
    ${arrow(lastChainX + 50, chainY, bufferX - 75, bufferY)}

    ${box(returnX, returnY, 150, 60, "Return G_t", { variant: "accent", subLabel: "sum of discounted future r" })}
    ${arrow(bufferX, bufferY + 30, returnX, returnY - 30, "after episode ends")}

    ${box((policyX + stateX) / 2, updateY, 220, 60, "Policy Gradient Update", { variant: "accent" })}
    ${arrow(returnX - 40, returnY + 20, (policyX + stateX) / 2 + 60, updateY - 25, "weights gradient", {
      curve: { cx: 300, cy: 350 },
      labelDy: -6,
    })}
    ${arrow((policyX + stateX) / 2, updateY - 30, policyX - 10, topY + 36, "", {
      color: "var(--diagram-update-arrow, #b91c1c)",
      curve: { cx: 60, cy: 300 },
    })}
  `;

  return svgRoot(width, height, "REINFORCE policy-based learning flow", inner);
}

// ---------------------------------------------------------------------------
// 4. PPO (actor-critic) flow diagram
// ---------------------------------------------------------------------------

/**
 * Renders the PPO flow: state feeds TWO parallel networks — Actor (action
 * distribution, like REINFORCE's policy network) and Critic (single state
 * value estimate). The critic's value estimate plus observed rewards feed an
 * Advantage node; the advantage feeds a Clipped Surrogate Objective node
 * (drawn distinctly, not just labeled "loss") that updates the actor. The
 * critic updates separately from its own value-prediction error.
 *
 * @param {object} [options]
 * @param {number} [options.stateDim=4]
 * @param {string[]} [options.actions=["a1","a2"]]
 * @returns {string} SVG markup string.
 */
export function renderPPODiagram(options = {}) {
  const { stateDim = 4, actions = ["a1", "a2"] } = options;

  const width = 680;
  const height = 460;

  const stateX = 70;
  const netX = 260;
  const actorY = 70;
  const criticY = 200;
  const actionX = 440;
  const advX = 440;
  const advY = criticY;
  const objX = 580;
  const objY = actorY;
  const valLossX = 580;
  const valLossY = criticY;

  const distLabel = `P(${actions.join(" | s), P(")} | s)`;

  const inner = `
    ${box(stateX, (actorY + criticY) / 2, 90, 56, "state", { subLabel: `dim = ${stateDim}` })}

    ${box(netX, actorY, 130, 64, "Actor", { subLabel: "action distribution" })}
    ${box(netX, criticY, 130, 64, "Critic", { subLabel: "value estimate V(s)" })}

    ${arrow(stateX + 45, (actorY + criticY) / 2, netX - 65, actorY + 10, "", { curve: { cx: 150, cy: 100 } })}
    ${arrow(stateX + 45, (actorY + criticY) / 2, netX - 65, criticY - 10, "", { curve: { cx: 150, cy: 170 } })}

    ${box(actionX, actorY, 100, 56, "action", { subLabel: distLabel })}
    ${arrow(netX + 65, actorY, actionX - 50, actorY)}

    ${stepNode(advX, advY, "Advantage\nA = G - V(s)", { rx: 60, ry: 32, variant: "accent" })}
    ${arrow(netX + 65, criticY, advX - 60, advY, "V(s)", { labelDy: -10 })}
    ${arrow(actionX, actorY + 28, advX + 20, advY - 32, "reward", {
      curve: { cx: 500, cy: 130 },
      labelDy: -6,
    })}

    ${box(objX, objY, 170, 64, "Clipped Surrogate\nObjective", { variant: "accent" })}
    ${arrow(advX + 60, advY - 15, objX - 40, objY + 20, "advantage", {
      curve: { cx: 540, cy: 150 },
      labelDy: -6,
    })}
    ${arrow(objX, objY + 32, netX + 20, actorY - 32, "update actor", {
      color: "var(--diagram-update-arrow, #b91c1c)",
      curve: { cx: 400, cy: -10 },
      labelDy: -6,
    })}

    ${box(valLossX, valLossY, 170, 64, "Value Loss\n(V(s) vs return)", { variant: "accent" })}
    ${arrow(advX + 60, advY, valLossX - 85, valLossY)}
    ${arrow(valLossX, valLossY + 32, netX + 20, criticY + 32, "update critic", {
      color: "var(--diagram-update-arrow, #b91c1c)",
      curve: { cx: 400, cy: 340 },
      labelDy: 14,
    })}
  `;

  return svgRoot(width, height, "PPO actor-critic learning flow", inner);
}

// ---------------------------------------------------------------------------
// Mount helper
// ---------------------------------------------------------------------------

/**
 * Injects rendered SVG markup into a container element. Trivial wrapper kept
 * here so pages don't need to remember innerHTML vs insertAdjacentHTML.
 *
 * @param {Element} container - The DOM element to render into.
 * @param {string} svgMarkup - SVG markup string, e.g. from renderDQNDiagram().
 */
export function mountDiagram(container, svgMarkup) {
  container.innerHTML = svgMarkup;
}
