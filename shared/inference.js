/**
 * shared/inference.js
 *
 * Tiny dependency-free forward-pass engine for the RL Demos site. All three
 * demo pages (DQN/Snake, REINFORCE/CartPole, PPO/Pendulum) import this module
 * to load a checkpoint JSON exported from offline Python training and run its
 * forward pass live in the browser as the user scrubs the training-progress
 * slider. No TF.js / ONNX.js — these networks are small feedforward MLPs, so
 * the matrix math is done by hand with plain arrays.
 */

/**
 * Fetches a checkpoint JSON file from the given path and returns it parsed,
 * unmodified. The returned object is expected to have the shape
 * { algorithm, env, checkpoint_pct, architecture, layers } as produced by the
 * training scripts. Throws if the network request fails or the response body
 * is not valid JSON.
 *
 * @param {string} path - URL or relative path to the checkpoint JSON file.
 * @returns {Promise<object>} The parsed checkpoint object.
 */
export async function loadCheckpoint(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(
      `loadCheckpoint: failed to fetch "${path}" (HTTP ${response.status} ${response.statusText})`
    );
  }
  const checkpoint = await response.json();
  return checkpoint;
}

/**
 * Applies the ReLU activation elementwise: max(0, x) for every entry.
 *
 * @param {number[]} vector - Input values.
 * @returns {number[]} A new array with ReLU applied.
 */
function applyRelu(vector) {
  return vector.map((value) => Math.max(0, value));
}

/**
 * Applies the softmax activation to a vector, turning raw scores into a
 * probability distribution that sums to 1. Subtracts the max value first for
 * numerical stability (standard "safe softmax" trick).
 *
 * @param {number[]} vector - Raw output scores (logits).
 * @returns {number[]} A new array of probabilities summing to 1.
 */
function applySoftmax(vector) {
  const maxValue = Math.max(...vector);
  const expValues = vector.map((value) => Math.exp(value - maxValue));
  const sumExpValues = expValues.reduce((sum, value) => sum + value, 0);
  return expValues.map((value) => value / sumExpValues);
}

/**
 * Applies the tanh activation elementwise, squashing every entry to (-1, 1).
 *
 * @param {number[]} vector - Input values.
 * @returns {number[]} A new array with tanh applied.
 */
function applyTanh(vector) {
  return vector.map((value) => Math.tanh(value));
}

/**
 * Runs one fully-connected layer: output[o] = sum_i(w[o][i] * input[i]) + b[o]
 * for every output index o. This matches the checkpoint's weight layout,
 * where each layer's "w" is shaped [output_dim, input_dim].
 *
 * @param {number[][]} weights - Weight matrix, shape [output_dim][input_dim].
 * @param {number[]} biases - Bias vector, length output_dim.
 * @param {number[]} input - Input vector, length input_dim.
 * @returns {number[]} The pre-activation output vector, length output_dim.
 */
function runLinearLayer(weights, biases, input) {
  const outputDim = weights.length;
  const output = new Array(outputDim);
  for (let outputIndex = 0; outputIndex < outputDim; outputIndex++) {
    const weightRow = weights[outputIndex];
    let sum = biases[outputIndex];
    for (let inputIndex = 0; inputIndex < weightRow.length; inputIndex++) {
      sum += weightRow[inputIndex] * input[inputIndex];
    }
    output[outputIndex] = sum;
  }
  return output;
}

/**
 * Runs the forward pass of a checkpoint's neural network on a single input
 * vector. Every hidden layer applies ReLU; the final layer applies
 * checkpoint.architecture.output_activation ("linear", "softmax", or "tanh").
 * If an action_scale value is present (checked on architecture.action_scale
 * first, then the top-level checkpoint.action_scale), and the output
 * activation is "tanh", the final output is multiplied elementwise by it —
 * this rescales a tanh-bounded (-1, 1) action to the environment's actual
 * action range (e.g. PPO/Pendulum torque).
 *
 * @param {object} checkpoint - A parsed checkpoint object from loadCheckpoint().
 * @param {number[]} inputArray - The observation vector to feed into the network.
 * @returns {number[]} The network's output vector (e.g. Q-values, action
 *   probabilities, or a scaled continuous action).
 */
export function predict(checkpoint, inputArray) {
  const architecture = checkpoint.architecture;

  if (inputArray.length !== architecture.input_dim) {
    throw new Error(
      `predict: inputArray has length ${inputArray.length}, but checkpoint.architecture.input_dim is ${architecture.input_dim}`
    );
  }

  const layers = checkpoint.layers;
  let activation = inputArray;

  for (let layerIndex = 0; layerIndex < layers.length; layerIndex++) {
    const layer = layers[layerIndex];
    const isLastLayer = layerIndex === layers.length - 1;

    const preActivation = runLinearLayer(layer.w, layer.b, activation);

    if (!isLastLayer) {
      // Every hidden layer uses ReLU.
      activation = applyRelu(preActivation);
      continue;
    }

    // Final layer: apply the checkpoint's configured output activation.
    const outputActivation = architecture.output_activation;
    if (outputActivation === "relu") {
      activation = applyRelu(preActivation);
    } else if (outputActivation === "softmax") {
      activation = applySoftmax(preActivation);
    } else if (outputActivation === "tanh") {
      activation = applyTanh(preActivation);

      // action_scale may live on architecture or on the checkpoint root;
      // prefer architecture.action_scale when both are present.
      const actionScale =
        architecture.action_scale !== undefined
          ? architecture.action_scale
          : checkpoint.action_scale;

      if (actionScale !== undefined) {
        activation = activation.map((value) => value * actionScale);
      }
    } else if (outputActivation === "linear") {
      activation = preActivation;
    } else {
      throw new Error(
        `predict: unrecognized output_activation "${outputActivation}"`
      );
    }
  }

  return activation;
}
