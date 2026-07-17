/**
 * snake/game.js
 *
 * Pure Snake environment logic (state encoding, stepping, collision) used by
 * snake/index.html. Kept dependency-free and framework-free so it can be
 * unit-sanity-checked with plain `node --check` / manual tracing, and so the
 * encoding here can be compared line-by-line against the Python training env
 * it must match.
 *
 * Grid coordinate system: x grows right, y grows DOWN (canvas convention).
 * Directions are unit vectors {dx, dy}:
 *   UP    = { dx:  0, dy: -1 }
 *   DOWN  = { dx:  0, dy:  1 }
 *   LEFT  = { dx: -1, dy:  0 }
 *   RIGHT = { dx:  1, dy:  0 }
 */

export const GRID_SIZE = 20;
export const MAX_STEPS_WITHOUT_FOOD = 500;

export const DIRS = {
  UP: { dx: 0, dy: -1 },
  DOWN: { dx: 0, dy: 1 },
  LEFT: { dx: -1, dy: 0 },
  RIGHT: { dx: 1, dy: 0 },
};

// Action index -> direction. Order fixed: up, down, left, right.
export const ACTIONS = [DIRS.UP, DIRS.DOWN, DIRS.LEFT, DIRS.RIGHT];

/**
 * Rotates a direction vector 90 degrees clockwise on screen (y-down coords).
 * up -> right -> down -> left -> up.
 */
function turnRight(dir) {
  return { dx: -dir.dy, dy: dir.dx };
}

/**
 * Rotates a direction vector 90 degrees counter-clockwise on screen.
 * up -> left -> down -> right -> up.
 */
function turnLeft(dir) {
  return { dx: dir.dy, dy: -dir.dx };
}

function sameDir(a, b) {
  return a.dx === b.dx && a.dy === b.dy;
}

/**
 * Creates a fresh episode state: a length-3 snake centered on the board
 * moving right, and one food cell placed on a random free tile.
 *
 * @param {() => number} rand - Random source in [0, 1); injectable for tests.
 */
export function createInitialState(rand = Math.random) {
  const cx = Math.floor(GRID_SIZE / 2);
  const cy = Math.floor(GRID_SIZE / 2);
  // Head first; body trails to the left since we start moving right.
  const snake = [
    { x: cx, y: cy },
    { x: cx - 1, y: cy },
    { x: cx - 2, y: cy },
  ];
  const state = {
    snake,
    dir: DIRS.RIGHT,
    food: null,
    stepsSinceFood: 0,
    stepCount: 0,
    score: 0,
    alive: true,
  };
  state.food = placeFood(state, rand);
  return state;
}

function isOnSnake(snake, x, y) {
  return snake.some((segment) => segment.x === x && segment.y === y);
}

/**
 * Places food on a uniformly random free (non-snake) cell.
 */
export function placeFood(state, rand = Math.random) {
  const freeCells = [];
  for (let y = 0; y < GRID_SIZE; y++) {
    for (let x = 0; x < GRID_SIZE; x++) {
      if (!isOnSnake(state.snake, x, y)) {
        freeCells.push({ x, y });
      }
    }
  }
  if (freeCells.length === 0) {
    return null; // board full (win condition, effectively unreachable at 20x20/len3)
  }
  const idx = Math.floor(rand() * freeCells.length);
  return freeCells[idx];
}

/**
 * True if the given absolute cell is a wall or occupied by the snake's body
 * (any segment, including the tail — matches "hits ... the snake's own body
 * within 1 cell" from a lookahead of exactly one step).
 */
function isDanger(state, x, y) {
  if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE) {
    return true;
  }
  return isOnSnake(state.snake, x, y);
}

/**
 * Builds the 11-float observation vector for the current state, in the
 * exact order the Python training env used:
 *   [danger_straight, danger_right, danger_left,
 *    moving_left, moving_right, moving_up, moving_down,
 *    food_left, food_right, food_up, food_down]
 */
export function encodeState(state) {
  const head = state.snake[0];
  const dir = state.dir;
  const rightDir = turnRight(dir);
  const leftDir = turnLeft(dir);

  const dangerStraight = isDanger(state, head.x + dir.dx, head.y + dir.dy) ? 1 : 0;
  const dangerRight = isDanger(state, head.x + rightDir.dx, head.y + rightDir.dy) ? 1 : 0;
  const dangerLeft = isDanger(state, head.x + leftDir.dx, head.y + leftDir.dy) ? 1 : 0;

  const movingLeft = sameDir(dir, DIRS.LEFT) ? 1 : 0;
  const movingRight = sameDir(dir, DIRS.RIGHT) ? 1 : 0;
  const movingUp = sameDir(dir, DIRS.UP) ? 1 : 0;
  const movingDown = sameDir(dir, DIRS.DOWN) ? 1 : 0;

  const food = state.food;
  const foodLeft = food && food.x < head.x ? 1 : 0;
  const foodRight = food && food.x > head.x ? 1 : 0;
  const foodUp = food && food.y < head.y ? 1 : 0;
  const foodDown = food && food.y > head.y ? 1 : 0;

  return [
    dangerStraight,
    dangerRight,
    dangerLeft,
    movingLeft,
    movingRight,
    movingUp,
    movingDown,
    foodLeft,
    foodRight,
    foodUp,
    foodDown,
  ];
}

/**
 * Advances the environment by one step given a chosen action index
 * (0=up, 1=down, 2=left, 3=right — see ACTIONS). Mutates and returns state.
 * No special-casing a reversal into the snake's own neck: choosing the
 * opposite of the current heading is legal and simply kills the snake via
 * normal self-collision, matching the training env.
 */
export function stepState(state, actionIndex, rand = Math.random) {
  if (!state.alive) {
    return state;
  }

  const newDir = ACTIONS[actionIndex];
  state.dir = newDir;

  const head = state.snake[0];
  const newHead = { x: head.x + newDir.dx, y: head.y + newDir.dy };

  state.stepCount += 1;

  // Wall collision.
  if (newHead.x < 0 || newHead.x >= GRID_SIZE || newHead.y < 0 || newHead.y >= GRID_SIZE) {
    state.alive = false;
    return state;
  }

  const ateFood = state.food && newHead.x === state.food.x && newHead.y === state.food.y;

  // Self collision: the tail cell vacates this step unless we just ate, so
  // it's not an obstacle when not growing.
  const bodyToCheck = ateFood ? state.snake : state.snake.slice(0, -1);
  if (isOnSnake(bodyToCheck, newHead.x, newHead.y)) {
    state.alive = false;
    return state;
  }

  state.snake.unshift(newHead);
  if (ateFood) {
    state.score += 1;
    state.stepsSinceFood = 0;
    state.food = placeFood(state, rand);
  } else {
    state.snake.pop();
    state.stepsSinceFood += 1;
  }

  if (state.stepsSinceFood >= MAX_STEPS_WITHOUT_FOOD) {
    state.alive = false;
  }

  return state;
}

/**
 * Picks the greedy (argmax) action index from a Q-value array.
 */
export function argmaxAction(qValues) {
  let bestIndex = 0;
  let bestValue = qValues[0];
  for (let i = 1; i < qValues.length; i++) {
    if (qValues[i] > bestValue) {
      bestValue = qValues[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}
