# RL Navigation Agent

A Reinforcement Learning agent that learns to navigate from a start cell to a goal 
cell in a dynamically generated grid environment, built entirely from scratch in 
Python — no ML frameworks, just NumPy and the Q-Learning algorithm.

The project includes a real-time tkinter GUI that visualizes the agent's learning 
process, live statistics, and a Q-value heatmap of the learned policy.

---

## Features

- **Grid environment** — configurable N×N grid (default 10×10) with randomized 
  obstacles, start, and goal positions
- **Q-Learning agent** — implemented from scratch using a 3D NumPy Q-table, 
  epsilon-greedy exploration, and the Bellman equation for updates
- **Reward shaping** — Manhattan-distance-based penalty guides the agent toward 
  goal-directed behavior faster than a flat step penalty
- **Real-time GUI** — live animation of agent movement, Q-value heatmap after 
  training, and adjustable sliders for grid size, episodes, obstacle density, and 
  animation speed
- **Thread-safe rendering** — all UI updates run through `root.after()` to avoid 
  race conditions during training

---

## Tech Stack

| Component | Library |
|---|---|
| Language | Python 3 |
| GUI | tkinter (standard library) |
| Numerical computation | NumPy |
| Randomness | random (standard library) |
| Concurrency | threading (standard library) |

No external ML frameworks (TensorFlow, PyTorch) are used — Q-Learning is implemented 
purely with NumPy arrays.

---

## How It Works

### Environment
- Grid is a 2D NumPy array; cells are free (0) or obstacle (1) at a configurable density (default 20%)
- Start (S) and Goal (G) are randomly assigned to distinct free cells
- Agent has 4 discrete actions: Up, Down, Left, Right

### Reward Structure

| Outcome | Reward |
|---|---|
| Goal reached | +10 |
| Hit an obstacle | −5 |
| Moved out of bounds | −2 |
| Normal valid move | −1 − (0.05 × Manhattan distance to goal) |

### Q-Learning Hyperparameters

| Parameter | Value | Role |
|---|---|---|
| Learning Rate (α) | 0.15 | Controls how much new info overrides old |
| Discount Factor (γ) | 0.99 | Weighs importance of future rewards |
| Initial Epsilon (ε) | 1.0 | Starting exploration rate |
| Epsilon Decay | 0.997 | Multiplier applied to ε after each episode |
| Minimum Epsilon (ε_min) | 0.01 | Floor — agent always retains 1% exploration |

Update rule (Bellman equation):
Q(s, a) ← Q(s, a) + α × [ r + γ × max Q(s', a') − Q(s, a) ]

---

## GUI Controls

| Slider | Range | Default |
|---|---|---|
| Grid Size | 5 – 100 | 10 |
| Episodes | 200 – 5000 | 1000 |
| Obstacles | 5% – 50% | 20% |
| Animation Speed | 0.02 – 0.8s | 0.5s |

**Visual legend:**
- 🔴 Red circle — agent's current position
- 🟢 Green — goal cell
- 🟠 Orange — start cell
- ⬛ Dark/black — obstacle
- Dark red trail — cells visited in current run
- Blue-to-warm gradient — Q-value heatmap (post-training)

---

## Requirements

```bash
pip install numpy
```
(tkinter ships with standard Python installations)

## How to Run

```bash
python main.py
```
Adjust sliders for grid size, episode count, and obstacle density, then start training. 
After training completes, run the agent to watch it navigate using the learned policy.

---

## Limitations

- Q-table is indexed by grid coordinates — no transfer learning between grid resets
- No validation that a path from Start to Goal actually exists (can be unreachable in dense grids)
- Loop-break mechanism during agent runs is heuristic, not guaranteed optimal
- Large grids (e.g. 100×100) scale the Q-table to 40,000 entries, slowing training significantly

## Possible Improvements

- BFS-based reachability check before training
- Deep Q-Network (DQN) implementation for generalization across grids
- Save/load Q-tables to resume training or reuse learned policies
- Live reward/episode graph during training
- Multiple agents or dynamic obstacles
- Enhance it to work similar to Google Maps