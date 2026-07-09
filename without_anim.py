import tkinter as tk
import time
import threading
import numpy as np
import random

class GridEnvironment:
    def __init__(self, size=10, obstacle_density=0.2):
        self.size = size
        self.obstacle_density = obstacle_density
        self.grid = None
        self.start = None
        self.goal = None
        self.agent_pos = None
        self.generate_new_grid()

    def generate_new_grid(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        #0 = free, 1 = obstacle

        #placing obstacles randomnly
        for r in range(self.size):
            for c in range(self.size):
                if random.random() < self.obstacle_density:
                    self.grid[r][c] = 1

        self.start = self._random_free_cell()
        self.goal = self._random_free_cell()
        while self.goal == self.start:
            self.goal = self._random_free_cell()

        self.grid[self.start] = 0
        self.grid[self.goal] = 0
        self.agent_pos = self.start

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def _random_free_cell(self):
        while True:
            r = random.randint(0,self.size - 1)
            c = random.randint(0,self.size - 1)
            if self.grid[r][c] == 0:
                return (r,c)
    
    def step(self, action):
        #Actions: 0=up, 1=down, 2=left, 3=right
        #returns: next_state, reward, done
        r,c = self.agent_pos
        dr, dc = [(-1,0),(1,0),(0,-1),(0,1)][action]
        nr, nc = r + dr, c + dc
        
        #out of bounds toh penalizes the agent
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return self.agent_pos, -2, False
        
        #hit obstacle toh penalizes the agent
        if self.grid[nr][nc] == 1:
            return self.agent_pos, -5, False
        
        self.agent_pos = (nr,nc)

        #Reached Goal State
        if self.agent_pos == self.goal:
            return self.agent_pos, +10, True
        
        dist = abs(nr - self.goal[0]) + abs(nc - self.goal[1])
        return self.agent_pos, -1 - 0.05 * dist, False
    

class QlearningAgent:
    def __init__(self, grid_size, learning_rate=0.15, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.997, epsilon_min=0.01):
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((grid_size, grid_size, 4))

    def choose_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, 3)
        r,c = state
        return int(np.argmax(self.q_table[r][c]))
    
    def update(self, state, action, reward, next_state, done):
        r,c = state
        nr,nc = next_state
        current = self.q_table[r][c][action]
        target = reward if done else reward + self.discount_factor * np.max(self.q_table[nr][nc])
        self.q_table[r][c][action] += self.learning_rate * (target - current)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class GUI:
    BG = "#180202"
    PANEL_BG = "#a81010"
    ACCENT = "#ff3535"
    CELL_FREE = "#612d2d"
    CELL_OBS = "#111111"
    CELL_TRAIL = "#6e1a1a"
    AGENT_COLOR = "#ff0000"
    GOAL_COLOR = "#26ff00"
    START_COLOR = "#ff6b35"
    TEXT_COLOR = "#e0e0e0"
    BTN_TRAIN = "#e94560"
    BTN_RUN = "#ffd000"
    BTN_RESET = "#ff6b35"

    def __init__(self, root):
        self.root = root
        self.root.title("RL Navigation Agent")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        self.grid_size = tk.IntVar(value=10)
        self.episodes_var = tk.IntVar(value=1000)
        self.speed_var = tk.DoubleVar(value=0.5)
        self.density_var = tk.DoubleVar(value=0.2)

        self.is_training = False
        self.is_running = False
        self.env = None
        self.agent = None
        self.trail = set()
        self.loop_breaks = 0
        self.episode_count = 0
        self.success_count = 0

        self._build_ui()
        self._init_env()

    def _build_ui(self):
        tf = tk.Frame(self.root, bg=self.BG)
        tf.pack(fill="x", padx=16, pady=(14,4))
        tk.Label(tf, text="RL Navigation Agent", font=("Segoe UI", 18, "bold"), bg=self.BG, fg=self.AGENT_COLOR).pack(side="left")
        
        main = tk.Frame(self.root, bg=self.BG)
        main.pack(padx=16, pady = 6)

        left = tk.Frame(main, bg=self.PANEL_BG, highlightbackground="#600f0f", highlightthickness=1)
        left.pack(side="left", padx=(0,12), fill="y")

        self._section(left, "SETTINGS")
        self._slider(left, "Grid Size", self.grid_size, 5, 100, integer=True)
        self._slider(left, "Episodes", self.episodes_var, 200, 5000, integer=True)
        self._slider(left, "Obstacles", self.density_var, 0.05, 0.50, fmt=".0%")
        self._slider(left, "Anim Speed", self.speed_var, 0.02, 0.8, fmt=".2f")

        self._section(left, "CONTROLS")
        self._btn(left, "Perceive Environment", self.BTN_TRAIN, self._start_training)
        self._btn(left, "RUN AGENT", self.BTN_RUN, self._run_agent)
        self._btn(left, "RESET", self.BTN_RESET, self._reset)

        self._section(left, "LIVE STATS")
        self.stat_ep = self._stat_row(left, "Episode")
        self.stat_sr = self._stat_row(left, "Success Rate")
        self.stat_eps = self._stat_row(left, "Epsilon")
        self.stat_step = self._stat_row(left, "Step(run)")
        self.stat_rew = self._stat_row(left, "Reward(run)")
        self.stat_loops = self._stat_row(left, "Loop Breaks")

        self._section(left, "KEY")
        for color, label in [
            (self.AGENT_COLOR, "Agent"),
            (self.GOAL_COLOR, "Goal(G)"),
            (self.START_COLOR, "Start(S)"),
            (self.CELL_OBS, "Obstacle"),
            (self.CELL_TRAIL, "Trail"),
        ]:
            row = tk.Frame(left, bg=self.PANEL_BG)
            row.pack(anchor="w", padx=14, pady=2)
            tk.Label(row, bg=color, width=2, height=1).pack(side="left")
            tk.Label(row, text=f" {label}", bg=self.PANEL_BG, fg=self.TEXT_COLOR, font=("Segoe UI", 9)).pack(side="left")

        right = tk.Frame(main, bg=self.BG)
        right.pack(side="left")
        self.canvas = tk.Canvas(right, bg=self.BG, highlightbackground= "#600f0f", highlightthickness=2)
        self.canvas.pack()

        self.status_var = tk.StringVar(value="Ready - press Perceive Environment to start.")
        tk.Label(self.root, textvariable=self.status_var,bg=self.ACCENT, fg=self.TEXT_COLOR, font=("Segoe UI", 9), anchor="w", padx=10).pack(fill="x", side="bottom")

    def _section(self, p, text):
        tk.Label(p, text=text, font=("Segoe UI", 9, "bold"), bg=self.PANEL_BG, fg="#aaa").pack(anchor="w", padx=14, pady=(14,2))
        tk.Frame(p, bg="#600f25", height=1).pack(fill="x", padx=14, pady=(0,6))
    
    def _btn(self, p, text, color, cmd):
        tk.Button(p, text=text, command=cmd, bg=color, fg="white", font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2", activebackground=color, activeforeground="white", padx=10, pady=6).pack(fill="x", padx=14, pady=3)
    
    def _slider(self, p, label, var, lo, hi, integer=False, fmt=".1f"):
        tk.Label(p, text=label, bg=self.PANEL_BG, fg=self.TEXT_COLOR, font=("Segoe UI", 9)).pack(anchor="w", padx=14)
        row = tk.Frame(p, bg=self.PANEL_BG)
        row.pack(fill="x", padx=14, pady=(0,4))
        lbl = tk.Label(row, bg=self.PANEL_BG, fg=self.AGENT_COLOR, font=("Segoe UI", 9, "bold"), width=6)
        lbl.pack(side="right")

        def refresh(_=None):
            v = var.get()
            if fmt == ".0%": lbl.config(text=f"{v:.0%}")
            elif integer: lbl.config(text=str(int(v)))
            else: lbl.config(text=f"{v:{fmt}}")
        
        tk.Scale(row, variable=var, from_=lo, to=hi, orient="horizontal", showvalue=False, bg=self.PANEL_BG, troughcolor=self.ACCENT, highlightthickness=0, resolution=1 if integer else 0.01, command=refresh).pack(side="left", fill="x", expand=True)
        refresh()
    
    def _stat_row(self, p, label):
        row = tk.Frame(p, bg=self.PANEL_BG)
        row.pack(fill="x", padx=14, pady=1)
        tk.Label(row, text=label+":", bg=self.PANEL_BG, fg="#888", font=("Segoe UI", 9), width=13, anchor="w").pack(side="left")
        val = tk.Label(row, text="-", bg=self.PANEL_BG, fg=self.AGENT_COLOR, font=("Segoe UI", 9, "bold"))
        val.pack(side="left")
        return val
    
    def _cell_px(self):
        return max(28, min(56, 500 // self.grid_size.get()))
    
    def _init_env(self):
        size = self.grid_size.get()
        px = self._cell_px()
        self.canvas.config(width=size*px, height=size*px)
        self.env = GridEnvironment(size=size, obstacle_density=self.density_var.get())
        self.agent = QlearningAgent(grid_size=size)
        self.trail = set()
        self.loop_breaks = 0
        self.episode_count = 0
        self.success_count = 0
        self._update_stats(0, 0)
        self._draw_grid()

    def _draw_grid(self):
        self.canvas.delete("all")
        px = self._cell_px()

        for r in range(self.env.size):
            for c in range(self.env.size):
                x0, y0 = c*px, r*px
                if self.env.grid[r][c] == 1:
                    fill = self.CELL_OBS
                elif (r, c) in self.trail:
                    fill = self.CELL_TRAIL
                else:
                    fill = self.CELL_FREE
                self.canvas.create_rectangle(x0, y0, x0+px, y0+px, fill=fill, outline= "#0d0d1a", width=1)

        sr, sc = self.env.start
        self.canvas.create_rectangle(sc*px, sr*px, sc*px+px, sr*px+px, fill=self.START_COLOR, outline= "#0d0d1a")
        self.canvas.create_text(sc*px+px//2, sr*px+px//2, text="S", fill="white", font=("Segoe UI", max(8, px//3), "bold"))

        gr, gc = self.env.goal
        self.canvas.create_rectangle(gc*px, gr*px, gc*px+px, gr*px+px,
                             fill=self.GOAL_COLOR, outline="#0d0d1a")
        self.canvas.create_text(gc*px+px//2, gr*px+px//2,
                        text="G", fill="white",
                        font=("Segoe UI", max(8, px//3), "bold"))

        ar, ac = self.env.agent_pos
        pad = px * 0.18
        self.canvas.create_oval(
            ac*px+pad, ar*px+pad, ac*px+px-pad, ar*px+px-pad,
            fill=self.AGENT_COLOR, outline="white", width=2)
        
    def _update_stats(self, steps, reward):
        ep = self.episode_count
        self.stat_ep.config(text=str(ep))
        self.stat_sr.config(
            text=f"{self.success_count/ep*100:.1f}%" if ep > 0 else "-")
        self.stat_eps.config(
            text=f"{self.agent.epsilon:.3f}" if self.agent else "-")
        self.stat_step.config(text=str(steps))
        self.stat_rew.config(text=f"{reward:.1f}" if steps > 0 else "-")
        self.stat_loops.config(text=str(self.loop_breaks))
    
    def _start_training(self):
        if self.is_training or self.is_running:
            return
        size = self.grid_size.get()
        if (self.env is None or self.env.size != size or abs(self.env.obstacle_density - self.density_var.get()) > 0.001):
            self._init_env()
        else:
            self.agent = QlearningAgent(grid_size=size)
            self.episode_count = 0
            self.success_count = 0
            self.trail = set()
            self._update_stats(0, 0)

        self.is_training = True
        self.status_var.set("Perceiving...")
        threading.Thread(target=self._train_thread, daemon=True).start()

    def _train_thread(self):
        episodes = self.episodes_var.get()
        max_steps = self.env.size * self.env.size * 4
        update_every = max(1, episodes // 100)

        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            for step in range(max_steps):
                action = self.agent.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # if ep > episodes*0.3:
                #     self.root.after(0, self._draw_grid)
                #     time.sleep(0.55)

                if done:
                    break

            self.agent.decay_epsilon()
            self.episode_count += 1
            if done:
                self.success_count += 1
            
            if (ep + 1) % update_every == 0:
                pct = int((ep + 1) / episodes * 100)
                sr = self.success_count / (ep + 1) * 100
                self.root.after(0, self._update_stats, step+1, total_reward)
                self.root.after(0, self.status_var.set, f"Perceiving {pct}% | Episode {ep+1}/{episodes} | " f"Success: {sr:.1f}% | Epsilon: {self.agent.epsilon:.3f}")

        self.is_training = False
        final_sr = self.success_count / self.episode_count * 100
        self.root.after(0, self.status_var.set, f"Environment perceiving complete! Final success rate: {final_sr:.1f}% " f"- Now press RUN AGENT to watch the agent navigate.")
        self.root.after(0, self._draw_grid)

    def _run_agent(self):
        if self.is_training or self.is_running:
            return
        if self.agent is None or np.all(self.agent.q_table == 0):
            self.status_var.set("Please perceive the environment first!")
            return
        self.is_running = True
        self.loop_breaks = 0
        self.trail = set()
        self.env.reset()
        self._draw_grid()
        self.status_var.set("Agent is navigating...")
        threading.Thread(target=self._run_thread, daemon=True).start()

    def _run_thread(self):
        state = self.env.agent_pos
        done = False
        steps = 0
        total_r = 0
        visited = {}
        max_run = self.env.size * self.env.size * 4

        while not done and steps < max_run:
            self.trail.add(self.env.agent_pos)

            vc = visited.get(state, 0)
            if vc >= 2:
                r, c = state
                sorted_actions = np.argsort(self.agent.q_table[r][c])[::-1]
                action = int(sorted_actions[min(vc - 1, 3)])
                self.loop_breaks += 1
            else:
                action = self.agent.choose_action(state, greedy=True)

            visited[state] = vc + 1

            state, reward, done = self.env.step(action)
            total_r += reward
            steps += 1

            self.root.after(0, self._draw_grid)
            self.root.after(0, self._update_stats, steps, total_r)
            time.sleep(self.speed_var.get())

        if done:
            self.root.after(0, self.status_var.set, f"Goal Reached in {steps} steps! " f"Total reward: {total_r:.1f} | Loop breaks used: {self.loop_breaks}")
        else:
            self.root.after(0, self.status_var.set, f"Could not reach goal after {steps} steps. " f"Try perceiving the environment again, or reset for a simpler grid.")
        self.is_running = False
        
    def _reset(self):
        if self.is_training or self.is_running:
            return
        self._init_env()
        self.status_var.set("New grid generated - ready to perceive.")


if __name__ == "__main__":
    root = tk.Tk()
    GUI(root)
    root.mainloop()