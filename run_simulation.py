# run_simulation.py
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from grid_env import GridEnv
from Agent import QLearningAgent

# ---------- PARAMETERS ----------
SIZE = 5
START = (0,0)
GOAL = (4,4)
# exactly two obstacles provided here
OBSTACLES = [(2,2), (3,1)]

PRETRAIN_EPISODES = 800      # pretrain to speed convergence
ANIM_EPISODES = 6           # number of episodes to animate
MAX_STEPS_PER_EP = 100
FRAME_INTERVAL_MS = 350     # animation frame interval
SHOW_VALUE_NUMBERS = True   # show numeric Qmax in each cell

# ---------- CREATE ENV + AGENT ----------
env = GridEnv(size=SIZE, start=START, goal=GOAL, obstacles=OBSTACLES,
              step_reward=-0.2, goal_reward=50.0, obstacle_reward=-20.0)

agent = QLearningAgent(env,
                       lr=0.15,
                       gamma=0.95,
                       epsilon=1.0,
                       epsilon_min=0.05,
                       epsilon_decay=0.998)  # slower decay -> more exploration

print("Env info:", env.get_info())
print("Pre-training agent for", PRETRAIN_EPISODES, "episodes...")
stats = agent.train(episodes=PRETRAIN_EPISODES, max_steps=MAX_STEPS_PER_EP, verbose=False)
print("Pretrain done. success_rate =", stats["success_rate"])

# ---------- MATPLOTLIB ANIMATION SETUP ----------
fig, ax = plt.subplots(figsize=(6,6))
plt.tight_layout()

# State holder for animation
anim_data = {
    "episode": 0,
    "step": 0,
    "state": env.reset(),
    "done": False,
    "show_goal_banner": 0  # countdown frames to display "GOAL!"
}

def draw_frame(ax, Q):
    ax.clear()
    size = env.size
    Qmax = env.qmax_to_grid(Q)

    # show heatmap (Qmax)
    im = ax.imshow(Qmax.T, origin='lower', cmap='viridis')

    # numeric values
    if SHOW_VALUE_NUMBERS:
        for r in range(size):
            for c in range(size):
                val = Qmax[r, c]
                # choose color based on background brightness for good contrast
                color = "white" if val < (Qmax.max() * 0.6) else "black"
                ax.text(r, c, f"{val:.1f}", ha='center', va='center', color=color, fontsize=10)

    # obstacles
    for (ro, co) in env.obstacles:
        # draw black square centered in cell
        ax.add_patch(plt.Rectangle((ro-0.5, co-0.5), 1, 1, color='black'))

    # goal: gold star marker
    gr, gc = env.goal
    ax.scatter([gr], [gc], marker='*', s=400, color='gold', edgecolors='k', zorder=3)

    # agent
    ar, ac = anim_data["state"]
    ax.scatter([ar], [ac], marker='o', s=300, color='cyan', edgecolors='k', linewidths=2, zorder=4)

    # if show_goal_banner > 0 -> display text
    if anim_data["show_goal_banner"] > 0:
        ax.text(0.5, 1.05, "GOAL REACHED!", transform=ax.transAxes,
                ha='center', va='bottom', fontsize=22, color='green', weight='bold')
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xlim(-0.5, size-0.5)
    ax.set_ylim(-0.5, size-0.5)
    ax.set_title(f"Episode {anim_data['episode']+1}  Step {anim_data['step']}", fontsize=16)

    return im

# Build sequence of states by running episodes (we step one frame per agent action)
# We'll generate a list of frames (state snapshots) to animate. Each frame is a (state, Q_table, banner_flag)
frames = []

# Build frames by running ANIM_EPISODES episodes using the current agent.q_table
for ep in range(ANIM_EPISODES):
    s = env.reset()
    anim_data["episode"] = ep
    anim_data["state"] = s
    anim_data["step"] = 0

    done = False
    step_count = 0
    while (not done) and (step_count < MAX_STEPS_PER_EP):
        # choose action greedily (agent already trained mostly)
        a = agent.choose_action(s, greedy=False)
        next_state, reward, done = env.step(a)
        # we DO NOT call agent.update here (we pre-trained). If you want learning on the fly, call agent.update(...)
        # But for deterministic visualization it's clearer without on-the-fly learning
        # Save frame snapshot
        frames.append({
            "state": next_state,
            "Q": agent.q_table.copy(),
            "banner": True if reward == env.goal_reward else False
        })
        s = next_state
        step_count += 1

    # if ended immediately due to goal/obstacle, still keep a small pause frames
    if len(frames) > 0 and frames[-1]["banner"]:
        # append a few identical frames so the banner is visible
        for _ in range(3):
            frames.append({
                "state": frames[-1]["state"],
                "Q": agent.q_table.copy(),
                "banner": True
            })

print(f"Prepared {len(frames)} frames for animation (from {ANIM_EPISODES} episodes).")

# Animation update function uses precomputed frames
def update(frame_index):
    frame = frames[frame_index]
    anim_data["state"] = frame["state"]
    anim_data["show_goal_banner"] = 3 if frame["banner"] else 0
    anim_data["step"] = frame_index  # global step index
    return draw_frame(ax, frame["Q"])

ani = animation.FuncAnimation(fig, update, frames=len(frames),
                              interval=FRAME_INTERVAL_MS, blit=False, repeat=False)

plt.show()
