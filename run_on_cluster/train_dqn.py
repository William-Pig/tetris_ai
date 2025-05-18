#!/usr/bin/env python
import os
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

from TetrisGym_0517 import TetrisGym
from batch_dqn import DQNAgent

# ——————————————————————————————
# 1) Simulation & hyper‐parameters
# ——————————————————————————————
NUM_EPISODES        = 100_000 #_000
MAX_STEPS_PER_EPISODE = 5_000
ALPHA               = 0.001
GAMMA               = 0.5
EPSILON_MIN         = 0.01
EPSILON_DECAY       = 0.9999
BATCH_SIZE          = 128
TARGET_UPDATE_FREQ  = 64

BOARD_WIDTH, BOARD_HEIGHT = 6, 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ── at the top, after defining SAVE_DIR ───────────────────────────────
GIF_CHECKPOINTS = {1000, } # 10_000, 100_000,1_000_000
# you can adjust these to whatever episode numbers you want GIFs for

def evaluate_and_save_gif(ep):
    eval_env = TetrisGym(
        width=BOARD_WIDTH,
        height=BOARD_HEIGHT,
        state_mode="tensor",
        render_mode="capture"
    )
    old_eps = agent.epsilon
    agent.epsilon = 0.0   # force greedy rollout

    state, done = eval_env.reset(), False
    while not done:
        valid_ids = eval_env.get_valid_action_ids()
        if not valid_ids:
            break
        action = agent.act(state, valid_ids)
        state, _, done, _ = eval_env.step(action)

    agent.epsilon = old_eps  # restore exploration

    gif_path = os.path.join(SAVE_DIR, f"ep_{ep}.gif")
    eval_env.save_gif(gif_path)
    print(f"Saved GIF: {gif_path}")

# ——————————————————————————————
# 2) Saving & resume setup
# ——————————————————————————————
SAVE_DIR         = f"batch_dqn_resume_results_NUM_EPISODES_{NUM_EPISODES}"
CHECKPOINT_PATH  = os.path.join(SAVE_DIR, "latest.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

# Determine if we’re resuming
if os.path.exists(CHECKPOINT_PATH):
    print("Resuming from checkpoint:", CHECKPOINT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    start_episode       = checkpoint["episode"] + 1
    episode_rewards     = checkpoint["episode_rewards"]
    episode_score       = checkpoint["episode_score"]
    epsilon_start       = checkpoint["epsilon"]
else:
    print("Starting fresh training")
    start_episode       = 1
    episode_rewards     = []
    episode_score       = []
    epsilon_start       = 1.0

# ——————————————————————————————
# 3) Environment & agent
# ——————————————————————————————
env = TetrisGym(
    width=BOARD_WIDTH,
    height=BOARD_HEIGHT,
    state_mode="tensor",
    max_steps=MAX_STEPS_PER_EPISODE
)
state_shape = env.get_state().shape
num_actions = env.get_action_space_size()

agent = DQNAgent(
    state_shape=state_shape,
    num_actions=num_actions,
    device=DEVICE,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=epsilon_start,
    epsilon_min=EPSILON_MIN,
    epsilon_decay=EPSILON_DECAY
)

if os.path.exists(CHECKPOINT_PATH):
    agent.load_from_dict(checkpoint)

# ——————————————————————————————
# 4) GIF saver (greedy)
# ——————————————————————————————
def evaluate_and_save_gif(ep):
    eval_env = TetrisGym(
        width=BOARD_WIDTH,
        height=BOARD_HEIGHT,
        state_mode="tensor",
        render_mode="capture"
    )
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    state, done = eval_env.reset(), False
    while not done:
        ids = eval_env.get_valid_action_ids()
        if not ids: break
        a = agent.act(state, ids)
        state, _, done, _ = eval_env.step(a)
    agent.epsilon = old_eps
    gif_path = os.path.join(SAVE_DIR, f"ep_{ep}.gif")
    eval_env.save_gif(gif_path)

# ——————————————————————————————
# 5) Main training loop
# ——————————————————————————————
progress = tqdm(
    range(start_episode, start_episode + NUM_EPISODES),
    desc="Training DQNResume"
)
for ep in progress:
    state, total_reward, done = env.reset(), 0.0, False

    while not done:
        valid = env.get_valid_action_ids()
        a = agent.act(state, valid)
        next_s, r, done, _ = env.step(a)
        agent.memorize(state, a, r, next_s, done)
        _ = agent.replay(BATCH_SIZE)
        state = next_s
        total_reward += r

    episode_rewards.append(total_reward)
    episode_score.append(env.game.score)

    # 1) sync target network
    if ep % TARGET_UPDATE_FREQ == 0:
        agent.target_model.load_state_dict(agent.model.state_dict())

    # 2) save GIF if desired
    # if ep in { ... }: evaluate_and_save_gif(ep)
    if ep in GIF_CHECKPOINTS:
        evaluate_and_save_gif(ep)


    # 3) checkpoint every 1k episodes
    if ep % 1_000 == 0:
        extra = {
            "episode": ep,
            "episode_rewards": episode_rewards,
            "episode_score": episode_score,
            "epsilon": agent.epsilon
        }
        agent.save(os.path.join(SAVE_DIR, "latest.pth"), extra)

    # 4) update progress bar
    if ep % 1000 == 0:
        mean100 = np.mean(episode_rewards[-100:])
        progress.set_description(
            f"Ep {ep} | mean100 {mean100:.2f} | eps {agent.epsilon:.3f}"
        )

# ——————————————————————————————
# 6) Save metrics & final plot
# ——————————————————————————————
with open(os.path.join(SAVE_DIR, "episode_data.pkl"), "wb") as f:
    pickle.dump({
        "rewards": episode_rewards,
        "scores": episode_score
    }, f)

# learning curve
batches = 100
avg_r = np.mean(np.array_split(np.array(episode_rewards), batches), axis=1)
plt.figure()
plt.plot(avg_r)
plt.xlabel("Episode batch")
plt.ylabel("Avg Reward")
plt.title("DQN Resume Training Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, f"learning_curve_NUM_EPISODES{NUM_EPISODES}.png"))

# final model
agent.save(os.path.join(SAVE_DIR, "model_final.pth"), {
    "episode": start_episode+NUM_EPISODES-1,
    "episode_rewards": episode_rewards,
    "episode_score": episode_score,
    "epsilon": agent.epsilon
})

print("All done. Outputs in", SAVE_DIR)
