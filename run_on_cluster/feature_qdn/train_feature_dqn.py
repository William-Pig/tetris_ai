#!/usr/bin/env python
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

from TetrisGym import TetrisGym
from feature_mlp import MLPAgent

# ——————————————————————————————
# Configurable batch sizes and early‐stop settings
# ——————————————————————————————
BATCH_SIZES        = [32, 64, 128]
EARLY_STOP_WINDOW  = 100        # compute moving average over last 100 episodes
EARLY_STOP_THRESH  = 0.05       # stop if Δμ₁₀₀ < 0.05
EARLY_STOP_ROUNDS  = 3          # require stability for 3 consecutive checks
CHUNK_SIZE         = EARLY_STOP_WINDOW  # train in chunks of 100 eps

def run_experiment(batch_size):
    print(f"\n=== Experiment: batch_size={batch_size} ===")
    # hyper-parameters
    BOARD_WIDTH, BOARD_HEIGHT = 10, 20
    ALPHA, GAMMA     = 0.001, 0.9
    EPS_PARAMS       = dict(start=1.0, min=0.01, decay=0.9995)
    NUM_EPISODES     = 100_000
    MAX_STEPS        = 10_000
    MEMORY_SIZE      = 10_000
    TARGET_SYNC      = 32
    DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # output dirs
    # SAVE_DIR = f"checkpoints_bs{batch_size}"
    SAVE_DIR = f"feature_mlp_checkpoints_board_{BOARD_WIDTH}*{BOARD_HEIGHT}_NUM_EPISODES_{NUM_EPISODES}_MAX_STEPS_{MAX_STEPS}_checkpoints_bs{batch_size}_episodes{NUM_EPISODES}"

    os.makedirs(SAVE_DIR, exist_ok=True)
    CKPT_PATH = os.path.join(SAVE_DIR, "latest.pth")

    # env & agent
    env = TetrisGym(width=BOARD_WIDTH, height=BOARD_HEIGHT, max_steps=MAX_STEPS)
    agent = MLPAgent(
        board_width   = BOARD_WIDTH,
        board_height  = BOARD_HEIGHT,
        alpha         = ALPHA,
        gamma         = GAMMA,
        eps_start     = EPS_PARAMS['start'],
        eps_min       = EPS_PARAMS['min'],
        eps_decay     = EPS_PARAMS['decay'],
        memory_size   = MEMORY_SIZE,
        batch_size    = batch_size,
        target_sync   = TARGET_SYNC,
        device        = DEVICE
    )
    if os.path.isfile(CKPT_PATH):
        agent.load_agent(CKPT_PATH)

    # training with early stopping
    prev_mu = None
    stable  = 0
    eps_done = 0

    while eps_done < NUM_EPISODES:
        # train next chunk
        to_train = min(CHUNK_SIZE, NUM_EPISODES - eps_done)
        agent.train(env, episodes=to_train, max_steps=MAX_STEPS)
        eps_done += to_train

        # compute recent window mean
        rewards = np.array(agent.rewards)
        if len(rewards) >= EARLY_STOP_WINDOW:
            mu = rewards[-EARLY_STOP_WINDOW:].mean()
            print(f"After {eps_done} eps, miu_100 = {mu:.3f}")
            if prev_mu is not None and abs(mu - prev_mu) < EARLY_STOP_THRESH:
                stable += 1
                print(f"  Δmiu < {EARLY_STOP_THRESH}, stable count = {stable}/{EARLY_STOP_ROUNDS}")
                if stable >= EARLY_STOP_ROUNDS:
                    print(f"Early stopping at {eps_done} episodes (miu_100 stabilized).")
                    break
            else:
                stable = 0
            prev_mu = mu

    # save final agent & data
    agent.save_agent(CKPT_PATH)
    with open(os.path.join(SAVE_DIR, "training_data.pkl"), "wb") as f:
        pickle.dump({"rewards": agent.rewards, "scores": agent.scores}, f)

    # plot learning curve
    avg_rewards = np.mean(np.array_split(np.array(agent.rewards), 100), axis=1)
    plt.figure()
    plt.plot(avg_rewards)
    plt.xlabel('Episode batch')
    plt.ylabel('Total Reward')
    plt.title(f'Avg Reward (batch={batch_size})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "learning_curve.png"))
    plt.close()

    # save a final GIF
    gif_path = os.path.join(SAVE_DIR, f"ep_{eps_done}.gif")
    agent.save_gif(save_path=gif_path)

def main():
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA version:", torch.version.cuda)
    if os.environ.get("SLURM_JOB_ID"):
        print("SLURM_JOB_ID:", os.environ["SLURM_JOB_ID"])
    for bs in BATCH_SIZES:
        run_experiment(bs)

if __name__ == "__main__":
    main()



# import os
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import torch

# from TetrisGym import TetrisGym
# from feature_mlp import MLPAgent

# # ——————————————————————————————
# # 0) Configurable batch sizes to sweep over
# # ——————————————————————————————
# BATCH_SIZES = [32, 64, 128]   # the values we want to try

# def run_experiment(batch_size):
#     print(f"\n=== Running experiment with BATCH_SIZE = {batch_size} ===")

#     # ——————————————————————————————
#     # 1) Hyper-parameters
#     # ——————————————————————————————
#     BOARD_WIDTH   = 10
#     BOARD_HEIGHT  = 20
#     ALPHA         = 0.001
#     GAMMA         = 0.9
#     EPS_START     = 1.0
#     EPS_MIN       = 0.01
#     EPS_DECAY     = 0.9995
#     NUM_EPISODES  = 100_000
#     MAX_STEPS     = 10_000
#     MEMORY_SIZE   = 10_000
#     TARGET_SYNC   = 32

#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", DEVICE)

#     # ——————————————————————————————
#     # 2) Output paths (unique per batch size)
#     # ——————————————————————————————
#     # SAVE_DIR = os.environ.get("SAVE_DIR", f"feature_mlp_checkpoints_board_{BOARD_WIDTH}*{BOARD_HEIGHT}_NUM_EPISODES_{NUM_EPISODES}_MAX_STEPS_{MAX_STEPS}")
#     SAVE_DIR = f"feature_mlp_checkpoints_board_{BOARD_WIDTH}*{BOARD_HEIGHT}_NUM_EPISODES_{NUM_EPISODES}_MAX_STEPS_{MAX_STEPS}_checkpoints_bs{batch_size}_episodes{NUM_EPISODES}"
#     os.makedirs(SAVE_DIR, exist_ok=True)
#     LOAD_PATH = os.path.join(SAVE_DIR, "latest.pth")
#     SAVE_PATH = LOAD_PATH

#     # ——————————————————————————————
#     # 3) Initialize env + agent
#     # ——————————————————————————————
#     env = TetrisGym(width=BOARD_WIDTH, height=BOARD_HEIGHT, max_steps=MAX_STEPS)
#     agent = MLPAgent(
#         board_width  = BOARD_WIDTH,
#         board_height = BOARD_HEIGHT,
#         alpha        = ALPHA,
#         gamma        = GAMMA,
#         eps_start    = EPS_START,
#         eps_min      = EPS_MIN,
#         eps_decay    = EPS_DECAY,
#         memory_size  = MEMORY_SIZE,
#         batch_size   = batch_size,      # <-- swept parameter
#         target_sync  = TARGET_SYNC,
#         device       = DEVICE
#     )

#     # load prior checkpoint if present
#     if os.path.isfile(LOAD_PATH):
#         print("Loading checkpoint:", LOAD_PATH)
#         agent.load_agent(LOAD_PATH)

#     # ——————————————————————————————
#     # 4) Train
#     # ——————————————————————————————
#     agent.train(env, episodes=NUM_EPISODES, max_steps=MAX_STEPS)

#     # ——————————————————————————————
#     # 5) Save model, data, and plots
#     # ——————————————————————————————
#     agent.save_agent(SAVE_PATH)
#     with open(os.path.join(SAVE_DIR, "training_data.pkl"), "wb") as f:
#         pickle.dump({"rewards": agent.rewards, "scores": agent.scores}, f)

#     # learning curve
#     avg_rewards = np.mean(np.array_split(np.array(agent.rewards), 100), axis=1)
#     plt.figure()
#     plt.plot(avg_rewards)
#     plt.xlabel('Episode batch')
#     plt.ylabel('Total Reward')
#     plt.title(f'Avg Reward (batch_size={batch_size})')
#     plt.grid(True)
#     plt.tight_layout()
#     curve_path = os.path.join(SAVE_DIR, "learning_curve.png")
#     plt.savefig(curve_path)
#     plt.close()
#     print("Saved learning curve to:", curve_path)

#     # final gameplay GIF
#     gif_path = os.path.join(SAVE_DIR, f"ep_{NUM_EPISODES}.gif")
#     agent.save_gif(save_path=gif_path)
#     print("Saved gameplay GIF to:", gif_path)


# def main():
#     print("CUDA available:", torch.cuda.is_available())
#     print("Torch CUDA version:", torch.version.cuda)
#     slurm_job = os.environ.get("SLURM_JOB_ID")
#     if slurm_job:
#         print(f"SLURM_JOB_ID: {slurm_job}")

#     for bs in BATCH_SIZES:
#         run_experiment(bs)

# if __name__ == "__main__":
#     main()

