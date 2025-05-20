#!/usr/bin/env python
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

from TetrisGym import TetrisGym
from feature_mlp import MLPAgent

def main():
    # ——————————————————————————————
    # 0) Check that PyTorch sees the GPU
    # ——————————————————————————————
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA version:", torch.version.cuda)
    # If running under SLURM, echo the job ID
    slurm_job = os.environ.get("SLURM_JOB_ID")
    if slurm_job:
        print(f"SLURM_JOB_ID: {slurm_job}")

    # ——————————————————————————————
    # 1) Hyper-parameters
    # ——————————————————————————————
    BOARD_WIDTH     = 6
    BOARD_HEIGHT    = 6

    ALPHA           = 0.001   # learning rate
    GAMMA           = 0.9     # discount factor

    EPSILON_START   = 1.0
    EPSILON_MIN     = 0.01
    EPSILON_DECAY   = 0.9995

    NUM_EPISODES    = 5_000
    MAX_STEPS       = 1_000

    MEMORY_SIZE     = 10_000
    BATCH_SIZE      = 256
    TARGET_SYNC     = 32

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # ——————————————————————————————
    # 2) Output paths
    # ——————————————————————————————
    # You can override this via: export SAVE_DIR=/some/path
    SAVE_DIR = os.environ.get("SAVE_DIR", f"feature_mlp_checkpoints_board_{BOARD_WIDTH}*{BOARD_HEIGHT}_NUM_EPISODES_{NUM_EPISODES}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    LOAD_PATH = os.path.join(SAVE_DIR, "latest.pth")
    SAVE_PATH = LOAD_PATH  # same

    # ——————————————————————————————
    # 3) Initialize env + agent
    # ——————————————————————————————
    env = TetrisGym(
        width=BOARD_WIDTH,
        height=BOARD_HEIGHT,
        max_steps=MAX_STEPS
    )

    agent = MLPAgent(
        board_width    = BOARD_WIDTH,
        board_height   = BOARD_HEIGHT,
        alpha          = ALPHA,
        gamma          = GAMMA,
        eps_start      = EPSILON_START,
        eps_min        = EPSILON_MIN,
        eps_decay      = EPSILON_DECAY,
        memory_size    = MEMORY_SIZE,
        batch_size     = BATCH_SIZE,
        target_sync    = TARGET_SYNC,
        device         = DEVICE
    )

    # load prior checkpoint if present
    if os.path.isfile(LOAD_PATH):
        print("Loading checkpoint:", LOAD_PATH)
        agent.load_agent(LOAD_PATH)

    # ——————————————————————————————
    # 4) Train
    # ——————————————————————————————
    agent.train(env, episodes=NUM_EPISODES, max_steps=MAX_STEPS)

    # ——————————————————————————————
    # 5) Save everything
    # ——————————————————————————————
    print("Saving final model to:", SAVE_PATH)
    agent.save_agent(SAVE_PATH)

    # pickle out the rewards & scores
    with open(os.path.join(SAVE_DIR, "training_data.pkl"), "wb") as f:
        pickle.dump({
            "rewards": agent.rewards,
            "scores":  agent.scores
        }, f)

    # plot & save the learning curve
    avg_rewards = np.mean(np.array_split(np.array(agent.rewards), 100), axis=1)
    plt.figure()
    plt.plot(avg_rewards)
    plt.xlabel('Episode batch')
    plt.ylabel('Total Reward')
    plt.title('Training Progress: Avg Reward over 100 segments')
    plt.grid(True)
    curve_path = os.path.join(SAVE_DIR, "learning_curve.png")
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()
    print("Saved learning curve to:", curve_path)

    # save a final GIF
    gif_path = os.path.join(SAVE_DIR, f"ep_{NUM_EPISODES}.gif")
    agent.save_gif(save_path=gif_path)
    print("Saved gameplay GIF to:", gif_path)

if __name__ == "__main__":
    main()
