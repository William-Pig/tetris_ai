import os
import random
import pickle
from collections import defaultdict, deque

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from TetrisGym import TetrisGym


def stringify_state(state):
    """Convert torch.Tensor or numpy state to a hashable key."""
    try:
        arr = state.numpy()
    except AttributeError:
        arr = state
    return arr.tobytes()


def select_action(state_key, valid_actions, Q, epsilon):
    """Epsilon-greedy selection among valid action IDs."""
    if random.random() < epsilon:
        return random.choice(valid_actions)
    q_values = Q[state_key]
    return max(valid_actions, key=lambda a: q_values[a])


def run_experiment(alpha, gamma, output_dir):
    # Fixed training parameters
    width, height = 6, 6
    state_mode = 'flat'
    num_episodes = 10_000 #_000
    max_steps = 500
    epsilon = 0.1
    epsilon_min = 0.01
    epsilon_decay = 0.99999
    rolling_window_size = 1000

    # Prepare environment and Q-table
    env = TetrisGym(width=width, height=height, state_mode=state_mode)
    action_space_size = env.get_action_space_size()
    Q = defaultdict(lambda: np.zeros(action_space_size))
    episode_rewards = []
    rolling_window = deque(maxlen=rolling_window_size)
    rolling_sum = 0.0

    # Training loop
    for ep in trange(num_episodes, desc=f"alpha={alpha}, gamma={gamma}"):
        state = env.reset()
        state_key = stringify_state(state)
        total_reward = 0.0

        for step in range(max_steps):
            valid_actions = env.get_valid_action_ids()
            if not valid_actions:
                break
            action = select_action(state_key, valid_actions, Q, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_key = stringify_state(next_state)
            total_reward += reward

            # Compute TD target
            if done:
                target = reward
            else:
                next_valid = env.get_valid_action_ids()
                best_next = max(Q[next_key][a] for a in next_valid)
                target = reward + gamma * best_next

            # Q-update
            Q[state_key][action] += alpha * (target - Q[state_key][action])

            if done:
                break
            state_key = next_key

        # End-of-episode
        episode_rewards.append(total_reward)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        if len(rolling_window) == rolling_window_size:
            rolling_sum -= rolling_window[0]
        rolling_window.append(total_reward)
        rolling_sum += total_reward

    # Save results
    prefix = f"alpha{alpha}_gamma{gamma}"
    pkl_path = os.path.join(output_dir, f"{prefix}_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"Q_table": dict(Q), "episode_rewards": episode_rewards}, f)

    # Plot training progress
    batches = 100
    avg_rewards = np.mean(np.array_split(np.array(episode_rewards), batches), axis=1)
    plt.figure()
    plt.plot(avg_rewards)
    plt.xlabel("Episode batch")
    plt.ylabel("Total Reward")
    plt.title(f"Training Progress: Average Reward over 100 Equal Training Segments: alpha={alpha}, gamma={gamma}")
    plt.grid(True)
    plt.tight_layout()
    png_path = os.path.join(output_dir, f"{prefix}_progress.png")
    plt.savefig(png_path)
    plt.close()

    print(f"Saved: {pkl_path} and {png_path}")


def main():
    # Create output directory
    output_dir = f"tabular_qlearning_result_10000_episode"
    os.makedirs(output_dir, exist_ok=True)

    alphas = [0.05, 0.1, 0.2, 0.5]
    gammas = [0.5, 0.8, 0.9, 0.95]

    for alpha in alphas:
        for gamma in gammas:
            random.seed(42)
            np.random.seed(42)
            run_experiment(alpha, gamma, output_dir)

if __name__ == "__main__":
    main()



# import random
# import pickle
# from collections import defaultdict, deque

# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import trange

# from TetrisGym import TetrisGym


# def stringify_state(state):
#     """Convert torch.Tensor or numpy state to a hashable key."""
#     try:
#         arr = state.numpy()
#     except AttributeError:
#         arr = state
#     return arr.tobytes()


# def select_action(state_key, valid_actions, Q, epsilon):
#     """Epsilon-greedy selection among valid action IDs."""
#     if random.random() < epsilon:
#         return random.choice(valid_actions)
#     q_values = Q[state_key]
#     return max(valid_actions, key=lambda a: q_values[a])


# def main():
#     # Fixed training parameters
#     width = 6
#     height = 6
#     state_mode = 'flat'
#     num_episodes = 100_000 #1_000_000
#     max_steps = 500
#     alpha = 0.2
#     gamma = 0.9
#     epsilon = 0.1
#     epsilon_min = 0.01
#     epsilon_decay = 0.99999
#     rolling_window_size = 1000
#     output_prefix = 'tabular_qlearning'

#     # Reproducibility
#     random.seed(42)
#     np.random.seed(42)

#     # Initialize environment and Q-table
#     env = TetrisGym(width=width, height=height, state_mode=state_mode)
#     action_space_size = env.get_action_space_size()
#     Q = defaultdict(lambda: np.zeros(action_space_size))
#     episode_rewards = []
#     rolling_window = deque(maxlen=rolling_window_size)
#     rolling_sum = 0.0

#     # Training loop
#     for ep in trange(num_episodes, desc="Training episodes"):
#         state = env.reset()
#         state_key = stringify_state(state)
#         total_reward = 0.0

#         for step in range(max_steps):
#             valid_actions = env.get_valid_action_ids()
#             if not valid_actions:
#                 break
#             action = select_action(state_key, valid_actions, Q, epsilon)
#             next_state, reward, done, _ = env.step(action)
#             next_key = stringify_state(next_state)
#             total_reward += reward

#             # Compute target
#             if done:
#                 target = reward
#             else:
#                 next_valid = env.get_valid_action_ids()
#                 best_next = max(Q[next_key][a] for a in next_valid)
#                 target = reward + gamma * best_next

#             # Q-update
#             Q[state_key][action] += alpha * (target - Q[state_key][action])

#             if done:
#                 break
#             state_key = next_key

#         # End-of-episode updates
#         episode_rewards.append(total_reward)
#         if epsilon > epsilon_min:
#             epsilon *= epsilon_decay

#         if len(rolling_window) == rolling_window.maxlen:
#             rolling_sum -= rolling_window[0]
#         rolling_window.append(total_reward)
#         rolling_sum += total_reward

#     # Save results
#     results = {"Q_table": dict(Q), "episode_rewards": episode_rewards}
#     with open(f"{output_prefix}_results.pkl", "wb") as f:
#         pickle.dump(results, f)

#     # Plot training progress
#     batches = 100
#     avg_rewards = np.mean(np.array_split(np.array(episode_rewards), batches), axis=1)
#     plt.figure()
#     plt.plot(avg_rewards)
#     plt.xlabel("Episode batch")
#     plt.ylabel("Total Reward")
#     plt.title(f"Training Progress: Average Reward over {batches} batches")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"{output_prefix}_progress.png")

#     print(f"Training complete. Saved results to {output_prefix}_results.pkl and plot to {output_prefix}_progress.png.")


# if __name__ == "__main__":
#     main()
