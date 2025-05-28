# tetris_ai

A reinforcement learning application on Tetris (update may 12 2025).

## Files

- `project_report.pdf` is the project write-up
- `main.ipynb` is my everyday data analysis notebook that trains the model
- `TetrisGame.py` is the game's core
- `TetrisGym.py` is our custom gym environmment
- The `agents` folder contains all models used, which include:
    - A tabular Q-learning agent
    - A Deep Q-network agent with MLP architecture
    - A Deep Q-network agent with CNN architecture
    - A model-based value-learning agent with MLP architecture
- `run_on_cluster` contains some training data and agent

## References
- Tetris game rule: https://tetris.wiki/Tetris_Guideline
- Tetris agent example: https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
- Another Tetris agent example: https://github.com/nuno-faria/tetris-ai?tab=readme-ov-file
- And Another Tetris agent example: https://github.com/vietnh1009/Tetris-deep-Q-learning-pytorch
- Tetris agent project paper example: https://openreview.net/pdf?id=8TLyqLGQ7Tg
- Deep Q Learning code example: https://github.com/keon/deep-q-learning/blob/master/dqn.py
- The famous Atari paper on DQN: http://arxiv.org/abs/1312.5602
- Double DQN paper: http://arxiv.org/abs/1509.06461
- Imitation Learning paper: https://doi.org/10.1038/nature16961