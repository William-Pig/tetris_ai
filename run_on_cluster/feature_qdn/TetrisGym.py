from collections import namedtuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from TetrisGame import TetrisGame

Observation = namedtuple("Observation", ["board", "curr_id", "next_id"])
PIECE2IDX   = {'I':0,'J':1,'L':2,'O':3,'S':4,'Z':5,'T':6}

class TetrisGym:
    def __init__(self, width=10, height=20, max_steps=None, render_mode='skip'):
        """
        - state_mode: 'flat', 'tensor', 'features'
        - render_mode: 'skip', 'render', 'capture'
        """
        self.game = TetrisGame(width, height)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.step_count = 0
        self.valid_actions = []  # cache valid_actions in each cycle
        self.frames = []  # cache frames for capture mode

        # Precompute the full action space: (rotation_idx, x_position)
        self.full_action_space = self._build_action_space()
        self.action_to_id = {action: i for i, action in enumerate(self.full_action_space)}  # maps action to an id
        self.id_to_action = {i: action for i, action in enumerate(self.full_action_space)}  # maps id back to action

        self.reset()  # initialize the environment to be playable

    def _build_action_space(self):
        """
        Build all possible (rotation, x) combo over all pieces. 
        i.e., the union of all possible pairs across all Tetromino types
        """
        seen = set()
        actions = []
        for _, rotations in self.game.TETROMINOES.items():
            for rot_idx, piece in enumerate(rotations):  # piece = the actual, rotated piece
                _, piece_width = piece.shape
                for x in range(self.game.width - piece_width + 1):
                    key = (rot_idx, x)
                    if key not in seen:
                        seen.add(key)
                        actions.append(key)
        return actions



    def _obs(self):
        """Raw state observation to be parsed in the agents"""
        board = self.game.board.astype(np.uint8)
        curr_id, _ = self.game.current_piece
        next_id, _ = self.game.next_piece
        return Observation(board, PIECE2IDX[curr_id], PIECE2IDX[next_id])



    def reset(self):
        """Resets the gym environment for a new episode (learning round)"""
        self.game.reset_board()
        self.game.spawn_new_piece()  # Since TetrisGame.reset_board does not push the next Tetromino to current_piece
        self.valid_actions = self.game.get_valid_actions()  # get valid_actions list
        self.step_count = 0
        self.frames = []
        return self._obs()



    def step(self, action_id):
        if self.game.game_over:
            raise Exception("Cannot step in a finished episode. call reset()")

        rot_idx, x = self.id_to_action[action_id]
        info = self.game.update_board(rot_idx, x)
        self.step_count += 1

        # check if training is done: when game is over or exceeds max training steps
        self.game.check_game_over()
        done = self.game.game_over or (self.max_steps is not None and self.step_count >= self.max_steps)

        # next step
        if not done:
            self.game.spawn_new_piece()
            self.valid_actions = self.game.get_valid_actions()
            if not self.valid_actions:  # If no valid action, the game ends.
                self.game.game_over = True
                done = True

        # Get observation
        obs = self._obs()

        # Rendering
        if self.render_mode == 'render':
            self.render(info)
        elif self.render_mode == 'capture':
            self.capture(info)

        return obs, 0.0, done, info  # 0.0 is reward, will be used by agent



    def get_valid_action_ids(self):
        return [self.action_to_id[a] for a in self.valid_actions]



    def render(self, info=None):
        """Renders the state of game. """
        placement_mask = info.get("placement_mask") if info else None
        pre_clear_board = info.get("pre_clear_board") if info else None
        self.game.render(valid_actions=self.valid_actions,
                        placement_mask=placement_mask,
                        pre_clear_board=pre_clear_board)


    def capture(self, info=None):
        placement_mask = info.get("placement_mask") if info else None
        pre_clear_board = info.get("pre_clear_board") if info else None
        fig = self.game.render(valid_actions=self.valid_actions,
                            placement_mask=placement_mask,
                            pre_clear_board=pre_clear_board,
                            return_fig=True)
        self.frames.append(fig)
        plt.close(fig)  # free memory

    def save_gif(self, filename, fps=2):
        """Save collected frames into a gif. Only works if render_mode='capture'"""
        if not self.frames:
            print("No frames to save. Set render_mode='capture'.")
            return

        images = []
        for fig in self.frames:
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)
            plt.close(fig)  # free memory

        imageio.mimsave(filename, images, fps=fps)
