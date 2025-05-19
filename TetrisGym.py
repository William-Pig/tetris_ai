import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from TetrisGame import TetrisGame


class TetrisGym:
    def __init__(self, width=10, height=20, max_steps=None, state_mode='flat', render_mode='skip'):
        """
        - state_mode: 'flat', 'tensor', 'features'
        - render_mode: 'skip', 'render', 'capture'
        """
        self.game = TetrisGame(width, height)
        self.max_steps = max_steps
        self.state_mode = state_mode
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



    def reset(self):
        """Resets the gym environment for a new episode (learning round)"""
        self.game.reset_board()
        self.game.spawn_new_piece()  # Since TetrisGame.reset_board does not push the next Tetromino to current_piece
        self.valid_actions = self.game.get_valid_actions()  # get valid_actions list
        self.step_count = 0
        self.frames = []
        state = self.get_state()
        return state

    def get_state(self, features_dict=None):
        """Returns the state of the game, depending on the state mode"""
        board = self.game.board
        curr_piece = self.game.current_piece[0]  # just the type for now
        next_piece = self.game.next_piece[0]

        if self.state_mode=='flat':
            return self._extract_tensor_flat(board, curr_piece, next_piece)
        elif self.state_mode=='tensor':
            return self._extract_tensor(board, curr_piece, next_piece)
        elif self.state_mode=='features':
            if features_dict is None:  # first time computation, only for reset()
                features_dict = self._compute_board_features(board)
            tensor = self._extract_tensor(board, curr_piece, next_piece)
            features = self._extract_features(features_dict, curr_piece, next_piece)
            return (tensor, features)

    def _extract_tensor(self, board, curr_piece, next_piece):
        """Returns the state as a tensor with the actual 2D tetris board"""
        h, w = board.shape
        channels = []
        piece_to_idx = {'I':0,'J':1,'L':2,'O':3,'S':4,'Z':5,'T':6}  # a lookup table to one-hot encode
        # C0: the board
        channels.append(board.astype(np.float32))
        # C1-7: current piece one‑hots, maintain board dimension (full 1s or 0s) for CNN-friendliness
        current_idx = piece_to_idx[curr_piece]
        for piece_idx in range(7):
            channels.append(
                np.full((h, w), 1.0 if piece_idx==current_idx else 0.0, dtype=np.float32)
                )
        # C8-14: next piece one‑hots
        next_idx = piece_to_idx[next_piece]
        for piece_idx in range(7):
            channels.append(
                np.full((h, w), 1.0 if piece_idx==next_idx else 0.0, dtype=np.float32)
                )
        return torch.from_numpy(np.stack(channels, axis=0))

    def _extract_tensor_flat(self, board, curr_piece, next_piece):
        """Returns the state as a tensor with a flatten tetris board"""
        v_board = board.flatten().astype(np.float32)  # (board width*height,)
        piece_to_idx = {'I':0,'J':1,'L':2,'O':3,'S':4,'Z':5,'T':6}  # a lookup table to one-hot encode
        v_curr  = self._one_hot(piece_to_idx[curr_piece])  # (7,)
        v_next  = self._one_hot(piece_to_idx[next_piece])  # (7,)
        return torch.from_numpy(np.concatenate([v_board, v_curr, v_next]))  # (...,)
    
    def _one_hot(self, idx, size=7):
        """Creates a one-hot vector for the Tetromino shapes"""
        v = np.zeros(size, dtype=np.float32)
        v[idx] = 1.0
        return v

    def _extract_features(self, features_dict, curr_piece, next_piece):
        # One-hot encode current and next piece
        piece_to_idx = {'I':0, 'J':1, 'L':2, 'O':3, 'S':4, 'Z':5, 'T':6}
        curr_idx = piece_to_idx[curr_piece]
        next_idx = piece_to_idx[next_piece]
        v_curr = self._one_hot(curr_idx)  # size 7
        v_next = self._one_hot(next_idx)  # size 7

        # other features
        ordered_keys = ["max_h","min_h","total_h","max_bump","total_bump","total_holes"]  # explicit order, for safety
        features = np.asarray([features_dict[k] for k in ordered_keys], dtype=np.float32)

        return torch.tensor(np.concatenate([features, v_curr, v_next]), dtype=torch.float32)



    def get_valid_action_ids(self):
        return [self.action_to_id[a] for a in self.valid_actions]

    def action_id_to_tuple(self, action_id):
        return self.id_to_action[action_id]



    def step(self, action_id):
        if self.game.game_over:
            raise Exception("Cannot step in a finished episode.")

        rot_idx, x = self.id_to_action[action_id]
        info = self.game.update_board(rot_idx, x)
        self.step_count += 1

        # check if training is done: when game is over or exceeds max training steps
        self.game.check_game_over()
        done = self.game.game_over or (self.max_steps is not None and self.step_count >= self.max_steps)

        # Compute board features
        if self.state_mode == 'features':
            features_dict = self._compute_board_features(self.game.board)
        else:
            features_dict = None

        # Compute reward
        reward = self._compute_reward(info, done, features_dict)
        info["reward"] = reward  # record down reward

        # next step
        if not done:
            self.game.spawn_new_piece()
            self.valid_actions = self.game.get_valid_actions()
            if not self.valid_actions:  # If no valid action, the game ends.
                self.game.game_over = True
                done = True
        else:
            self.valid_actions = []  # empty possible actions
        next_state = self.get_state(features_dict)

        # Rendering
        if self.render_mode == 'render':
            self.render(info)
        elif self.render_mode == 'capture':
            self.capture(info)

        return next_state, reward, done, info

    def _compute_reward(self, info, done, features_dict):
        reward = 0
    
        lines_cleared = info["lines_cleared"]
        # clear_line_reward = {0: 0, 1: 2, 2: 5, 3: 15, 4: 60}.get(lines_cleared, 0) * self.game.width
        clear_line_reward = 1 + lines_cleared**2 * self.game.width
        reward += clear_line_reward

        survival_reward = 0
        reward += survival_reward

        death_reward = -2 if done and self.game.game_over else 0
        reward += death_reward

        if self.state_mode=='features' and features_dict is not None:
            reward += -0. * features_dict["total_h"]
            reward += -0. * features_dict["total_holes"]
            reward += -0. * features_dict["total_bump"]
        return reward

    def _compute_board_features(self, board):
        """When chaning features, be sure to update _extract_features' ordered key"""
        h, _ = board.shape
        filled = (board != 0)  # mask for filled cells
        filled_rows = np.argmax(filled, axis=0)  # first filled row per column, i.e. max height of each col
        empty_cols = ~np.any(filled, axis=0)
        filled_rows[empty_cols] = h  # if column empty, treat as full height from bottom

        col_heights = h - filled_rows  # vector of shape (w,)
        abs_height_diff = np.abs(np.diff(col_heights))

        max_height = float(np.max(col_heights))
        min_height = float(np.min(col_heights))
        total_height = float(np.sum(col_heights))
        max_bumpiness = float(np.max(abs_height_diff))
        total_bumpiness = float(np.sum(abs_height_diff))

        accum_filled = np.maximum.accumulate(filled, axis=0)  # accumulate: once we see a 1, every row below keeps 1
        holes = (~filled) & accum_filled  # hole if filled accumulatively but not actually filled
        total_holes = float(holes.sum())

        return {
            "max_h": max_height,
            "min_h": min_height,
            "total_h": total_height,
            "max_bump": max_bumpiness,
            "total_bump": total_bumpiness,
            "total_holes": total_holes,
        }



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
        print(f"GIF saved to {filename}")

    def get_action_space_size(self):
        """Returns how many possible actions there are"""
        return len(self.full_action_space)
