import torch
import numpy as np

from TetrisGame import TetrisGame


class TetrisGym:
    def __init__(self, width=10, height=20, max_steps=None, state_mode='flat', render_mode=False):
        self.game = TetrisGame(width, height)
        self.max_steps = max_steps
        self.state_mode = state_mode
        self.render_mode = render_mode
        self.step_count = 0
        self.valid_actions = []  # cache valid_actions in each cycle

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
        state = self.get_state()
        return state

    def get_state(self):
        """Returns the state of the game, depending on the state mode"""
        board = self.game.board
        curr_piece = self.game.current_piece[0]  # just the type for now
        next_piece = self.game.next_piece[0]

        if self.state_mode=='tensor':
            return self._extract_tensor(board, curr_piece, next_piece)
        elif self.state_mode=='flat':
            return self._extract_tensor_flat(board, curr_piece, next_piece)
        elif self.state_mode=='features':
            return self._extract_features(board, curr_piece, next_piece)

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

    def _extract_features(self, board, curr_piece, next_piece):
        """
        TODO: brainstorm features
        - Number of lines cleared
        - Number of holes
        - Bumpiness (sum of the difference between heights of adjacent pairs of columns)
        - Total Height
        - Max height
        - Min height
        - Max bumpiness
        - Next piece
        - Current piece
        """
        pass

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
        reward = info["reward"]

        # check if training is done: when game is over or exceeds max training steps
        self.game.check_game_over()
        done = self.game.game_over or (self.max_steps is not None and self.step_count >= self.max_steps)

        # next step
        if not done:
            self.game.spawn_new_piece()
            self.valid_actions = self.game.get_valid_actions()
        else:
            self.valid_actions = []  # empty possible actions
        next_state = self.get_state()

        # Rendering
        if self.render_mode:
            self.render()

        return next_state, reward, done, info
    
    def render(self):
        """Renders the state of game. TODO: include action chosen by the RL"""
        self.game.render(valid_actions=self.valid_actions)

    def get_action_space_size(self):
        """Returns how many possible actions there are"""
        return len(self.full_action_space)
