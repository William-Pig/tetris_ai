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
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.state_mode = state_mode
        self.render_mode = render_mode
        self.step_count = 0
        self.valid_actions = []
        self.frames = []

        # reward-shaping hyperparameters
        self.hole_creation_penalty = 10.0   # penalty for new holes created
        self.hole_fill_reward      = 5.0    # reward for holes filled
        self.well_depth_weight     = 1.0    # penalty per unit well depth
        self.height_diff_weight    = 2.0    # penalty per unit height difference
        self.edge_gap_weight       = 2.0    # penalty per unit gap in rightmost two columns

        # Precompute action space
        self.full_action_space = self._build_action_space()
        self.action_to_id      = {action: i for i, action in enumerate(self.full_action_space)}
        self.id_to_action      = {i: action for i, action in enumerate(self.full_action_space)}

        self.reset()

    def _build_action_space(self):
        seen = set()
        actions = []
        for _, rotations in self.game.TETROMINOES.items():
            for rot_idx, piece in enumerate(rotations):
                _, piece_width = piece.shape
                for x in range(self.game.width - piece_width + 1):
                    key = (rot_idx, x)
                    if key not in seen:
                        seen.add(key)
                        actions.append(key)
        return actions

    def reset(self):
        self.game.reset_board()
        self.game.spawn_new_piece()
        self.valid_actions = self.game.get_valid_actions()
        self.step_count = 0
        self.frames = []
        return self.get_state()

    def get_state(self):
        board = self.game.board
        curr_piece = self.game.current_piece[0]
        next_piece = self.game.next_piece[0]
        if self.state_mode == 'tensor':
            return self._extract_tensor(board, curr_piece, next_piece)
        elif self.state_mode == 'flat':
            return self._extract_tensor_flat(board, curr_piece, next_piece)
        elif self.state_mode == 'features':
            return self._extract_features(board, curr_piece, next_piece)

    def _extract_tensor(self, board, curr_piece, next_piece):
        h, w = board.shape
        channels = []
        piece_to_idx = {'I':0,'J':1,'L':2,'O':3,'S':4,'Z':5,'T':6}

        # C0: board
        channels.append(board.astype(np.float32))
        # C1-7: current piece
        curr_idx = piece_to_idx[curr_piece]
        for i in range(7):
            channels.append(np.full((h,w), 1.0 if i==curr_idx else 0.0, dtype=np.float32))
        # C8-14: next piece
        next_idx = piece_to_idx[next_piece]
        for i in range(7):
            channels.append(np.full((h,w), 1.0 if i==next_idx else 0.0, dtype=np.float32))

        return torch.from_numpy(np.stack(channels, axis=0))

    def _extract_tensor_flat(self, board, curr_piece, next_piece):
        v_board = board.flatten().astype(np.float32)
        piece_to_idx = {'I':0,'J':1,'L':2,'O':3,'S':4,'Z':5,'T':6}
        v_curr = self._one_hot(piece_to_idx[curr_piece])
        v_next = self._one_hot(piece_to_idx[next_piece])
        return torch.from_numpy(np.concatenate([v_board, v_curr, v_next]))

    def _one_hot(self, idx, size=7):
        v = np.zeros(size, dtype=np.float32)
        v[idx] = 1.0
        return v

    def _extract_features(self, board, curr_piece, next_piece):
        raise NotImplementedError

    def get_valid_action_ids(self):
        return [self.action_to_id[a] for a in self.valid_actions]

    def action_id_to_tuple(self, action_id):
        return self.id_to_action[action_id]

    def step(self, action_id):
        if self.game.game_over:
            raise Exception("Episode has finished.")

        rot_idx, x = self.id_to_action[action_id]
        board_before = self.game.board.copy()
        info = self.game.update_board(rot_idx, x)
        self.step_count += 1

        self.game.check_game_over()
        done = self.game.game_over or (self.max_steps is not None and self.step_count >= self.max_steps)

        reward = self._compute_reward(info, done, board_before)
        info["reward"] = reward

        if not done:
            self.game.spawn_new_piece()
            self.valid_actions = self.game.get_valid_actions()
            if not self.valid_actions:
                self.game.game_over = True
                done = True
        else:
            self.valid_actions = []

        next_state = self.get_state()
        if self.render_mode == 'render':
            self.render(info)
        elif self.render_mode == 'capture':
            self.capture(info)

        return next_state, reward, done, info

    def _compute_reward(self, info, done, board_before):
        board_after = self.game.board

        # 1) Line-clear reward
        lines = info.get("lines_cleared", 0)
        line_rewards = {0:0, 1:1, 2:3, 3:5, 4:10}
        reward = line_rewards.get(lines, 0) * 25

        # 2) Survival bonus
        reward += 0.2

        # compute heights and bumpiness
        heights = self.height - np.argmax(board_after[::-1, :], axis=0)
        reward -= 1 * heights.sum()
        bumpiness = np.abs(np.diff(heights)).sum()
        reward -= 2 * bumpiness

        # 3) Hole metrics
        holes_before = self._count_holes(board_before)
        holes_after  = self._count_holes(board_after)
        new_holes    = max(0, holes_after - holes_before)
        filled_holes = max(0, holes_before - holes_after)
        reward -= self.hole_creation_penalty * new_holes
        reward += self.hole_fill_reward      * filled_holes

        # 4) Well-depth penalty
        well_pen = 0
        for i in range(self.width):
            h = heights[i]
            left  = heights[i-1] if i>0 else self.height
            right = heights[i+1] if i<self.width-1 else self.height
            well_depth = max(0, min(left, right) - h)
            well_pen += well_depth
        reward -= self.well_depth_weight * well_pen

        # 5) Height-difference penalty
        height_diff = heights.max() - heights.min()
        reward -= self.height_diff_weight * height_diff

        # 6) Right-edge gap penalty
        max_h = heights.max()
        right_gap = (max_h - heights[-1]) + (max_h - heights[-2])
        reward -= self.edge_gap_weight * right_gap

        # 7) Death penalty
        if done and self.game.game_over:
            reward -= 99

        return reward

    def _count_holes(self, board):
        holes = 0
        for col in range(self.width):
            col_data = board[:, col]
            first_block = np.argmax(col_data)
            if col_data[first_block] == 1:
                holes += np.sum(col_data[first_block+1:] == 0)
        return holes

    def render(self, info=None):
        placement_mask   = info.get("placement_mask") if info else None
        pre_clear_board = info.get("pre_clear_board") if info else None
        self.game.render(valid_actions=self.valid_actions,
                         placement_mask=placement_mask,
                         pre_clear_board=pre_clear_board)

    def capture(self, info=None):
        placement_mask   = info.get("placement_mask") if info else None
        pre_clear_board = info.get("pre_clear_board") if info else None
        fig = self.game.render(valid_actions=self.valid_actions,
                               placement_mask=placement_mask,
                               pre_clear_board=pre_clear_board,
                               return_fig=True)
        self.frames.append(fig)
        plt.close(fig)

    def save_gif(self, filename, fps=2):
        if not self.frames:
            print("No frames to save.")
            return
        images = []
        for fig in self.frames:
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(img)
            plt.close(fig)
        imageio.mimsave(filename, images, fps=fps)
        print(f"GIF saved to {filename}")

    def get_action_space_size(self):
        return len(self.full_action_space)
    
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import imageio.v2 as imageio

# from TetrisGame import TetrisGame

# class TetrisGym:
#     def __init__(self, width=10, height=20, max_steps=None, state_mode='flat', render_mode='skip'):
#         """
#         - state_mode: 'flat', 'tensor', 'features'
#         - render_mode: 'skip', 'render', 'capture'
#         """
#         self.game = TetrisGame(width, height)
#         self.width = width
#         self.height = height
#         self.max_steps = max_steps
#         self.state_mode = state_mode
#         self.render_mode = render_mode
#         self.step_count = 0
#         self.valid_actions = []
#         self.frames = []


#         # Precompute action space
#         self.full_action_space = self._build_action_space()
#         self.action_to_id      = {action: i for i, action in enumerate(self.full_action_space)}
#         self.id_to_action      = {i: action for i, action in enumerate(self.full_action_space)}

#         self.reset()

#     def _build_action_space(self):
#         seen = set()
#         actions = []
#         for _, rotations in self.game.TETROMINOES.items():
#             for rot_idx, piece in enumerate(rotations):
#                 _, piece_width = piece.shape
#                 for x in range(self.game.width - piece_width + 1):
#                     key = (rot_idx, x)
#                     if key not in seen:
#                         seen.add(key)
#                         actions.append(key)
#         return actions

#     def reset(self):
#         self.game.reset_board()
#         self.game.spawn_new_piece()
#         self.valid_actions = self.game.get_valid_actions()
#         self.step_count = 0
#         self.frames = []
#         return self.get_state()

#     def get_state(self):
#         board = self.game.board
#         curr_piece = self.game.current_piece[0]
#         next_piece = self.game.next_piece[0]
#         if self.state_mode == 'tensor':
#             return self._extract_tensor(board, curr_piece, next_piece)
#         elif self.state_mode == 'flat':
#             return self._extract_tensor_flat(board, curr_piece, next_piece)
#         elif self.state_mode == 'features':
#             return self._extract_features(board, curr_piece, next_piece)

#     def _extract_tensor(self, board, curr_piece, next_piece):
#         h, w = board.shape
#         channels = []
#         piece_to_idx = {'I':0,'J':1,'L':2,'O':3,'S':4,'Z':5,'T':6}

#         # C0: board
#         channels.append(board.astype(np.float32))
#         # C1-7: current piece one-hot
#         curr_idx = piece_to_idx[curr_piece]
#         for i in range(7):
#             channels.append(np.full((h,w), 1.0 if i==curr_idx else 0.0, dtype=np.float32))
#         # C8-14: next piece one-hot
#         next_idx = piece_to_idx[next_piece]
#         for i in range(7):
#             channels.append(np.full((h,w), 1.0 if i==next_idx else 0.0, dtype=np.float32))

#         return torch.from_numpy(np.stack(channels, axis=0))

#     def _extract_tensor_flat(self, board, curr_piece, next_piece):
#         v_board = board.flatten().astype(np.float32)
#         piece_to_idx = {'I':0,'J':1,'L':2,'O':3,'S':4,'Z':5,'T':6}
#         v_curr = self._one_hot(piece_to_idx[curr_piece])
#         v_next = self._one_hot(piece_to_idx[next_piece])
#         return torch.from_numpy(np.concatenate([v_board, v_curr, v_next]))

#     def _one_hot(self, idx, size=7):
#         v = np.zeros(size, dtype=np.float32)
#         v[idx] = 1.0
#         return v

#     def _extract_features(self, board, curr_piece, next_piece):
#         # placeholder for custom features
#         raise NotImplementedError

#     def get_valid_action_ids(self):
#         return [self.action_to_id[a] for a in self.valid_actions]

#     def action_id_to_tuple(self, action_id):
#         return self.id_to_action[action_id]

#     def step(self, action_id):
#         if self.game.game_over:
#             raise Exception("Episode has finished.")

#         rot_idx, x = self.id_to_action[action_id]
#         board_before = self.game.board.copy()
#         info = self.game.update_board(rot_idx, x)
#         self.step_count += 1

#         self.game.check_game_over()
#         done = self.game.game_over or (self.max_steps is not None and self.step_count >= self.max_steps)

#         reward = self._compute_reward(info, done, board_before)
#         info["reward"] = reward

#         if not done:
#             self.game.spawn_new_piece()
#             self.valid_actions = self.game.get_valid_actions()
#             if not self.valid_actions:
#                 self.game.game_over = True
#                 done = True
#         else:
#             self.valid_actions = []

#         next_state = self.get_state()
#         if self.render_mode == 'render':
#             self.render(info)
#         elif self.render_mode == 'capture':
#             self.capture(info)

#         return next_state, reward, done, info

#     def _compute_reward(self, info, done, board_before):
#         board_after = self.game.board

#         # 1) Line-clear reward
#         lines = info.get("lines_cleared", 0)
#         line_rewards = {0:0, 1:1, 2:3, 3:5, 4:10}
#         reward = line_rewards.get(lines, 0) * 25

#         # 2) Survival bonus
#         reward += 0.2

#         # 3a) Aggregate height & bumpiness
#         heights = self.height - np.argmax(board_after[::-1, :], axis=0)
#         reward -= 1 * heights.sum()
#         bumpiness = np.abs(np.diff(heights)).sum()
#         reward -= 2 * bumpiness

#         # 3b) Hole-creation penalty
#         holes_before = self._count_holes(board_before)
#         holes_after  = self._count_holes(board_after)
#         holes_created = holes_after - holes_before
#         reward -= 5.0 * holes_created

#         # 3c) Well-depth penalty
#         well_penalty = 0
#         for i in range(self.width):
#             h = heights[i]
#             left  = heights[i-1] if i>0 else self.height
#             right = heights[i+1] if i<self.width-1 else self.height
#             well_depth = max(0, min(left, right) - h)
#             well_penalty += well_depth
#         well_depth_weight = 0.5
#         reward -= well_depth_weight * well_penalty

#         # 3d) Height-difference penalty
#         height_diff = heights.max() - heights.min()
#         height_diff_weight = 1
#         reward -= height_diff_weight * height_diff

#         # 4) Death penalty
#         if done and self.game.game_over:
#             reward -= 9999

#         return reward



#     def _count_holes(self, board):
#         holes = 0
#         for col in range(self.width):
#             col_data = board[:, col]
#             first_block = np.argmax(col_data)
#             if col_data[first_block] == 1:
#                 holes += np.sum(col_data[first_block+1:] == 0)
#         return holes

#     def render(self, info=None):
#         placement_mask   = info.get("placement_mask") if info else None
#         pre_clear_board = info.get("pre_clear_board") if info else None
#         self.game.render(valid_actions=self.valid_actions,
#                          placement_mask=placement_mask,
#                          pre_clear_board=pre_clear_board)

#     def capture(self, info=None):
#         placement_mask   = info.get("placement_mask") if info else None
#         pre_clear_board = info.get("pre_clear_board") if info else None
#         fig = self.game.render(valid_actions=self.valid_actions,
#                                placement_mask=placement_mask,
#                                pre_clear_board=pre_clear_board,
#                                return_fig=True)
#         self.frames.append(fig)
#         plt.close(fig)

#     def save_gif(self, filename, fps=2):
#         if not self.frames:
#             print("No frames to save.")
#             return
#         images = []
#         for fig in self.frames:
#             fig.canvas.draw()
#             img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#             img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#             images.append(img)
#             plt.close(fig)
#         imageio.mimsave(filename, images, fps=fps)
#         print(f"GIF saved to {filename}")

#     def get_action_space_size(self):
#         return len(self.full_action_space)

# # 
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import imageio.v2 as imageio

# from TetrisGame import TetrisGame


# class TetrisGym:
#     def __init__(self, width=10, height=20, max_steps=None, state_mode='flat', render_mode='skip'):
#         """
#         - state_mode: 'flat', 'tensor', 'features'
#         - render_mode: 'skip', 'render', 'capture'
#         """
        
#         self.game = TetrisGame(width, height)
#         self.width  = width        
#         self.height = height      
#         self.max_steps = max_steps
#         self.state_mode = state_mode
#         self.render_mode = render_mode
#         self.step_count = 0
#         self.valid_actions = []  # cache valid_actions in each cycle
#         self.frames = []  # cache frames for capture mode

#         # Precompute the full action space: (rotation_idx, x_position)
#         self.full_action_space = self._build_action_space()
#         self.action_to_id = {action: i for i, action in enumerate(self.full_action_space)}  # maps action to an id
#         self.id_to_action = {i: action for i, action in enumerate(self.full_action_space)}  # maps id back to action

#         self.reset()  # initialize the environment to be playable

#     def _build_action_space(self):
#         """
#         Build all possible (rotation, x) combo over all pieces. 
#         i.e., the union of all possible pairs across all Tetromino types
#         """
#         seen = set()
#         actions = []
#         for _, rotations in self.game.TETROMINOES.items():
#             for rot_idx, piece in enumerate(rotations):  # piece = the actual, rotated piece
#                 _, piece_width = piece.shape
#                 for x in range(self.game.width - piece_width + 1):
#                     key = (rot_idx, x)
#                     if key not in seen:
#                         seen.add(key)
#                         actions.append(key)
#         return actions



#     def reset(self):
#         """Resets the gym environment for a new episode (learning round)"""
#         self.game.reset_board()
#         self.game.spawn_new_piece()  # Since TetrisGame.reset_board does not push the next Tetromino to current_piece
#         self.valid_actions = self.game.get_valid_actions()  # get valid_actions list
#         self.step_count = 0
#         self.frames = []
#         state = self.get_state()
#         return state



#     def get_state(self):
#         """Returns the state of the game, depending on the state mode"""
#         board = self.game.board
#         curr_piece = self.game.current_piece[0]  # just the type for now
#         next_piece = self.game.next_piece[0]

#         if self.state_mode=='tensor':
#             return self._extract_tensor(board, curr_piece, next_piece)
#         elif self.state_mode=='flat':
#             return self._extract_tensor_flat(board, curr_piece, next_piece)
#         elif self.state_mode=='features':
#             return self._extract_features(board, curr_piece, next_piece)

#     def _extract_tensor(self, board, curr_piece, next_piece):
#         """Returns the state as a tensor with the actual 2D tetris board"""
#         h, w = board.shape
#         channels = []
#         piece_to_idx = {'I':0,'J':1,'L':2,'O':3,'S':4,'Z':5,'T':6}  # a lookup table to one-hot encode

#         # C0: the board
#         channels.append(board.astype(np.float32))

#         # C1-7: current piece one‑hots, maintain board dimension (full 1s or 0s) for CNN-friendliness
#         current_idx = piece_to_idx[curr_piece]
#         for piece_idx in range(7):
#             channels.append(
#                 np.full((h, w), 1.0 if piece_idx==current_idx else 0.0, dtype=np.float32)
#                 )

#         # C8-14: next piece one‑hots
#         next_idx = piece_to_idx[next_piece]
#         for piece_idx in range(7):
#             channels.append(
#                 np.full((h, w), 1.0 if piece_idx==next_idx else 0.0, dtype=np.float32)
#                 )

#         return torch.from_numpy(np.stack(channels, axis=0))

#     def _extract_tensor_flat(self, board, curr_piece, next_piece):
#         """Returns the state as a tensor with a flatten tetris board"""
#         v_board = board.flatten().astype(np.float32)  # (board width*height,)
#         piece_to_idx = {'I':0,'J':1,'L':2,'O':3,'S':4,'Z':5,'T':6}  # a lookup table to one-hot encode
#         v_curr  = self._one_hot(piece_to_idx[curr_piece])  # (7,)
#         v_next  = self._one_hot(piece_to_idx[next_piece])  # (7,)
#         return torch.from_numpy(np.concatenate([v_board, v_curr, v_next]))  # (...,)
    
#     def _one_hot(self, idx, size=7):
#         """Creates a one-hot vector for the Tetromino shapes"""
#         v = np.zeros(size, dtype=np.float32)
#         v[idx] = 1.0
#         return v

#     def _extract_features(self, board, curr_piece, next_piece):
#         """
#         TODO: brainstorm features
#         - Number of lines cleared
#         - Number of holes
#         - Bumpiness (sum of the difference between heights of adjacent pairs of columns)
#         - Total Height
#         - Max height
#         - Min height
#         - Max bumpiness
#         - Next piece
#         - Current piece
#         """
#         pass



#     def get_valid_action_ids(self):
#         return [self.action_to_id[a] for a in self.valid_actions]

#     def action_id_to_tuple(self, action_id):
#         return self.id_to_action[action_id]



#     def step(self, action_id):
#         if self.game.game_over:
#             raise Exception("Cannot step in a finished episode.")

#         rot_idx, x = self.id_to_action[action_id]
#         info = self.game.update_board(rot_idx, x)
#         self.step_count += 1

#         # check if training is done: when game is over or exceeds max training steps
#         self.game.check_game_over()
#         done = self.game.game_over or (self.max_steps is not None and self.step_count >= self.max_steps)

#         # Compute reward
#         reward = self._compute_reward(info, done)
#         info["reward"] = reward  # record down reward

#         # next step
#         if not done:
#             self.game.spawn_new_piece()
#             self.valid_actions = self.game.get_valid_actions()
#             if not self.valid_actions:  # If no valid action, the game ends.
#                 self.game.game_over = True
#                 done = True
#         else:
#             self.valid_actions = []  # empty possible actions
#         next_state = self.get_state()

#         # Rendering
#         if self.render_mode == 'render':
#             self.render(info)
#         elif self.render_mode == 'capture':
#             self.capture(info)

#         return next_state, reward, done, info
    
#     def _compute_reward(self, info, done):
#         """
#         Compute a shaped reward combining:
#           • Strong positive for line clears (bigger for 3‐ and 4‐line clears)
#           • Small survival bonus each step
#           • Negative penalties for aggregate height, bumpiness, and holes
#           • Large penalty for death
#         """
#         board = self.game.board   

#         # 1) Line‐clear reward
#         lines = info.get("lines_cleared", 0)
#         line_rewards = {0: 0, 1: 1, 2: 3, 3: 5, 4: 10}
#         reward = line_rewards.get(lines, 0) * 10

#         # 2) Survival bonus
#         reward += 0.2

#         # 3) Heuristic penalties from the board
#         #   a) Column heights
#         heights = self.height - np.argmax(board[::-1, :], axis=0)
#         reward -= 0.1 * heights.sum()              # aggregate‐height penalty

#         #   b) Bumpiness (adjacent height differences)
#         bumpiness = np.abs(np.diff(heights)).sum()
#         reward -= 0.2 * bumpiness

#         #   c) Holes (empty cells under a block in each column)
#         holes = 0
#         for col in range(self.width):
#             col_data = board[:, col]
#             first_block = np.argmax(col_data)
#             if col_data[first_block] == 1:
#                 holes += np.sum(col_data[first_block+1:] == 0)
#         reward -= 5 * holes

#         # 4) Death penalty if the game ends
#         if done and self.game.game_over:
#             reward -= 9999

#         return reward



#     def render(self, info=None):
#         """Renders the state of game. """
#         placement_mask = info.get("placement_mask") if info else None
#         pre_clear_board = info.get("pre_clear_board") if info else None
#         self.game.render(valid_actions=self.valid_actions,
#                         placement_mask=placement_mask,
#                         pre_clear_board=pre_clear_board)


#     def capture(self, info=None):
#         placement_mask = info.get("placement_mask") if info else None
#         pre_clear_board = info.get("pre_clear_board") if info else None
#         fig = self.game.render(valid_actions=self.valid_actions,
#                             placement_mask=placement_mask,
#                             pre_clear_board=pre_clear_board,
#                             return_fig=True)
#         self.frames.append(fig)
#         plt.close(fig)  # free memory

#     def save_gif(self, filename, fps=2):
#         """Save collected frames into a gif. Only works if render_mode='capture'"""
#         if not self.frames:
#             print("No frames to save. Set render_mode='capture'.")
#             return

#         images = []
#         for fig in self.frames:
#             fig.canvas.draw()
#             image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#             image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#             images.append(image)
#             plt.close(fig)  # free memory

#         imageio.mimsave(filename, images, fps=fps)
#         print(f"GIF saved to {filename}")

#     def get_action_space_size(self):
#         """Returns how many possible actions there are"""
#         return len(self.full_action_space)
