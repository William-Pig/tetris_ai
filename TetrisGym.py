from TetrisGame import TetrisGame


class TetrisGym:
    def __init__(self, width=10, height=20, max_steps=None, render_mode=False):
        self.game = TetrisGame(width, height)
        self.max_steps = max_steps
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
        """Returns a flattened board, current and next piece info. TODO: to be improved, replace with full tensor"""
        board_flat = self.game.board.flatten()  # Will be an array
        curr_piece = self.game.current_piece[0]  # just the type for now
        next_piece = self.game.next_piece[0]
        return (board_flat, curr_piece, next_piece)  # Replace with full tensor later

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

        # check if training is done: when game is over or exceeds max trainign steps
        self.game.check_game_over()
        done = self.game.game_over or (self.max_steps is not None and self.step_count >= self.max_steps)

        # next step
        if not done:
            self.game.spawn_new_piece()
            self.valid_actions = self.game.get_valid_actions()
        else:
            self.valid_actions = []
        next_state = self.get_state()

        # Rendering, TODO: also include which action is chosen by the RL agent
        if self.render_mode:
            self.game.render(valid_actions=self.valid_actions)


        return next_state, reward, done, info
    
    def get_action_space_size(self):
        return len(self.full_action_space)
