import gymnasium as gym

class ActionRestrictWrapper(gym.ActionWrapper):
    """
    Restrict Pong's 6-action space to {NOOP, UP, DOWN}.
    Map 0->NOOP, 1->UP, 2->DOWN
    """
    def __init__(self, env):
        super().__init__(env)
        # Original Pong actions
        # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        self._restricted_actions = [0, 2, 3]
        self.action_space = gym.spaces.Discrete(len(self._restricted_actions))

    def action(self, action):
        return self._restricted_actions[action]

    def reverse_action(self, act):
        return self._restricted_actions.index(act)