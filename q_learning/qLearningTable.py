from random import uniform, randrange
import numpy as np

from model.env.action import Action


class QLearningTable:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.content = np.empty((len(state_space), len(action_space)))
        self.content.fill(1)

    def epsilon_greedy_policy(self, state_index, epsilon):
        random = uniform(0, 1)
        if random <= epsilon:
            action_mask = self.state_space[state_index].valid_actions_mask
            filtered_actions = np.multiply(action_mask, self.content[state_index])
            action_index = np.argmax(filtered_actions)
        else:
            action_mask = self.state_space[state_index].valid_actions_mask
            valid_indices = []
            for index, action in enumerate(action_mask):
                if action > 0:
                    valid_indices.append(index)
            valid_index = randrange(len(valid_indices))
            action_index = valid_indices[valid_index]
        return action_index

