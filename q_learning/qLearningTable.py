from random import uniform
import numpy as np

from model.env.action import Action


class QLearningTable:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.content = np.empty((len(state_space), len(action_space)))
        self.content.fill(1)

    def epsilon_greedy_policy(self, state_index, epsilon):
        action_mask = self.state_space[state_index].valid_actions_mask
        #print("action_mask", action_mask)
        filtered_actions = np.multiply(action_mask, self.content[state_index])
        #print("filtered_actions", filtered_actions)
        action_index = np.argmax(filtered_actions)
        return action_index



        # I'll think later about the epsilon

        # # Randomly generate a number between 0 and 1
        # random = uniform(0, 1)
        # # if random_int > greater than epsilon --> exploitation
        # if random <= epsilon:
        #     # Take the action with the highest value given a state
        #     # np.argmax can be useful here
        #     action_mask = self.state_space[state_index].valid_actions_mask
        #     action = np.argmax(self.content[state_index])
        # # else --> exploration
        # else:
        #     action = Action.get_random_action()
        #
        # return action
