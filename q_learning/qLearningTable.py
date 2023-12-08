import sys
from random import uniform
from tqdm.notebook import trange
import numpy as np
import pandas as pd
from config import learning_config

from model.action import Action


class QLearningTable:
    def __init__(self, state_space_cardinality, action_space_cardinality):
        self.content = np.zeros((state_space_cardinality, action_space_cardinality))

    def epsilon_greedy_policy(self, state_index, epsilon):
        # Randomly generate a number between 0 and 1
        random = uniform(0, 1)
        # if random_int > greater than epsilon --> exploitation
        if random > epsilon:
            # Take the action with the highest value given a state
            # np.argmax can be useful here
            action = np.argmax(self.content[state_index])
        # else --> exploration
        else:
            action = Action.get_random_action()

        return action
