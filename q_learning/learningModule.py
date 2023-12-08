import sys

import numpy as np
from tqdm import trange

from config import learning_config
from model.action import Action
from model.environment import Environment
from misc.singleton import Singleton
from model.instanceRepository import InstanceRepository
from q_learning.qLearningTable import QLearningTable
from config.learning_config import min_epsilon, max_epsilon, decay_rate, max_steps, learning_rate, gamma

@Singleton
class LearningModule:
    # Potentially pass arguments
    def __init__(self):
        print("Init the QLearner")
        self.qLearningTable = None

    def start(self):
        print("start")
        InstanceRepository.instance().consolidate_repository()
        Environment.instance().set_num_of_req(200)
        state_space = Action.get_state_space()
        action_space = Action.get_action_space()
        self.qLearningTable = QLearningTable(len(state_space), len(action_space))

        self.train(100, min_epsilon, max_epsilon, decay_rate, max_steps, learning_rate, gamma)

        # for action in list(action_space.items()):
        #     print(f"{action[0]}: {action[1].method} {action[1].instance.name}")
        #
        # Environment.instance().execute_action(0)
        # Environment.instance().execute_action(2)
        # Environment.instance().execute_action(4)

        # action_space = Action.get_action_space(Environment.instance().instances)
        # instance = None
        # for action in action_space:
        #     print(action.instance.name)
        #     instance = action.instance
        #
        # Environment.instance().execute_action(Action("REMOVE", instance))

    def train(self, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, learning_rate, gamma):
        for episode in trange(n_training_episodes):
            # Reduce epsilon (because we need less and less exploration)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            # Reset the environment
            state_index = Environment.instance().reset()

            # repeat
            for step in range(max_steps):
                # Choose the action At using epsilon greedy policy
                action_index = self.qLearningTable.epsilon_greedy_policy(state_index, epsilon)
                # Take action At and observe Rt+1 and St+1
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state_index, reward, done = Environment.instance().execute_action(action_index)
                print("Reward: ", reward)
                print("Done: ", done)

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                self.qLearningTable.content[state_index][action_index] = self.qLearningTable.content[state_index][action_index] + learning_rate * (
                        reward + gamma * np.max(self.qLearningTable.content[new_state_index]) - self.qLearningTable.content[state_index][action_index])

                # If done, finish the episode
                if done:
                    break

                # Our state is the new state
                state_index = new_state_index
