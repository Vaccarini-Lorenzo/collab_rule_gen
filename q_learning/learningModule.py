import sys

import numpy as np
from tqdm import trange

from model.env.action import Action
from model.env.environment import Environment
from misc.singleton import Singleton
from model.domain.instanceRepository import InstanceRepository
from model.env.state import State
from q_learning.qLearningTable import QLearningTable
from config.learning_config import min_epsilon, max_epsilon, decay_rate, max_steps, learning_rate, gamma
from config.model_config import min_load, max_load, load_jump


@Singleton
class LearningModule:
    def __init__(self):
        self.qLearningTable = None

    def start(self):
        InstanceRepository.instance().consolidate_repository()
        curr_load = min_load

        while curr_load <= max_load:
            Environment.instance().set_num_of_req(curr_load)
            state_space = State.get_state_space()
            action_space = Action.get_action_space()
            self.qLearningTable = QLearningTable(state_space, action_space)
            feasible, max_reward, max_reward_configuration, max_reward_done, min_response_time, min_response_time_configuration = self.train(
                1000, min_epsilon, max_epsilon, decay_rate, max_steps, learning_rate, gamma)

            print("max_reward ", max_reward)
            print("max_reward_configuration ", max_reward_configuration.name)
            print("max_reward_done ", max_reward_done)
            print("min_response_time ", min_response_time)
            print("min_response_time_configuration ", min_response_time_configuration.name)

            print()
            print()

            if feasible:
                print("The objective is feasible.")
                done = False
                state = Environment.instance().reset()
                iteration_counter = 0
                while not done and iteration_counter < 20:
                    action_mask = self.qLearningTable.state_space[state.index].valid_actions_mask
                    filtered_actions = np.multiply(action_mask, self.qLearningTable.content[state.index])
                    action_index = np.argmax(filtered_actions)
                    state, reward, done = Environment.instance().execute_action(action_index)
                    iteration_counter += 1
                if not done:
                    print("Cycle detected...")
                print(state.name, state.average_response_time, done)
            else:
                print("The objective is not feasible")

            print()
            print()
            print()

            curr_load += load_jump

        #
        # print()
        # print("instances")
        # for instance in InstanceRepository.instance().get_all():
        #     print(instance.name)
        #     print(instance.performance_map)
        # print()
        #

    def train(self, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, learning_rate, gamma):
        feasible = False
        max_reward = -1000
        max_reward_configuration = None
        min_response_time = sys.maxsize
        min_response_time_configuration = None
        for episode in trange(n_training_episodes):
            # Reduce epsilon (because we need less and less exploration)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            # Reset the environment
            state = Environment.instance().reset()

            # repeat
            for step in range(max_steps):
                # Choose the action At using epsilon greedy policy
                action_index = self.qLearningTable.epsilon_greedy_policy(state.index, epsilon)
                # Take action At and observe Rt+1 and St+1
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, done = Environment.instance().execute_action(action_index)

                if reward > max_reward:
                    max_reward = reward
                    max_reward_configuration = new_state
                    max_reward_done = done
                if new_state.average_response_time < min_response_time:
                    min_response_time = new_state.average_response_time
                    min_response_time_configuration = new_state

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                self.qLearningTable.content[state.index][action_index] = self.qLearningTable.content[state.index][
                                                                             action_index] + learning_rate * (
                                                                                 reward + gamma * np.max(
                                                                             self.qLearningTable.content[
                                                                                 new_state.index]) -
                                                                                 self.qLearningTable.content[
                                                                                     state.index][action_index])

                # If done, finish the episode
                if done:
                    feasible = True
                    break

                # Our state is the new state
                state = new_state

        return feasible, max_reward, max_reward_configuration, max_reward_done, min_response_time, min_response_time_configuration
