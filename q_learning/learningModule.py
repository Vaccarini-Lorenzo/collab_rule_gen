import numpy as np
from tqdm import trange

from model.env.action import Action
from model.env.environment import Environment
from misc.singleton import Singleton
from model.domain.instanceRepository import InstanceRepository
from model.env.state import State
from q_learning.qLearningTable import QLearningTable
from config.learning_config import min_epsilon, max_epsilon, decay_rate, max_steps, learning_rate, gamma

@Singleton
class LearningModule:
    # Potentially pass arguments
    def __init__(self):
        self.qLearningTable = None

    def start(self):
        InstanceRepository.instance().consolidate_repository()
        Environment.instance().set_num_of_req(400)
        state_space = State.get_state_space()
        action_space = Action.get_action_space()
        self.qLearningTable = QLearningTable(state_space, action_space)
        #
        # print()
        # print("instances")
        # for instance in InstanceRepository.instance().get_all():
        #     print(instance.name)
        #     print(instance.performance_map)
        # print()
        #
        # print()
        # print("state space")
        # for state in list(state_space.items()):
        #     print(f"{state[1].index}: {state[1].name}")
        #     print(state[1].valid_actions_mask)
        #     print()
        #
        # for action in list(action_space.items()):
        #     print(f"{action[0]}: {action[1].method} {action[1].instance.name}")
        #
        #
        # new_state, reward, done = Environment.instance().execute_action(0)
        # print("Reward: ", reward)
        # print("Done: ", done)
        #
        # new_state, reward, done = Environment.instance().execute_action(1)
        # print("Reward: ", reward)
        # print("Done: ", done)
        #
        # new_state, reward, done = Environment.instance().execute_action(2)
        # print("Reward: ", reward)
        # print("Done: ", done)


        self.train(10000, min_epsilon, max_epsilon, decay_rate, max_steps, learning_rate, gamma)












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
            state = Environment.instance().reset()

            # repeat
            for step in range(max_steps):
                # Choose the action At using epsilon greedy policy
                action_index = self.qLearningTable.epsilon_greedy_policy(state.index, epsilon)
                # Take action At and observe Rt+1 and St+1
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, done = Environment.instance().execute_action(action_index)
                print("Reward: ", reward)
                print("Done: ", done)

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                self.qLearningTable.content[state.index][action_index] = self.qLearningTable.content[state.index][action_index] + learning_rate * (
                        reward + gamma * np.max(self.qLearningTable.content[new_state.index]) - self.qLearningTable.content[state.index][action_index])

                # If done, finish the episode
                if done:
                    break

                # Our state is the new state
                state = new_state
