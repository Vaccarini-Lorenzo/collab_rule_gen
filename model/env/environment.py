import sys

from misc.singleton import Singleton
from model.env.action import Action
from misc.helper import Helper
from config.model_config import acceptable_latency
import re

from model.env.state import State


@Singleton
class Environment:
    def __init__(self):
        # self.instances = [InstanceBuilder.instance().get_instance_by_name("low-mem-low-cpu")]
        self.instances = []
        self.n_requests = 0
        self.state_space = State.get_state_space()
        self.curr_state = self.state_space.get(0)
        self.action_space = Action.get_action_space()
        # self.instances[0].update_current_requests(n_requests)

    def set_num_of_req(self, n_requests):
        self.n_requests = n_requests

    def reset(self):
        self.instances = []
        self.curr_state = self.state_space.get(0)
        return self.curr_state

    def execute_action(self, action_index):
        # new_state_index, reward, done
        try:
            self.update_state(action_index)
        except ValueError:
            print("Trying to reach an illegal state. Cost = ", sys.maxsize)
            return self.curr_state, -1, False
        if len(self.instances) == 0:
            return self.curr_state, -1, False
        self.distribute_load()
        done = self.check_done()
        reward = self.compute_reward()
        return self.curr_state, reward, done

    def update_state(self, action_index):
        action = self.action_space.get(action_index)
        print("")
        print("Executing action: " + action.method + " " + action.instance.name)
        print("Current state: " + self.curr_state.name)
        if action.method == "ADD":
            self.instances.append(action.instance)
            new_state_name = action.instance.name if self.curr_state.name == "" else self.curr_state.name + " " + action.instance.name
            new_state_name = Helper.sort_state_name(new_state_name)
            self.curr_state = self.get_state_by_name(new_state_name)
        elif action.method == "REMOVE":
            self.instances.remove(action.instance)
            new_state_name = re.sub(rf'{action.instance.name}', "", self.curr_state.name)
            new_state_name = Helper.collapse_whitespace(new_state_name)
            new_state_name = Helper.sort_state_name(new_state_name)
            self.curr_state = self.get_state_by_name(new_state_name)
        print("update state =", self.curr_state.name)


    def get_state_by_name(self, new_state_name):
        print("looking for state " + new_state_name)
        for state in list(self.state_space.items()):
            if state[1].name == new_state_name:
                return state[1]

        print("[WARN]: State not found")
        raise ValueError

    def get_index_of_state(self, curr_state):
        if curr_state not in list(self.state_space.values()):
            print("[WARN]: State not found")
            raise ValueError
        for state in self.state_space.items():
            if curr_state == state[1]:
                return state[0]

    def check_done(self):
        average_response_time = 0
        for instance in self.instances:
            instance_load_weight = instance.current_load / self.n_requests
            instance_current_response_time = instance.get_current_response_time()
            average_response_time += instance_load_weight * instance_current_response_time
        print("average_response_time: ", average_response_time)
        return average_response_time <= acceptable_latency


    def distribute_load(self):
        # reset load
        for instance in self.instances:
            instance.current_load = 0

        for concurrent_req in range(0, self.n_requests):
            best_instance = self.get_best_instance()
            best_instance.current_load += 1

        print()
        print("distributed load:")
        for instance in self.instances:
            print(f"{instance.name}: {instance.current_load} --- latency: {instance.get_current_response_time()} --- value: {instance.get_current_performance_value()}")

        print()


    def get_best_instance(self):
        best_instance = self.instances[0]
        for instance in self.instances:
            if instance.get_current_performance_value() < best_instance.get_current_performance_value():
                best_instance = instance
        return best_instance

    def compute_reward(self):
        cost = 0
        if len(self.instances) == 0:
            return -1

        for instance in self.instances:
            # print("resources cost: ", instance.cost)
            cost += instance.cost
            # print("performance cost: ", instance.get_current_performance_value())
            cost += instance.get_current_performance_value()
        return 1 / cost
