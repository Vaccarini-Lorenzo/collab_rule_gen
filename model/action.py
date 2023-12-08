import itertools

from model.instanceRepository import InstanceRepository
from model.printable import Printable
from math import factorial
from random import randrange
from misc.helper import Helper


class Action(Printable):
    def __init__(self, method, instance):
        self.method = method
        self.instance = instance

    def __hash__(self):
        return hash((self.method, self.instance.name))

    def __eq__(self, other):
        return (self.method, self.instance.name) == (other.method, other.instance.name)

    def __ne__(self, other):
        return not (self == other)

    @staticmethod
    def get_action_space():
        action_space = {}
        instances = InstanceRepository.instance().get_all()
        index = 0
        for instance in instances:
            action_space[index] = Action("ADD", instance)
            index += 1
        for instance in instances:
            action_space[index] = Action("REMOVE", instance)
            index += 1
        return action_space

    @staticmethod
    def get_state_space():
        combination_map = {}
        instances = InstanceRepository.instance().get_all()
        instance_names = list(map(lambda instance: instance.name, instances))
        n = len(instance_names)
        # State zero is a state with zero instances
        combination_map[0] = ""
        index = 1
        for r in range(1, n + 1):
            for combination in itertools.combinations(instance_names, r):
                print(combination)
                sorted_state = Helper.sort_state(" ".join(combination))
                combination_map[index] = sorted_state
                index += 1

        return combination_map

    @staticmethod
    def get_state_space_cardinality():
        total_states = 0
        num_of_instances = len(InstanceRepository.instance().get_all())
        for r in range(1, num_of_instances + 1):
            combinations = factorial(num_of_instances) // (factorial(r) * factorial(num_of_instances - r))
            total_states += combinations
            print(f"Combinations of {r} elements: {combinations}")

        return total_states

    @staticmethod
    def get_action_space_cardinality():
        return 2 * len(InstanceRepository.instance().get_all())

    @staticmethod
    def get_random_action():
        return randrange(2 * len(InstanceRepository.instance().get_all()))
