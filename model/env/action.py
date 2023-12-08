import numpy as np

from model.domain.instanceRepository import InstanceRepository
from misc.printable import Printable
from random import randrange
from config.model_config import max_number_same_instance, max_number_instances


class Action(Printable):
    def __init__(self, index, method, instance):
        self.index = index
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
            action_space[index] = Action(index, "ADD", instance)
            index += 1
        for instance in instances:
            action_space[index] = Action(index, "REMOVE", instance)
            index += 1
        return action_space

    @staticmethod
    def get_random_action():
        return randrange(2 * len(InstanceRepository.instance().get_all()))

    @staticmethod
    def get_valid_actions_mask(state_name):
        instances = InstanceRepository.instance().get_all()

        if state_name == "":
            mask = np.empty(len(instances) * 2)
            mask.fill(1)
            for i in range(len(instances), len(instances) * 2):
                mask[i] = -1000
            return mask

        mask = np.empty(len(instances) * 2)
        mask.fill(1)
        state_instance_names = state_name.split(" ")
        index = 0
        # Can't allocate # of same instances > max_number_same_instance
        for instance in instances:
            if state_instance_names.count(instance.name) >= max_number_same_instance:
                mask[index] = -1000
            if len(state_instance_names) >= max_number_instances:
                mask[index] = -1000
            index += 1
        # Can't remove instances that have not been allocated yet
        for instance in instances:
            if instance.name not in state_instance_names:
                mask[index] = -1000
            index += 1


        return mask





