import itertools

from misc.helper import Helper
from model.domain.instanceRepository import InstanceRepository
from model.env.action import Action


class State:
    def __init__(self, index, name, mask):
        self.index = index
        self.name = name
        self.valid_actions_mask = mask

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return not (self == other)

    @staticmethod
    def get_state_space():
        combination_map = {}
        instances = InstanceRepository.instance().get_all()
        instance_names = list(map(lambda instance: instance.name, instances))
        n = len(instance_names)
        mask = Action.get_valid_actions_mask("")
        # State zero is a state with zero instances
        combination_map[0] = State(0, "", mask)
        index = 1
        for r in range(1, n + 1):
            for combination in itertools.combinations(instance_names, r):
                sorted_state_name = Helper.sort_state_name(" ".join(combination))
                mask = Action.get_valid_actions_mask(sorted_state_name)
                combination_map[index] = State(index, sorted_state_name, mask)
                index += 1

        return combination_map

