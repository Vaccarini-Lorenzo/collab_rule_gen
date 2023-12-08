from misc.singleton import Singleton
from model.instance import Instance
from model.instanceRepository import InstanceRepository
import copy


@Singleton
class InstanceBuilder:
    def get_instance_by_name(self, instance_name):
        instance = InstanceRepository.instance().get_instance(instance_name)
        if instance is None:
            print("Error building instance: Instance name not in instance repository")
            print("Current repository:")
            print(InstanceRepository.instance().get_all())
            exit(1)
        return copy.deepcopy(instance)
