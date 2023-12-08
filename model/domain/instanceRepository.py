from model.domain.instance import Instance
from misc.singleton import Singleton


# This singleton is called when a new instance finishes its simulation.
# The instance will get added to the registry


@Singleton
class InstanceRepository:
    def __init__(self):
        # <name, instance>
        self.registry = dict()

    def manage_new_instance(self, json_data):
        new_instance = Instance.from_json(json_data)
        if new_instance.name in self.registry:
            instance = self.registry[new_instance.name]
            instance.update_simulation_data(json_data)
        else:
            self.registry[new_instance.name] = new_instance


    def consolidate_repository(self):
        for instance in self.registry.values():
            instance.perform_regression()
            instance.build_performance_map()


    def get_all(self, stringify=False):
        if not stringify:
            return list(self.registry.values())
        stringified = ""
        for item in self.registry.items():
            instance_name = item[0]
            instance = item[1].to_json()
            stringified += f"[{instance_name}: {instance}]"
        return stringified

    def get_instance(self, instance_name):
        return self.registry.get(instance_name)
