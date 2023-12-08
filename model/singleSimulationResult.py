from misc.helper import Helper
from json import dumps

class SingleSimulationResult:

    def __init__(self, simulation_data):
        Helper.check_in_dict(simulation_data, "load", "failedRequestPercentage", "ninetyFivePercentileResponseTime")
        self.load = simulation_data["load"]
        self.fail_percentage = simulation_data["failedRequestPercentage"]
        self.response_time = simulation_data["ninetyFivePercentileResponseTime"]

    def get_values(self):
        return {
            "response_time": self.response_time,
            "fail_percentage": self.fail_percentage
        }