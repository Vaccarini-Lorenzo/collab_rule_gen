from model.domain.singleSimulationResult import SingleSimulationResult


class SimulationResults:
    def __init__(self, single_simulation_result):
        self.results = [single_simulation_result]

    def update_data(self, new_simulation_data):
        self.results.append(SingleSimulationResult(new_simulation_data))

    # @staticmethod
    # def getTestSimulationResult():
    #     data = {
    #         "0-20": {"response_time": 20, "fail_percentage": 0},
    #         "40-50": {"response_time": 40, "fail_percentage": 1},
    #         "60-70": {"response_time": 60, "fail_percentage": 2},
    #         "80-90": {"response_time": 80, "fail_percentage": 3},
    #     }
    #     return SimulationResults(data)

    @staticmethod
    def from_json(json_data):
        single_simulation_result = SingleSimulationResult(json_data)
        return SimulationResults(single_simulation_result)
