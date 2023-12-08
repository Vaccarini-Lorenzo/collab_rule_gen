import config.model_config
from config import model_config
from misc.printable import Printable
from model.domain.simulationResults import SimulationResults
from misc.helper import Helper
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np



class Instance(Printable):
    def __init__(self, name, CPUs, memory, simulation_results):
        self.name = name
        self.CPUs = CPUs
        self.memory = memory
        self.simulation_results = simulation_results
        self.cost = CPUs * model_config.instance_cost["CPU_cost"] + memory * model_config.instance_cost[
            "memory_cost_MB"]
        self.current_load = 0
        self.performance_map = dict()
        self.regression_model = None

    @staticmethod
    def from_json(json_data):
        Helper.check_in_dict(json_data, "instanceName", "instanceCPUs", "instanceMemory")
        return Instance(json_data["instanceName"], json_data["instanceCPUs"], json_data["instanceMemory"],
                        SimulationResults.from_json(json_data))

    def perform_regression(self, degree=2):
        loads = np.array([data.load for data in self.simulation_results.results]).reshape(-1, 1)
        response_times = np.array([data.response_time for data in self.simulation_results.results])
        failed_percentages = np.array([data.fail_percentage for data in self.simulation_results.results])

        # Creating a polynomial regression model
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

        # Fitting the model
        model.fit(loads, np.column_stack((response_times, failed_percentages)))

        # Returning the trained model
        self.regression_model = model


    def build_performance_map(self):
        load_range = np.arange(0, config.model_config.max_req_per_instance + 1).reshape(-1, 1)
        predictions = self.regression_model.predict(load_range)
        predictions = np.clip(predictions, 0, None)
        # predictions is in the form [[response_time_0, fail_0] ... ]
        for load, prediction in enumerate(predictions):
            self.performance_map[load] = {
                "response_time": prediction[0] if prediction[0] >= 0 else 0,
                "fail_percentage": prediction[1] if prediction[1] <= 100 else 100
            }

    def get_current_response_time(self):
        return self.performance_map[self.current_load]["response_time"]

    def get_current_performance_value(self):
        if self.current_load >= config.model_config.max_req_per_instance:
            current_performance = self.performance_map[config.model_config.max_req_per_instance]
        else:
            current_performance = self.performance_map[self.current_load]
        latency_cost = config.model_config.latency_penalize * max(0, current_performance[
            "response_time"] - config.model_config.acceptable_latency)
        fail_cost = config.model_config.fail_penalize * current_performance["fail_percentage"] * self.current_load
        return latency_cost + fail_cost

    def update_simulation_data(self, new_simulation_data):
        self.simulation_results.update_data(new_simulation_data)

    def update_current_requests(self, value):
        self.simulation_results = value
