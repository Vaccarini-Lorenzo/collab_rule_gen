from model.domain.instanceRepository import InstanceRepository
from q_learning.learningModule import LearningModule

if __name__ == "__main__":
    # Start server
    # api = Server(9876)
    instances = [
        {'instanceName': 'high-mem-high-cpu', 'instanceCPUs': 3, 'instanceMemory': 1000, 'meanResponseTime': 37,
         'ninetyFivePercentileResponseTime': 53, 'failedRequestPercentage': 0, 'load': 1},
        {'instanceName': 'high-mem-high-cpu', 'instanceCPUs': 3, 'instanceMemory': 1000, 'meanResponseTime': 55,
         'ninetyFivePercentileResponseTime': 389, 'failedRequestPercentage': 0, 'load': 21},
        {'instanceName': 'high-mem-high-cpu', 'instanceCPUs': 3, 'instanceMemory': 1000, 'meanResponseTime': 149,
         'ninetyFivePercentileResponseTime': 934, 'failedRequestPercentage': 0, 'load': 41},
        {'instanceName': 'high-mem-high-cpu', 'instanceCPUs': 3, 'instanceMemory': 1000, 'meanResponseTime': 306,
         'ninetyFivePercentileResponseTime': 1413, 'failedRequestPercentage': 0, 'load': 61},
        {'instanceName': 'high-mem-high-cpu', 'instanceCPUs': 3, 'instanceMemory': 1000, 'meanResponseTime': 514,
         'ninetyFivePercentileResponseTime': 1706, 'failedRequestPercentage': 0, 'load': 81},
        {'instanceName': 'high-mem-high-cpu', 'instanceCPUs': 3, 'instanceMemory': 1000, 'meanResponseTime': 753,
         'ninetyFivePercentileResponseTime': 1486, 'failedRequestPercentage': 0, 'load': 101},
        {'instanceName': 'high-mem-high-cpu', 'instanceCPUs': 3, 'instanceMemory': 1000, 'meanResponseTime': 999,
         'ninetyFivePercentileResponseTime': 1483, 'failedRequestPercentage': 0, 'load': 121},
        {'instanceName': 'high-mem-high-cpu', 'instanceCPUs': 3, 'instanceMemory': 1000, 'meanResponseTime': 1067,
         'ninetyFivePercentileResponseTime': 1448, 'failedRequestPercentage': 0, 'load': 141},
        {'instanceName': 'high-mem-high-cpu', 'instanceCPUs': 3, 'instanceMemory': 1000, 'meanResponseTime': 3149,
         'ninetyFivePercentileResponseTime': 5217, 'failedRequestPercentage': 0, 'load': 161},
        {'instanceName': 'high-mem-high-cpu', 'instanceCPUs': 3, 'instanceMemory': 1000, 'meanResponseTime': 8787,
         'ninetyFivePercentileResponseTime': 17932, 'failedRequestPercentage': 0, 'load': 181},
        # Skipping the next heavier simulations for high-mem-high-cpu
        {'instanceName': 'medium-mem-low-cpu', 'instanceCPUs': 1, 'instanceMemory': 800, 'meanResponseTime': 43,
         'ninetyFivePercentileResponseTime': 64, 'failedRequestPercentage': 0, 'load': 1},
        {'instanceName': 'medium-mem-low-cpu', 'instanceCPUs': 1, 'instanceMemory': 800, 'meanResponseTime': 35,
         'ninetyFivePercentileResponseTime': 44, 'failedRequestPercentage': 0, 'load': 11},
        {'instanceName': 'medium-mem-low-cpu', 'instanceCPUs': 1, 'instanceMemory': 800, 'meanResponseTime': 87,
         'ninetyFivePercentileResponseTime': 561, 'failedRequestPercentage': 0, 'load': 21},
        {'instanceName': 'medium-mem-low-cpu', 'instanceCPUs': 1, 'instanceMemory': 800, 'meanResponseTime': 2052,
         'ninetyFivePercentileResponseTime': 3791, 'failedRequestPercentage': 0, 'load': 31},
        {'instanceName': 'medium-mem-low-cpu', 'instanceCPUs': 1, 'instanceMemory': 800, 'meanResponseTime': 2463,
         'ninetyFivePercentileResponseTime': 3583, 'failedRequestPercentage': 0, 'load': 41},
        {'instanceName': 'medium-mem-low-cpu', 'instanceCPUs': 1, 'instanceMemory': 800, 'meanResponseTime': 3588,
         'ninetyFivePercentileResponseTime': 6051, 'failedRequestPercentage': 0, 'load': 51},
        {'instanceName': 'medium-mem-low-cpu', 'instanceCPUs': 1, 'instanceMemory': 800, 'meanResponseTime': 7793,
         'ninetyFivePercentileResponseTime': 12776, 'failedRequestPercentage': 0, 'load': 61},
        # Skipping the next heavier simulations for medium-mem-low-cpu
        {'instanceName': 'low-mem-medium-cpu', 'instanceCPUs': 2, 'instanceMemory': 400, 'meanResponseTime': 43,
         'ninetyFivePercentileResponseTime': 58, 'failedRequestPercentage': 0, 'load': 1},
        {'instanceName': 'low-mem-medium-cpu', 'instanceCPUs': 2, 'instanceMemory': 400, 'meanResponseTime': 34,
         'ninetyFivePercentileResponseTime': 41, 'failedRequestPercentage': 0, 'load': 11},
        {'instanceName': 'low-mem-medium-cpu', 'instanceCPUs': 2, 'instanceMemory': 400, 'meanResponseTime': 51,
         'ninetyFivePercentileResponseTime': 338, 'failedRequestPercentage': 0, 'load': 21},
        {'instanceName': 'low-mem-medium-cpu', 'instanceCPUs': 2, 'instanceMemory': 400, 'meanResponseTime': 114,
         'ninetyFivePercentileResponseTime': 737, 'failedRequestPercentage': 0, 'load': 31},
        {'instanceName': 'low-mem-medium-cpu', 'instanceCPUs': 2, 'instanceMemory': 400, 'meanResponseTime': 200,
         'ninetyFivePercentileResponseTime': 1099, 'failedRequestPercentage': 0, 'load': 41},
        {'instanceName': 'low-mem-medium-cpu', 'instanceCPUs': 2, 'instanceMemory': 400, 'meanResponseTime': 287,
         'ninetyFivePercentileResponseTime': 1234, 'failedRequestPercentage': 0, 'load': 51},
        {'instanceName': 'low-mem-medium-cpu', 'instanceCPUs': 2, 'instanceMemory': 400, 'meanResponseTime': 515,
         'ninetyFivePercentileResponseTime': 1530, 'failedRequestPercentage': 0, 'load': 61},
        {'instanceName': 'low-mem-medium-cpu', 'instanceCPUs': 2, 'instanceMemory': 400, 'meanResponseTime': 745,
         'ninetyFivePercentileResponseTime': 1879, 'failedRequestPercentage': 0, 'load': 71},
        {'instanceName': 'low-mem-medium-cpu', 'instanceCPUs': 2, 'instanceMemory': 400, 'meanResponseTime': 940,
         'ninetyFivePercentileResponseTime': 1773, 'failedRequestPercentage': 0, 'load': 81},
        {'instanceName': 'low-mem-medium-cpu', 'instanceCPUs': 2, 'instanceMemory': 400, 'meanResponseTime': 7019,
         'ninetyFivePercentileResponseTime': 16399, 'failedRequestPercentage': 0, 'load': 91},
        # Skipping the next heavier simulations for low-mem-medium-cpu
        {'instanceName': 'low-mem-low-cpu', 'instanceCPUs': 1, 'instanceMemory': 400, 'meanResponseTime': 40,
         'ninetyFivePercentileResponseTime': 56, 'failedRequestPercentage': 0, 'load': 1},
        {'instanceName': 'low-mem-low-cpu', 'instanceCPUs': 1, 'instanceMemory': 400, 'meanResponseTime': 34,
         'ninetyFivePercentileResponseTime': 43, 'failedRequestPercentage': 0, 'load': 11},
        {'instanceName': 'low-mem-low-cpu', 'instanceCPUs': 1, 'instanceMemory': 400, 'meanResponseTime': 92,
         'ninetyFivePercentileResponseTime': 606, 'failedRequestPercentage': 0, 'load': 21},
        {'instanceName': 'low-mem-low-cpu', 'instanceCPUs': 1, 'instanceMemory': 400, 'meanResponseTime': 951,
         'ninetyFivePercentileResponseTime': 1860, 'failedRequestPercentage': 0, 'load': 31},
        {'instanceName': 'low-mem-low-cpu', 'instanceCPUs': 1, 'instanceMemory': 400, 'meanResponseTime': 743,
         'ninetyFivePercentileResponseTime': 1710, 'failedRequestPercentage': 0, 'load': 41},
        {'instanceName': 'low-mem-low-cpu', 'instanceCPUs': 1, 'instanceMemory': 400, 'meanResponseTime': 5880,
         'ninetyFivePercentileResponseTime': 10724, 'failedRequestPercentage': 0, 'load': 51},
    ]

    for data in instances:
        InstanceRepository.instance().manage_new_instance(data)

    LearningModule.instance().start()
