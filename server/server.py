from flask import Flask, request, jsonify
from threading import Timer
from model.instanceRepository import InstanceRepository
from q_learning.learningModule import LearningModule

class Server:
    def __init__(self, port_number):
        self.port_number = port_number
        self.app = Flask(__name__)
        self.register_routes()
        self.run()

    def register_routes(self):
        @self.app.route('/reportInstance', methods=['POST'])
        def report_instance():
            data = request.get_json()
            InstanceRepository.instance().manage_new_instance(data)
            response = {
                'message': 'Instance reported successfully'
            }

            return jsonify(response), 200

        @self.app.route('/start', methods=['GET'])
        def start_learner():
            response = {
                'message': 'Starting the rule generation'
            }
            t = Timer(2, LearningModule.instance().start)
            t.start()
            return jsonify(response), 200

        @self.app.route('/getInstances', methods=['GET'])
        def get_instances():
            response = InstanceRepository.instance().get_all(stringify=True)
            return response, 200

    def run(self):
        self.app.run(debug=True, port=self.port_number)
