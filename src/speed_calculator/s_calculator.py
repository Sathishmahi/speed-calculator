from utils import read_yaml
from const import PARAMS_YAML_PATH
from tracker import Tracking



class SpeedCalCulator:

    def __init__(self):

        self.params = read_yaml(PARAMS_YAML_PATH)["params"]
        self.config = read_yaml(CONFIG_YAML_PATH)

        self.tracking = Tracking(self.params["iou_threshold"])


    def variables_initializer(self):
        pass

    def process(self):
        pass

    def video_writer(self):
        pass
    
    def model_downloader(self):
        pass

    def line_drawer(self):
        pass

    def measure_speed(self):
        pass