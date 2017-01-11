from configparser import SafeConfigParser

class SimpleDNNConfig(object):
    def __init__(self, config_path = "config.ini"):
      config_parser = SafeConfigParser()
      config_parser.read(config_path)

      layers_config = config_parser["Layers"]
      self.layer_count = int(layers_config["layer_count"])
      self.layers_size = int(layers_config["layers_size"])

      training_config = config_parser["Training"]
      self.learning_rate = float(training_config["learning_rate"])