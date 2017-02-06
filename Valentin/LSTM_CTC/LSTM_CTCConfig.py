from configparser import SafeConfigParser

class LSTM_CTCConfig(object):
  def __init__(self, config_path = "model_config.ini"):
    config_parser = SafeConfigParser()
    config_parser.read(config_path)

    input_config = config_parser["Input"]
    self.input_max_time = int(input_config["input_max_time"])
    self.input_frame_size = int(input_config["input_frame_size"])

    output_config = config_parser["Output"]
    self.output_max_time = int(output_config["output_max_time"])
    self.output_frame_size = int(output_config["output_frame_size"])

    lstm_config = config_parser["LSTM"]
    self.cell_count = int(lstm_config["cell_count"])
    self.cell_size = int(lstm_config["cell_size"])

    fully_connected_config = config_parser["FullyConnected"]
    self.fully_connected_count = int(fully_connected_config["fully_connected_count"])
    self.fully_connected_size = int(fully_connected_config["fully_connected_size"])

    training_config = config_parser["Training"]
    self.learning_rate = float(training_config["learning_rate"])
