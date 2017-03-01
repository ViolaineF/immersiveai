from LSTM_CTCServer import LSTM_CTCServer
from LSTM_CTCModel import LSTM_CTCModel
from LSTM_CTCConfig import LSTM_CTCModelConfig, LSTM_CTCServerConfig

from Berlin_utils.BerlinDatabase import BerlinDatabase

def train():
    model_config = LSTM_CTCModelConfig("berlin_model_config.ini")
    model = LSTM_CTCModel(model_config)

    berlin_database = BerlinDatabase(r"C:\tmp\Berlin")
    train_data = berlin_database.train_dataset

    model.train(berlin_database, 0, 11)

def run_server():
    model_config = LSTM_CTCModelConfig("berlin_model_config.ini")
    model = LSTM_CTCModel(model_config)

    server_config = LSTM_CTCServerConfig()
    server = LSTM_CTCServer(model, server_config)
    server.start_server()

def main():
    train()
    #run_server()


if __name__ == '__main__':
  main()