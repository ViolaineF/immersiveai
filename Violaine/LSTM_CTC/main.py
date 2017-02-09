from LSTM_CTCServer import LSTM_CTCServer
from LSTM_CTCModel import LSTM_CTCModel
from LSTM_CTCConfig import LSTM_CTCModelConfig, LSTM_CTCServerConfig

from Timit_utils.TimitDatabase import TimitDatabase

def train():
    model_config = LSTM_CTCModelConfig("timit_model_config.ini")
    model = LSTM_CTCModel(model_config)

    timit_database = TimitDatabase(r"C:\tmp\TIMIT")
    train_data = timit_database.train_dataset

    model.train(timit_database, 0, 11)

def run_server():
    model_config = LSTM_CTCModelConfig("timit_model_config.ini")
    model = LSTM_CTCModel(model_config)

    server_config = LSTM_CTCServerConfig()
    server = LSTM_CTCServer(model, server_config)
    server.start_server()

def main():
    train()
    #run_server()


if __name__ == '__main__':
  main()