import tensorflow as tf
import numpy as np
import socket
import struct
import math

from ImmersiveTensorflowServer import ImmersiveTensorflowServer
from ImmersiveTensorflowServerConfig import ImmersiveTensorflowServerConfig
from ModelSkeleton import ModelSkeleton
#from SimpleDNN import SimpleDNNModel, SimpleDNNConfig
from SimpleDNN.SimpleDNNConfig import SimpleDNNConfig
from SimpleDNN.SimpleDNNModel import SimpleDNNModel

class ImmersiveTensorflowSupervisedServer(ImmersiveTensorflowServer):
  def __init__(self, config : ImmersiveTensorflowServerConfig, model : ModelSkeleton):
    super().__init__()

    self.config = config
    self.model = model

  def run_model_training(self):
    session_config = self.get_session_config()
    with tf.Session(config = session_config) as session:
      self.init_model(session)
      self.start_listen_training(session)

  def init_model(self, session):
    init = tf.global_variables_initializer()
    print("Initializing variables...")
    session.run(init)
    print("Variables initialized !")

  def get_session_config(self):
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = self.config.gpu_allow_growth

    return session_config

  def start_listen_training(self, session : tf.Session):
    if self.listening:
      print("Already listening.")
      return

    self.listening = True
    self.sock.bind((self.config.ip, self.config.listen_port))
    #self.sock.setblocking(False);

    placeholders = self.model.placeholders

        #inference_op = self.model.inference
    #train_op = self.model.training
    #loss_op = self.model.loss
    #eval_op = self.model.evaluation

    #observing = True
    #observations = []
    #observationsCount = 0

    while self.listening:
      recv_data, length = self.get_data()

      #if observing:
      #  last_screenshot
      #  last_action
      #  reward


      #  observations.append([recv_data])
      #  observationsCount += 1

      #  if observationsCount > 36000:
      #    observing = False

      #else:
      inputs = recv_data[4:self.model.input_size + 4]
      inputs = np.fromstring(inputs, dtype=np.uint8)
      inputs = inputs.astype(np.float32)
      outputs = struct.unpack('%sf' % self.model.output_size, recv_data[-self.model.output_size * 4:])

      feed_dict = {
        self.model.placeholders[0] : inputs,
        self.model.placeholders[1] : outputs
        }

      inference = session.run(self.model.inference, feed_dict = feed_dict)
      answer = inference.tolist()[0]

      answer = struct.pack('%sf' % self.model.output_size, *answer)
      self.send_data(answer)

def main():
  # Add model loading
  config = ImmersiveTensorflowServerConfig()
  model_config = SimpleDNNConfig("SimpleDNN/config.ini")
  model = SimpleDNNModel(model_config, 480*270*3, 4)
  immersiveTensorflowServer = ImmersiveTensorflowSupervisedServer(config, model)
  immersiveTensorflowServer.run_model_training()

main()