import tensorflow as tf
import numpy as np
from tqdm import tqdm
import socket

from ImmersiveTensorflowServerConfig import ImmersiveTensorflowServerConfig
from ModelSkeleton import ModelSkeleton
#from SimpleDNN import SimpleDNNModel, SimpleDNNConfig
from SimpleDNN.SimpleDNNConfig import SimpleDNNConfig
from SimpleDNN.SimpleDNNModel import SimpleDNNModel

class ImmersiveTensorflowServer():
  def __init__(self, config : ImmersiveTensorflowServerConfig, model : ModelSkeleton):
    self.config = config
    self.model = model

  #def __init__(self, ip, send_port, listen_port):
    #self.ip = ip
    #self.send_port = send_port
    #self.listen_port = listen_port

    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.listening = False

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

  def start_listen_training(self, session):
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

    while self.listening:
      print(self.get_data())

      # session.run(train_op, feed_dict = {... data}=

  def get_data(self):
    data = None
    done = False

    while not done:
      data, _ = self.sock.recvfrom(1024)
      #if get marker done:
      done = True

    return data

  def stop_listening(self):
    self.listening = False
    self.sock.close()

  def send_data(self, data : list):
    self.sock.sendto(data, (self.config.ip, self.config.send_port))

def main():
  config = ImmersiveTensorflowServerConfig()
  model_config = SimpleDNNConfig("SimpleDNN/config.ini")
  model = SimpleDNNModel(model_config, 20, 2)
  immersiveTensorflowServer = ImmersiveTensorflowServer(config, model)
  immersiveTensorflowServer.run_model_training()

main()