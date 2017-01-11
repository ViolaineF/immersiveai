import tensorflow as tf
import numpy as np
from tqdm import tqdm
import socket
import struct
import math

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
      inputs = struct.unpack('%sf' % self.model.input_size, recv_data[4:(self.model.input_size + 1) * 4])
      outputs = struct.unpack('%sf' % self.model.output_size, recv_data[-self.model.output_size * 4:])

      feed_dict = {
        self.model.placeholders[0] : inputs,
        self.model.placeholders[1] : outputs
        }

      inference = session.run(self.model.inference, feed_dict = feed_dict)
      answer = inference.tolist()[0]

      answer = struct.pack('%sf' % self.model.output_size, *answer)

      self.send_data(answer)

      # session.run(train_op, feed_dict = {... data}=

  def get_data(self):
    recv_size = 32768
    data, _ = self.recv_ack(recv_size)
    length = int.from_bytes(data[:4], 'little')

    recv_count = math.ceil(length / recv_size)

    for i in range(1, recv_count):
      tmp, _ = self.recv_ack(recv_size)
      data += tmp

    return data, length

  def recv_ack(self, size):
    data, ip = self.sock.recvfrom(size)
    ack = "ACK".encode()
    self.send_data(ack)
    return data, ip

  def stop_listening(self):
    self.listening = False
    self.sock.close()

  def send_data(self, data : bytes):
    self.sock.sendto(data, (self.config.ip, self.config.send_port))

def main():
  # Add model loading
  config = ImmersiveTensorflowServerConfig()
  model_config = SimpleDNNConfig("SimpleDNN/config.ini")
  model = SimpleDNNModel(model_config, 480*270, 4)
  immersiveTensorflowServer = ImmersiveTensorflowServer(config, model)
  immersiveTensorflowServer.run_model_training()

main()