import tensorflow as tf
import numpy as np
import socket
import struct
import math
from collections import deque

from ImmersiveTensorflowServer import ImmersiveTensorflowServer
from ImmersiveTensorflowServerConfig import ImmersiveTensorflowServerConfig
from DQN.DQNModel import DQNModel, DQNConfig

from ModelSkeleton import ModelSkeleton

class Observation():
  pass

class ImmersiveTensorflowReinforcementServer(ImmersiveTensorflowServer):
  def __init__(self, config : ImmersiveTensorflowServerConfig, model : ModelSkeleton):
    super().__init__()

    self.config = config
    self.model = model

    self.last_state = None
    self.last_action = 0

    self.observing = True
    self.observations = deque()
    self.observations_count = 0
    self.max_observation_count = 36000

    self.rewards = deque()
    self.rewards_count = 0
    self.max_rewards_count = 36000

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

    #placeholders = self.model.placeholders

    #inference_op = self.model.inference
    #train_op = self.model.training
    #loss_op = self.model.loss
    #eval_op = self.model.evaluation

    #1] Get Frame
    #2] If Reward : Append Reward
    #3] Append Observation (Last_State : last Frame, Last_Action : last Action, Reward, Current_State : Frame, ?)
    #4] If not observing : TRAIN
    #5] Update (State : Frame, Action : new Action)
    #6] If not observing : Reduce random action probability (until a minimum)

    while self.listening:
      recv_type, recv_frame, recv_actions, recv_reward = self.get_frame()
      self.observing = (recv_type == 0)
      self.listening = (recv_type != 2)

      self.register_reward(recv_reward)
      self.create_observation()

      if self.observing:
        last_screenshot
        last_action
        reward

        self.observations.append([recv_data])
        self.observations_count += 1

        if self.observationsCount > 36000:
          self.observing = False

        answer = "tmp".encode()

      else:
        inputs = recv_data[4:self.model.input_size + 4]
        inputs = np.fromstring(inputs, dtype=np.uint8)
        inputs = inputs.astype(np.float32)
        outputs = struct.unpack('%sf' % self.model.output_size, recv_data[-self.model.output_size * 4:])

        feed_dict = {
          self.model.placeholders[0] : inputs,
          self.model.placeholders[1] : outputs
          }

        inference = session.run(inference_op, feed_dict = feed_dict)
        answer = inference.tolist()[0]

        answer = struct.pack('%sf' % self.model.output_size, *answer)

      self.send_data(answer)


  def get_frame(self):
    recv_data, length = self.get_data()
    i = 4 # length size (int => 4 bytes)

    recv_type = recv_data[i : i + 4]
    recv_type = int.from_bytes(recv_type)
    if recv_type == 2:
      return recv_type, None, None, None
    i += 4 # recv_type size (int => 4 bytes)

    recv_frame = recv_data[i : i + self.model.input_size]
    i += self.model.input_size # pixel count size (1 pixel => 1 byte)

    recv_action = recv_data[i : i + 4]
    recv_action = int.from_bytes(recv_action)
    i += 4  # recv_action size (int => 4 bytes)

    recv_reward = recv_data[i : i + 4]
    recv_reward = struct.unpack('f', recv_reward)[0]
    # recv_reward size (float => 4 bytes)

    return recv_type, recv_frame, recv_action, recv_reward

  def register_reward(self, reward : float):
    if reward != 0:
      if self.rewards_count > self.max_rewards_count:
        self.rewards.popleft()
      self.rewards.append(reward)

  def create_observation(self, frame, reward):
    # First frame ever ?
    if self.last_state is None:
      self.last_state = frame
    # Building observation
    observation = (self.last_state, self.last_action, reward, frame)
    # Appending observation, if full then we pop the oldest one to make room for the new one
    if self.observations_count > self.max_observation_count:
      self.observations.popleft()
    self.observations.append(observation)
    # Update last state (to current)
    self.last_state = frame
    self.last_action = self.choose_next_action()

  def choose_next_action(self):
    # TO DO
    return 0

def main():
  # add model loading
  config = ImmersiveTensorflowServerConfig()
  model_config = DQNConfig("DQN/DQNConfig.ini")
  model = DQNModel(model_config, 480*270*3, 4)
  immersivetensorflowserver = ImmersiveTensorflowReinforcementServer(config, model)
  immersivetensorflowserver.run_model_training()

main()