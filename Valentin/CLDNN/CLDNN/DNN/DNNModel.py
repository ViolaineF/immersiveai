import numpy as np
import tensorflow as tf
from tqdm import tqdm

import functools

def define_scope(function):
  attribute = '_cache_' + function.__name__

  @property
  @functools.wraps(function)
  def decorator(self):
    if not hasattr(self, attribute):
      with tf.variable_scope(function.__name__):
        setattr(self, attribute, function(self))
    return getattr(self, attribute)

  return decorator

class DNNModel:
  def __init__(self, input_max_timesteps, mfcc_features_count, output_max_timesteps, dictionary_size):
    self.input_max_timesteps = input_max_timesteps
    self.mfcc_features_count = mfcc_features_count
    self.output_max_timesteps = output_max_timesteps
    self.dictionary_size = dictionary_size

    self.input_placeholder = tf.placeholder(tf.float32, [None, self.input_max_timesteps, self.mfcc_features_count], name="input__placeholder")
    self.output_placeholder = tf.placeholder(tf.int32, shape=[None, self.output_max_timesteps, self.dictionary_size], name="true_output_placeholder")

    print("Input placeholder : ", self.input_placeholder)
    print("Output placeholder : ", self.output_placeholder)
    print("Inference : ", self.inference)
    print("Loss : ", self.loss)
    print("Training : ", self.training)
    print("Evaluation : ", self.evaluation)

  @define_scope
  def inference(self):
    input_flatten = tf.reshape(self.input_placeholder, (-1, self.input_max_timesteps * self.mfcc_features_count))

    network_size = 256

    weights = DNNModel.weight_variable([self.input_max_timesteps * self.mfcc_features_count, network_size], "weights_first_layer")
    biases = DNNModel.biases_variable([network_size],"biases_first_layer")

    layer = tf.matmul(input_flatten, weights) + biases

    for i in range(5):
      weights = DNNModel.weight_variable([network_size, network_size], "weights_hidden_layer_" + str(i + 1))
      biases = DNNModel.biases_variable([network_size],"biases_hidden_layer_" + str(i + 1))

      layer = tf.matmul(layer, weights) + biases

    weights = DNNModel.weight_variable([network_size, self.output_max_timesteps * self.dictionary_size], "weights_last_layer")
    biases = DNNModel.biases_variable([self.output_max_timesteps * self.dictionary_size],"biases_last_layer")
    layer = tf.matmul(layer, weights) + biases

    logits = tf.reshape(layer, [-1, self.output_max_timesteps, self.dictionary_size])
    return logits

  @define_scope
  def loss(self):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.inference, self.output_placeholder)
    return tf.reduce_mean(cross_entropy)

  @define_scope
  def training(self):
    tf.summary.scalar('loss', self.loss)
    #optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
    optimizer = tf.train.MomentumOptimizer(1e-4, 0.9)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(self.loss, global_step = global_step)
    return train_op

  @define_scope
  def evaluation(self):
    correct_prediction = tf.equal(tf.argmax(self.inference, 2), tf.argmax(self.output_placeholder, 2))
    return tf.cast(correct_prediction, tf.float32)

  @staticmethod
  def weight_variable(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01, dtype = tf.float32), name = name)

  @staticmethod
  def biases_variable(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01, dtype = tf.float32), name = name)