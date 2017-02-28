import tensorflow as tf
import numpy as np

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModelFnOps

from Berlin_utils.BerlinDatabase import BerlinDatabase

FEATURES = ["mfcc", "mfcc_lengths"]

def cldnn_model_fn(features, labels, mode, params):
  inputs = tf.cast(features["mfcc"], tf.float32)
  inputs_lengths = features["mfcc_lengths"]

  with tf.name_scope("Convolution"):
    input_layer = tf.reshape(inputs, shape = [-1, params["max_timesteps"], params["mfcc_features_count"], 1])

    conv_layer1 = tf.layers.conv2d(
      inputs = input_layer,
      filters = params["conv1_features_count"],
      kernel_size = [params["conv1_kernel_size1"], params["conv1_kernel_size2"]],
      padding = "same",
      activation = tf.nn.relu)

    pool_layer = tf.layers.max_pooling2d(
      inputs = conv_layer1,
      pool_size = [1, params["max_pooling_size"]],
      strides = [1, params["max_pooling_size"]])

    conv_layer2 = tf.layers.conv2d(
      inputs = pool_layer,
      filters = params["conv2_features_count"],
      kernel_size = [params["conv2_kernel_size1"], params["conv2_kernel_size2"]],
      padding = "same",
      activation = tf.nn.relu)

    pool_output_freq_size = int(params["mfcc_features_count"] / params["max_pooling_size"])

    conv_layer2_flat = tf.reshape(
      conv_layer2, 
      shape = [-1, params["max_timesteps"] * pool_output_freq_size * params["conv2_features_count"]]
      )

  with tf.name_scope("Dimension_reduction"):
    dim_reduction_layer = tf.layers.dense(
      inputs = conv_layer2_flat,
      units = params["max_timesteps"] *params["dimension_reduction_size"],
      activation = tf.nn.relu)

    dim_reduction_layer = tf.reshape(
      tensor = dim_reduction_layer,
      shape = [-1, params["max_timesteps"], params["dimension_reduction_size"]])

  with tf.name_scope("Concatenation"):
    concatenation_layer = tf.concat(
      values = [inputs, dim_reduction_layer],
      axis = 2)

  with tf.name_scope("Recurrent"):
    lstm_cell = tf.contrib.rnn.LSTMCell(
      num_units = params["lstm_units"],
      num_proj = params["lstm_projection"],
      activation = tf.tanh)

    lstm_cell = tf.contrib.rnn.MultiRNNCell(
      cells = [lstm_cell] * params["lstm_cell_count"])

    lstm_output, lstm_state = tf.nn.dynamic_rnn(
      cell = lstm_cell,
      inputs = concatenation_layer,
      sequence_length = inputs_lengths,
      dtype = tf.float32)

    lstm_output = tf.reshape(
      tensor = lstm_output,
      shape = [-1, params["max_timesteps"] * params["lstm_projection"]])

  with tf.name_scope("Fully_connected"):
    dense_layer1 = tf.layers.dense(
      inputs = lstm_output,
      units = params["fully_connected1_size"],
      activation = tf.nn.relu)

    dense_layer2 = tf.layers.dense(
      inputs = dense_layer1,
      units = params["fully_connected2_size"],
      activation = tf.nn.relu)

  dropout = tf.layers.dropout(
    inputs = dense_layer2,
    rate = 0.4,
    training = mode==learn.ModeKeys.TRAIN)

  with tf.name_scope("Logits"):
    logits_flat = tf.layers.dense(
      inputs = dropout,
      units = params["labels_class_count"]
      )

    logits = tf.reshape(
      logits_flat,
      shape = [-1, params["labels_class_count"]])

  predictions = \
  {
    "classes" : tf.argmax(input = logits, axis = 1),
    "probabilities" : tf.nn.softmax(logits, name = "softmax_tensor")
  }
    
  loss = None
  train_op = None
  eval_metric_ops = None

  if mode != learn.ModeKeys.INFER:
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf.one_hot(labels, 7))
    loss = tf.reduce_mean(cross_entropy)

  if mode == learn.ModeKeys.TRAIN:
    def exp_decay(learning_rate, global_step):
        return tf.train.natural_exp_decay(
            learning_rate=learning_rate, global_step=global_step,
            decay_steps=250, decay_rate=0.1)
    train_op = tf.contrib.layers.optimize_loss(
      loss = loss,
      global_step = tf.contrib.framework.get_global_step(),
      learning_rate = params["learning_rate"],
      optimizer = params["optimizer"],
      learning_rate_decay_fn = exp_decay,
      )

  if mode == learn.ModeKeys.EVAL:
    batch_size = int(logits.get_shape()[0])
    eval_metric_ops = {
      "accuracy" : tf.metrics.accuracy(
        labels = labels,
        predictions = tf.argmax(input = logits, axis = 1))
    }

  return ModelFnOps(mode = mode, predictions = predictions, loss = loss, train_op = train_op, eval_metric_ops = eval_metric_ops)

BERLIN_DATABASE = BerlinDatabase(r"C:\tmp\Berlin")
BERLIN_TRAIN_DATASET = BERLIN_DATABASE.next_batch(435)
BERLIN_TEST_DATASET = BERLIN_DATABASE.next_batch(100)

def berlin_input_fn_train_dataset():
  return berlin_input_fn(BERLIN_TRAIN_DATASET)

def berlin_input_fn_test_dataset():
  return berlin_input_fn(BERLIN_TEST_DATASET)

def berlin_input_fn(dataset):
  feat_count = len(FEATURES)
  batch_size = len(dataset)

  feature_cols = {FEATURES[k] : tf.constant(dataset[k])
                  for k in range(feat_count)}

  labels = tf.constant(dataset[feat_count])

  return feature_cols, labels

def main(unused_argv):
  parameters = \
  {
    "max_timesteps" : 36,
    "mfcc_features_count" : 40,

    "labels_class_count" : 7,

    "conv1_features_count" : 64,     # 256
    "conv1_kernel_size1" : 9,         # 9
    "conv1_kernel_size2" : 9,         # 9

    "max_pooling_size" : 3,           # 3

    "conv2_features_count" : 16,     # 256
    "conv2_kernel_size1" : 4,         # 4
    "conv2_kernel_size2" : 3,         # 3

    "dimension_reduction_size" : 32, # 256
    
    "lstm_units" : 832,               # 832
    "lstm_projection" : 256,          # 512
    "lstm_cell_count" : 2,            # 2

    "fully_connected1_size" : 128,   # 1024
    "fully_connected2_size" : 128,   # 1024

    "learning_rate" : 1e-2,
    "optimizer" : "SGD"
  }

  run_config = learn.RunConfig(gpu_memory_fraction = 0.8)

  cldnn_classifier = learn.Estimator(
    model_fn = cldnn_model_fn,
    model_dir = r"C:\tmp\berlin_cldnn",
    params = parameters,
    config = run_config)

  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors = tensors_to_log,
      every_n_iter = 50)

  for i in range(1000):
      print("Step ", i)
      cldnn_classifier.fit(
        input_fn = berlin_input_fn_train_dataset,
        steps = 250,
        monitors=[logging_hook])

    #Evaluate the model and print results
      eval_results = cldnn_classifier.evaluate(
        input_fn = berlin_input_fn_test_dataset,
        steps = 1)
      print(eval_results)

      BERLIN_TRAIN_DATASET = BERLIN_DATABASE.next_batch(435)
      BERLIN_TEST_DATASET = BERLIN_DATABASE.next_batch(100)

if __name__ == "__main__":
  tf.app.run()