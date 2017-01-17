import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import functools
from SpeechDataUtils import SpeechDataUtils
from SpeechDataUtils import SpeechDataSet

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

class CLDNNModelOptions:
  def __init__(self, max_timesteps : int, mfcc_features_count : int, dictionary_size : int, max_output_sequence_length : int,
                learning_rate = 1e-4,
                conv_kernel_size = 5, conv_features_count = 32,
                max_pooling_size = 2,
                dimension_reduction_output_size = 256, time_reduction_factor = 1,
                lstm1_hidden_units_count = 512, lstm2_hidden_units_count = 512,
                fully_connected1_size = 256
               ):
    self.max_timesteps = max_timesteps
    self.mfcc_features_count = mfcc_features_count
    self.dictionary_size = dictionary_size
    self.max_output_sequence_length = max_output_sequence_length

    self.learning_rate = learning_rate

    self.conv_kernel_size = conv_kernel_size
    self.conv_features_count = conv_features_count

    self.max_pooling_size = max_pooling_size
    self.dimension_reduction_output_size = dimension_reduction_output_size
    self.time_reduction_factor = time_reduction_factor

    self.lstm1_hidden_units_count = lstm1_hidden_units_count
    self.lstm2_hidden_units_count = lstm2_hidden_units_count

    self.fully_connected1_size = fully_connected1_size


class CLDNNModel:
  def __init__(self, options : CLDNNModelOptions):
    self.options = options

    self.init_placeholders()
        
    self.input_size = int(self.options.max_timesteps)
    self.output_size = int(self.options.max_output_sequence_length)

    print("Inference : ", self.inference)
    print("Loss : ", self.loss)
    print("Training : ", self.training)
    print("Evaluation : ", self.evaluation)

  def init_placeholders(self):
    self.input_placeholder = tf.placeholder(tf.float32, [None, self.options.max_timesteps, self.options.mfcc_features_count], name="input__placeholder")
    self.input_lengths_placeholder = tf.placeholder(tf.int32, [None], name="input_lengths_placeholder")

    #self.output_placeholder = tf.sparse_placeholder(tf.int32, shape=[None, self.options.max_output_sequence_length, self.options.dictionary_size])
    self.output_lengths_placeholder = tf.placeholder(tf.int32, [None], name="output_lengths_placeholder")
    
    #self.output_placeholder = tf.placeholder(tf.int32, [None, self.options.max_output_sequence_length, self.options.dictionary_size], name="true_output_placeholder")
    self.output_placeholder_idx = tf.placeholder(tf.int64, name="true_output_placeholder_idx")
    self.output_placeholder_val = tf.placeholder(tf.int32, name="true_output_placeholder_val")
    self.output_placeholder_shape = tf.placeholder(tf.int64, name="true_output_placeholder_shape")

    self.sparse_output = tf.SparseTensor(self.output_placeholder_idx, self.output_placeholder_val, self.output_placeholder_shape)

  @define_scope
  def inference(self):
    input_as_2d_tensor = tf.reshape(self.input_placeholder, [-1, self.options.max_timesteps, self.options.mfcc_features_count, 1])

    # 1st Layer (Convolution)
    ## Weights & Bias
    with tf.name_scope("Convolution"):
      weights_conv_layer = CLDNNModel.weight_variable([self.options.conv_kernel_size, self.options.conv_kernel_size, 1, self.options.conv_features_count])
      bias_conv_layer = CLDNNModel.bias_variable([self.options.conv_features_count])
      ## Result
      conv_layer = CLDNNModel.conv2d(input_as_2d_tensor, weights_conv_layer) + bias_conv_layer
      relu_conv_layer = tf.nn.relu(conv_layer)

    # 2nd Layer (Max Pooling)
    with tf.name_scope("Max_pooling"):
      max_pool_layer = CLDNNModel.max_pool_1xN(relu_conv_layer, self.options.max_pooling_size)

    # 3rd Layer (Dimension reduction)
    ## Flattening (from 2D to 1D)
    with tf.name_scope("Dim_reduction"):
      convoluted_size = int(self.options.max_timesteps) * int(self.options.mfcc_features_count / self.options.max_pooling_size)
      flatten_size = convoluted_size * self.options.conv_features_count
      #flatten_size = int(convoluted_size * self.conv_features_count / self.options.max_timesteps)
      max_pool_layer_flatten = tf.reshape(max_pool_layer, [-1, flatten_size], name="Flatten_maxpool")
      ## Weights and Bias
      time_red_size = int(self.options.max_timesteps / self.options.time_reduction_factor)
      dim_red_size = time_red_size * self.options.dimension_reduction_output_size
      weights_dim_red_layer = CLDNNModel.weight_variable([flatten_size, dim_red_size])
      bias_dim_red_layer = CLDNNModel.bias_variable([dim_red_size])
      ## Result
      dim_red_layer = tf.matmul(max_pool_layer_flatten, weights_dim_red_layer) + bias_dim_red_layer

    # Input reduction (for memory issues :( )
    with tf.name_scope("Input_reduction"):
      flatten_input_size = self.options.max_timesteps * self.options.mfcc_features_count
      flatten_input_size_red = int(flatten_input_size / self.options.time_reduction_factor)
      flatten_input = tf.reshape(self.input_placeholder, [-1, flatten_input_size], name="flatten_input")
      
      weights = CLDNNModel.weight_variable([flatten_input_size, flatten_input_size_red])
      biaises = CLDNNModel.bias_variable([flatten_input_size_red])

      red_input = tf.matmul(flatten_input, weights) + biaises
      red_time = tf.cast(tf.ceil(self.input_lengths_placeholder / self.options.time_reduction_factor), tf.int32)

    # 4th Layer (Concatenation)
    with tf.name_scope("Concatenation"):
      concatenation_layer = tf.concat(1, [dim_red_layer, red_input])
      concatenation_layer_reshaped = tf.reshape(concatenation_layer, (-1, time_red_size, self.options.dimension_reduction_output_size + self.options.mfcc_features_count), name="reshape_timesteps_concat")

    # 5th Layer (LSTM 1)
    with tf.name_scope("LSTM1"):
      with tf.variable_scope("LSTMCell1"):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.options.lstm1_hidden_units_count)
        lstm1_output, lstm_state = tf.nn.dynamic_rnn(lstm_cell, concatenation_layer_reshaped, dtype=tf.float32, sequence_length = red_time)

    # 6th Layer (LSTM 2)
    with tf.name_scope("LSTM2"):
      with tf.variable_scope("LSTMCell2"):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.options.lstm2_hidden_units_count)
        lstm2_output, lstm_state = tf.nn.dynamic_rnn(lstm_cell, lstm1_output, dtype=tf.float32)

    lstm2_output_shape = lstm2_output.get_shape()
    lstm2_output_shape = [-1, int(lstm2_output_shape[1] * lstm2_output_shape[2])]
    lstm2_output_reshaped = tf.reshape(lstm2_output, lstm2_output_shape)

    # 7th Layer (Fully connected 1)
    with tf.name_scope("Fully_connected1"):
      weights = CLDNNModel.weight_variable([lstm2_output_shape[1], self.options.fully_connected1_size])
      biases = CLDNNModel.bias_variable([self.options.fully_connected1_size])

      fully_connected_layer1 = tf.matmul(lstm2_output_reshaped, weights) + biases

    # 7th Layer (Fully connected 2)
    with tf.name_scope("Fully_connected2"):
      weights = CLDNNModel.weight_variable([self.options.fully_connected1_size, self.output_size * self.options.dictionary_size])
      biases = CLDNNModel.bias_variable([self.output_size * self.options.dictionary_size])

      fully_connected_layer2 = tf.matmul(fully_connected_layer1, weights) + biases

    logits = tf.reshape(fully_connected_layer2, [-1, self.output_size , self.options.dictionary_size])
        
    return logits # Should be the 7th layer's ouput

  @define_scope
  def loss(self):
    inference_time_major = tf.transpose(self.inference, [1, 0, 2])
    #ctc = tf.nn.ctc_loss(self.inference, self.output_placeholder, self.output_lengths_placeholder, time_major = False)
    ctc = tf.nn.ctc_loss(inference_time_major, self.sparse_output, self.output_lengths_placeholder, time_major = True)
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.inference, self.output_placeholder))

    return tf.reduce_mean(ctc)

  @define_scope
  def training(self):
    tf.summary.scalar('loss', self.loss)

    optimizer = tf.train.GradientDescentOptimizer(self.options.learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(self.loss, global_step = global_step)

    return train_op
    
  @define_scope
  def evaluation(self):
    #dense_output = tf.sparse_to_dense(self.output_placeholder.indices, self.output_placeholder.shape, self.output_placeholder.values)
    dense_output = tf.sparse_to_dense(self.sparse_output.indices, self.sparse_output.shape, self.sparse_output.values)
    correct_prediction = tf.equal(tf.argmax(self.inference, 2), tf.argmax(dense_output, 2))
    return tf.cast(correct_prediction, tf.float32)
    
  @staticmethod
  def conv2d(inputTensor, weights):
    return tf.nn.conv2d(inputTensor, weights, strides=[1, 1, 1, 1], padding='SAME')

  @staticmethod
  def max_pool_1xN(inputTensor, max_pooling_size):
    return tf.nn.max_pool(inputTensor, ksize=[1, 1, max_pooling_size, 1], strides=[1, 1, max_pooling_size, 1], padding='SAME')

  @staticmethod
  def init_variable(shape, init_method='uniform', xavier_params = (None, None)):
    if init_method == 'zeros':
      return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
      return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
      (fan_in, fan_out) = xavier_params
      low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
      high = 4*np.sqrt(6.0/(fan_in + fan_out))
      return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    # Need for gaussian (for LSTM)

  @staticmethod
  def weight_variable(shape, init_method='uniform', xavier_params = (None, None)):
    return CLDNNModel.init_variable(shape, init_method, xavier_params)

  @staticmethod
  def bias_variable(shape, init_method='uniform', xavier_params = (None, None)):
    return CLDNNModel.init_variable(shape, init_method, xavier_params)

    #  indices = tf.constant(indices, tf.int64)
    #values = tf.constant(values, tf.int32)
    #dense_shape = tf.constant(dense_shape, tf.int64)
    #

def train_on_librispeech():
# For testing purpose
  BATCH_SIZE = 1
  MAX_OUTPUT_SEQUENCE_LENGTH = 22
  FEATURES_COUNT = 40
  TRAINING_ITERATION_COUNT = 125000

  summary_base_path = '/tmp/custom/CLDNN_CTC/logs/'
  if not os.path.exists(summary_base_path):
    os.mkdir(summary_base_path)

  data = SpeechDataUtils(librispeech_path = r"E:\tmp\LibriSpeech")
  train_data = data.train
  eval_data = data.eval

  eval_iterations_count = eval_data.batch_total_count
  dictionary_size = data.dictionary_size
  max_timesteps = data.bucket_size

  with tf.Graph().as_default():
    options = init_model_options(max_timesteps, FEATURES_COUNT, dictionary_size, MAX_OUTPUT_SEQUENCE_LENGTH)
    cldnn = CLDNNModel(options)

    train_op = cldnn.training
    loss_op = cldnn.loss
    eval_op = cldnn.evaluation

    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with tf.Session(config = session_config) as session:
      summary_writer = tf.summary.FileWriter(summary_base_path, session.graph)

      checkpoint = tf.train.get_checkpoint_state(summary_base_path)

      print("Initializing variables...")
      if checkpoint and checkpoint.model_checkpoint_path:
        print("Loading checkpoints from", checkpoint.model_checkpoint_path)
        saver.restore(session, checkpoint.model_checkpoint_path)
      else:
        print("No checkpoint found. Starting from scratch...")
        session.run(init)
      print("Variables initialized !")

      # TRAINING
      for i in tqdm(range(TRAINING_ITERATION_COUNT)):
        #batch_inputs, batch_lengths, batch_outputs = data.get_batch(BATCH_SIZE)
        batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = train_data.next_batch(BATCH_SIZE, one_hot = False)

        batch_outputs_indices, batch_outputs_values, batch_outputs_shape = batch_outputs

        feed_dict = {
          cldnn.input_placeholder : batch_inputs,
          cldnn.input_lengths_placeholder : batch_input_lengths,
          cldnn.output_placeholder_idx : batch_outputs_indices,
          cldnn.output_placeholder_val : batch_outputs_values,
          cldnn.output_placeholder_shape : batch_outputs_shape,
          #cldnn.output_placeholder : batch_outputs,
          cldnn.output_lengths_placeholder : batch_output_lengths
          }

        _, loss_value = session.run([train_op, loss_op], feed_dict = feed_dict)

        if i%100 == 0:
          summary_str = session.run(summary, feed_dict = feed_dict)
          summary_writer.add_summary(summary_str, i)
          summary_writer.flush()

        if i%5000 == 0 or (i + 1) == TRAINING_ITERATION_COUNT:
          checkpoint_file = summary_base_path + "model.ckpt"
          print("\nAt step", i, "loss = ", loss_value,'\n')
          saver.save(session, checkpoint_file, global_step = i)

      # EVALUTATION
      total_eval = np.zeros(MAX_OUTPUT_SEQUENCE_LENGTH)
      print("Testing model on", eval_iterations_count, "samples")
      for i in tqdm(range(eval_iterations_count)):
        batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = eval_data.next_batch(1, one_hot = False)

        batch_outputs_indices, batch_outputs_values, batch_outputs_shape = batch_outputs

        feed_dict = {
          cldnn.input_placeholder : batch_inputs,
          cldnn.input_lengths_placeholder : batch_input_lengths,
          cldnn.output_placeholder_idx : batch_outputs_indices,
          cldnn.output_placeholder_val : batch_outputs_values,
          cldnn.output_placeholder_shape : batch_outputs_shape,
          #cldnn.output_placeholder : batch_outputs,
          cldnn.output_lengths_placeholder : batch_output_lengths
          }
        session_eval = session.run(eval_op, feed_dict = feed_dict)
        total_eval += np.array(session_eval)
      total_eval /= eval_iterations_count
      print("\nAccuracy = " +str(total_eval * 100) + "%\n")
      input("\nPress to exit ...")

def use_librispeech_trained_model():
  BATCH_SIZE = 1
  MAX_OUTPUT_SEQUENCE_LENGTH = 22
  FEATURES_COUNT = 40
  TRAINING_ITERATION_COUNT = 125000

  summary_base_path = '/tmp/custom/CLDNN_CTC/logs/'

  data = SpeechDataUtils(librispeech_path = r"C:\tmp\LibriSpeech")
  train_data = data.train
  eval_data = data.eval
  eval_iterations_count = eval_data.batch_total_count
  dictionary_size = data.dictionary_size
  max_timesteps = data.bucket_size

  with tf.Graph().as_default():
    options = init_model_options(max_timesteps, FEATURES_COUNT, dictionary_size, MAX_OUTPUT_SEQUENCE_LENGTH)
    cldnn = CLDNNModel(options)

    train_op = cldnn.training
    loss_op = cldnn.loss
    eval_op = cldnn.evaluation

    saver = tf.train.Saver()

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with tf.Session(config = session_config) as session:

      print("Initializing variables...")
      checkpoint_file = summary_base_path
      checkpoint = tf.train.get_checkpoint_state(checkpoint_file)
      saver.restore(session, checkpoint.model_checkpoint_path)
      print("Variables initialized !")

      # EVALUTATION
      total_eval = np.zeros(MAX_OUTPUT_SEQUENCE_LENGTH)
      print("Testing model on", eval_iterations_count, "samples")
      for i in tqdm(range(eval_iterations_count)):
        batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = eval_data.next_batch(1, one_hot = False)

        batch_outputs_indices, batch_outputs_values, batch_outputs_shape = batch_outputs

        feed_dict = {
          cldnn.input_placeholder : batch_inputs,
          cldnn.input_lengths_placeholder : batch_input_lengths,
          cldnn.output_placeholder_idx : batch_outputs_indices,
          cldnn.output_placeholder_val : batch_outputs_values,
          cldnn.output_placeholder_shape : batch_outputs_shape,
          #cldnn.output_placeholder : batch_outputs,
          cldnn.output_lengths_placeholder : batch_output_lengths
          }

        session_eval = session.run(eval_op, feed_dict = feed_dict)
        total_eval += session_eval.reshape(MAX_OUTPUT_SEQUENCE_LENGTH,)
      total_eval /= eval_iterations_count
      print("\nAccuracy = " +str(total_eval * 100) + "%\n")
      input("\nPress to exit ...")
      

def init_model_options(max_timesteps : int, features_count : int, dictionary_size : int, max_output_sequence_length : int) -> CLDNNModelOptions:
  options = CLDNNModelOptions(max_timesteps, features_count, dictionary_size, max_output_sequence_length)
  options.conv_features_count = 32 #4
  options.dimension_reduction_output_size = 128  #128
  options.fully_connected1_size = 128 #16
  options.lstm1_hidden_units_count = 512
  options.lstm2_hidden_units_count = 512
  options.max_pooling_size = 4
  options.time_reduction_factor = 10
  return options

def main():
  train_on_librispeech()
  #use_librispeech_trained_model()

if __name__ == '__main__':
  main()
