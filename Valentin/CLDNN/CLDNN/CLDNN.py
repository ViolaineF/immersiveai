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

class CLDNNModel:
  def __init__(self,
                input_placeholder, output_placeholder,
                input_width : int, input_height : int,
                dictionary_size : int,
                learning_rate = 1e-4,
                conv_kernel_size = 5, conv_features_count = 32,
                dimension_reduction_output_size = 256,
                lstm1_hidden_units_count = 512, lstm2_hidden_units_count = 512,
                fully_connected1_size = 256,
                lookup_table_size = 64
                ):
    self.input_placeholder = input_placeholder
    self.output_placeholder = output_placeholder
        
    self.input_size = int(input_placeholder.get_shape()[1])
    self.output_size = int(output_placeholder.get_shape()[1])

    self.input_width = input_width
    self.input_height = input_height

    self.learning_rate = learning_rate

    self.conv_kernel_size = conv_kernel_size
    self.conv_features_count = conv_features_count

    self.dimension_reduction_output_size = dimension_reduction_output_size

    self.lstm1_hidden_units_count = lstm1_hidden_units_count
    self.lstm2_hidden_units_count = lstm2_hidden_units_count

    self.dictionary_size = dictionary_size
    self.fully_connected1_size = fully_connected1_size

    self.lookup_table_size = lookup_table_size

    print("Inference : ", self.inference)
    print("Loss : ", self.loss)
    print("Training : ", self.training)
    print("Evaluation : ", self.evaluation)

  @define_scope
  def inference(self):
    input_as_2d_tensor = tf.reshape(self.input_placeholder, [-1, self.input_width, self.input_height, 1])

    # 1st Layer (Convolution)
    ## Weights & Bias
    with tf.name_scope("Convolution"):
      weights_conv_layer = CLDNNModel.weight_variable([self.conv_kernel_size, self.conv_kernel_size, 1, self.conv_features_count])
      bias_conv_layer = CLDNNModel.bias_variable([self.conv_features_count])
      ## Result
      conv_layer = CLDNNModel.conv2d(input_as_2d_tensor, weights_conv_layer) + bias_conv_layer
      relu_conv_layer = tf.nn.relu(conv_layer)

    # 2nd Layer (Max Pooling)
    with tf.name_scope("Max_pooling"):
      max_pool_layer = CLDNNModel.max_pool_1x2(relu_conv_layer)

    # 3rd Layer (Dimension reduction)
    ## Flattening (from 2D to 1D)
    with tf.name_scope("Dim_reduction"):
      convoluted_size = int(self.input_width / 2) * int(self.input_height)
      flatten_size = convoluted_size * self.conv_features_count
      #flatten_size = int(convoluted_size * self.conv_features_count / self.input_width)
      max_pool_layer_flatten = tf.reshape(max_pool_layer, [-1, flatten_size])
      ## Weights and Bias
      weights_dim_red_layer = CLDNNModel.weight_variable([flatten_size, self.input_width * self.dimension_reduction_output_size])
      bias_dim_red_layer = CLDNNModel.bias_variable([self.input_width * self.dimension_reduction_output_size])
      ## Result
      dim_red_layer = tf.matmul(max_pool_layer_flatten, weights_dim_red_layer) + bias_dim_red_layer
      

    # 4th Layer (Concatenation)
    with tf.name_scope("Concatenation"):
      concatenation_layer = tf.concat(1, [dim_red_layer, self.input_placeholder])
      concatenation_layer_reshaped = tf.reshape(concatenation_layer, (-1, self.input_width, self.dimension_reduction_output_size + self.input_height))

    # 5th Layer (LSTM 1)
    with tf.name_scope("LSTM1"):
      with tf.variable_scope("LSTMCell1"):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm1_hidden_units_count)
        lstm1_output, lstm_state = tf.nn.dynamic_rnn(lstm_cell, concatenation_layer_reshaped, dtype=tf.float32)

    # 6th Layer (LSTM 2)
    with tf.name_scope("LSTM2"):
      with tf.variable_scope("LSTMCell2"):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm2_hidden_units_count)
        lstm2_output, lstm_state = tf.nn.dynamic_rnn(lstm_cell, lstm1_output, dtype=tf.float32)

    lstm2_output_shape = lstm2_output.get_shape()
    lstm2_output_shape = [-1, int(lstm2_output_shape[1] * lstm2_output_shape[2])]
    lstm2_output_reshaped = tf.reshape(lstm2_output, lstm2_output_shape)

    # 7th Layer (Fully connected 1)
    with tf.name_scope("Fully_connected1"):
      weights = CLDNNModel.weight_variable([lstm2_output_shape[1], self.fully_connected1_size])
      biases = CLDNNModel.bias_variable([self.fully_connected1_size])

      fully_connected_layer1 = tf.matmul(lstm2_output_reshaped, weights) + biases

    # 7th Layer (Fully connected 2)
    with tf.name_scope("Fully_connected2"):
      weights = CLDNNModel.weight_variable([self.fully_connected1_size, self.output_size])
      biases = CLDNNModel.bias_variable([self.output_size])

      fully_connected_layer2 = tf.matmul(fully_connected_layer1, weights) + biases
        
    return fully_connected_layer2 # Should be the 7th layer's ouput

  @define_scope
  def loss(self):
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.output_placeholder))
    embedding_map = tf.Variable(tf.truncated_normal([self.dictionary_size, self.lookup_table_size], stddev = 0.1), name = "embedding_map")
    seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.output_placeholder)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.inference, seq_embeddings))

    return cross_entropy

  @define_scope
  def training(self):
    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    return optimizer.minimize(self.loss)
    
  @define_scope
  def evaluation(self):
    correct_prediction = tf.equal(tf.argmax(self.inference, 1), tf.argmax(self.output_placeholder, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
  @staticmethod
  def conv2d(inputTensor, weights):
    return tf.nn.conv2d(inputTensor, weights, strides=[1, 1, 1, 1], padding='SAME')

  @staticmethod
  def max_pool_1x2(inputTensor):
    return tf.nn.max_pool(inputTensor, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

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

# For testing purpose
def main():
  BATCH_SIZE = 64
  MAX_INPUT_SEQUENCE_LENGTH = 150
  MAX_OUTPUT_SEQUENCE_LENGTH = 22
  FEATURES_COUNT = 40
  TRAINING_ITERATION_COUNT = 1000

  input_placeholder = tf.placeholder(tf.float32, [None, MAX_INPUT_SEQUENCE_LENGTH * FEATURES_COUNT], name="Input__placeholder")
  lengths_placeholder = tf.placeholder(tf.int32, [None], name="Lengths_placeholder")
  output_placeholder = tf.placeholder(tf.int32, [None, MAX_OUTPUT_SEQUENCE_LENGTH], name="True_output_placeholder")
  cldnn = CLDNNModel(input_placeholder, output_placeholder, MAX_INPUT_SEQUENCE_LENGTH, FEATURES_COUNT, 700000)

  init = tf.global_variables_initializer()

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True

  with tf.Session(config = session_config) as session:
    print("Initializing variables...")
    session.run(init)
    print("Variables initialized !")

if __name__ == '__main__':
  main()