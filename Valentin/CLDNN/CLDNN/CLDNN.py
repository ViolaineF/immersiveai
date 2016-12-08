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
                 input_width, input_height,
                 learning_rate = 1e-4,
                 conv_kernel_size = 5, conv_features_count = 32,
                 dimension_reduction_output_size = 256,
                 lstm1_hidden_units_count = 512, lstm2_hidden_units_count = 512
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

        self.prediction
        self.optimize
        self.accuracy

    @define_scope
    def prediction(self):
        input_as_2d_tensor = tf.reshape(self.input_placeholder, [-1, self.input_width, self.input_height, 1])

        # 1st Layer (Convolution)
        ## Weights & Bias
        weights_conv_layer = CLDNNModel.weight_variable([self.conv_kernel_size, self.conv_kernel_size, 1, self.conv_features_count])
        bias_conv_layer = CLDNNModel.bias_variable([self.conv_features_count])
        ## Result
        conv_layer = CLDNNModel.conv2d(input_as_2d_tensor, weights_conv_layer) + bias_conv_layer
        print(conv_layer)
        relu_conv_layer = tf.nn.relu(conv_layer)

        # 2nd Layer (Max Pooling)
        max_pool_layer = CLDNNModel.max_pool_2x2(relu_conv_layer)
        print(max_pool_layer)

        # 3rd Layer (Dimension reduction)
        ## Flattening (from 2D to 1D)
        convoluted_size = int(self.input_width / 2) * int(self.input_height)
        flatten_size = convoluted_size * self.conv_features_count
        max_pool_layer_flatten = tf.reshape(max_pool_layer, [-1, flatten_size])
        ## Weights and Bias
        weights_dim_red_layer = CLDNNModel.weight_variable([flatten_size, self.dimension_reduction_output_size])
        bias_dim_red_layer = CLDNNModel.bias_variable([self.dimension_reduction_output_size])
        ## Result
        dim_red_layer = tf.matmul(max_pool_layer_flatten, weights_dim_red_layer) + bias_dim_red_layer

        # 4th Layer (Concatenation)
        concatenation_layer = tf.concat(1, [dim_red_layer, self.input_placeholder])

        # 5th Layer (LSTM 1)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm1_hidden_units_count)
        lstm_output, lstm_state = tf.nn.rnn(lstm_cell, concatenation_layer, dtype=tf.float32)
        print(lstm_output)

        # 6th Layer (LSTM 2)

        # 7th Layer (Fully connected 1)

        # 7th Layer (Fully connected 2)
        
        return 0 # Should be the 7th layer's ouput

    @define_scope
    def optimize(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.output_placeholder))
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        return optimizer.minimize(cross_entropy)
    
    @define_scope
    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.output_placeholder, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    @staticmethod
    def conv2d(inputTensor, weights):
        return tf.nn.conv2d(inputTensor, weights, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(inputTensor):
        return tf.nn.max_pool(inputTensor, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

# For testing purpose
cldnn = CLDNNModel(tf.placeholder(tf.float32, shape=[None, 32 * 84]), tf.placeholder(tf.float32, shape=[None, 10]), 32, 84)