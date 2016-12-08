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
                 dimension_reduction_output_size = 256
                 ):
        self.input_placeholder = input_placeholder
        self.output_placeholder = output_placeholder
        
        self.input_size = int(input_placeholder.get_shape()[1])
        self.output_size = int(output_placeholder.get_shape()[1])

        self.input_width = input_width
        self.input_height = input_height

        self.learning_rate = learning_rate

        self.dimension_reduction_output_size = dimension_reduction_output_size

        self.conv_kernel_size = conv_kernel_size
        self.conv_features_count = conv_features_count

        self.prediction
        self.optimize
        self.accuracy

    @define_scope
    def prediction(self):
        input_image_placeholder = tf.reshape(self.input_placeholder, [-1, self.input_width, self.input_height, 1])

        # 1st Layer (Convolution)
        ## Weights & Bias
        weights_conv_layer = CLDNNModel.weight_variable([self.conv_kernel_size, self.conv_kernel_size, 1, self.conv_features_count])
        bias_conv_layer = CLDNNModel.bias_variable([self.conv_features_count])
        ## Result
        conv_layer = CLDNNModel.conv2d(input_image_placeholder, weights_conv_layer) + bias_conv_layer
        relu_conv_layer = tf.nn.relu(conv_layer)

        # 2nd Layer (Max Pooling)
        max_pool_layer = CLDNNModel.max_pool_2x2(relu_conv_layer)

        # 3rd Layer (Dimension reduction)
        ## Flattening (from 2D to 1D)
        max_pool_layer_flatten = tf.reshape(max_pool_layer, [])
        ## Weights and Bias
        weights_dim_red_layer = CLDNNModel.weight_variable([1, self.dimension_reduction_output_size])
        bias_dim_red_layer = CLDNNModel.bias_variable([self.dimension_reduction_output_size])
        ## Result
        dim_red_layer = tf.matmul(max_pool_layer_flatten, weights_dim_red_layer) + bias_dim_red_layer

        # 4th Layer (Concatenation)

        # 5th Layer (LSTM 1)

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
        return tf.nn.max_pool(inputTensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

# For testing purpose
#cldnn = CLDNNModel(tf.placeholder(tf.float32, shape=[None, 20]), tf.placeholder(tf.float32, shape=[None, 10]), 4, 5)