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

class lstm_model:
	def __init__(self,
			  input_placeholder, output_placeholder,
			  learning_rate = 1e-4
			  ):
		self.input_placeholder = input_placeholder
		self.output_placeholder = output_placeholder

		self.input_size = int(input_placeholder.get_shape()[1])
		self.output_size = int(output_placeholder.get_shape()[1])
		
		self.learning_rate = learning_rate

		self.prediction
		self.optimize
		self.accuracy

	@define_scope
	def prediction(self):
		input_image_placeholder = tf.reshape(self.input_placeholder, [-1, self.image_width, self.image_height, 1])

		# 1st Layer (Convolution + ReLU + Max pooling)
		weights_conv_layer1 = mnist_model.weight_variable([self.kernel_size, self.kernel_size, 1, self.conv1_features_count])
		biaises_conv_layer1 = mnist_model.bias_variable([self.conv1_features_count])
		hidden_conv_layer1 = tf.nn.relu(mnist_model.conv2d(input_image_placeholder, weights_conv_layer1) + biaises_conv_layer1)
		hidden_max_pool_layer1 = mnist_model.max_pool_2x2(hidden_conv_layer1)

		return ...

	@define_scope
	def optimize(self):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.output_placeholder))
		optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
		return optimizer.minimize(cross_entropy)

	@define_scope
	def accuracy(self):
		correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.output_placeholder, 1))
		return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
