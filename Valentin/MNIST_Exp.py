import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
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

class mnist_model:
	def __init__(self,
			  input_placeholder, output_placeholder, dropout_placeholder, 
			  image_width = 28, image_height = 28,
			  kernel_size = 5, conv1_features_count = 32, conv2_features_count = 64,
			  fully_connected_units_count = 1024,
			  learning_rate = 1e-4
			  ):
		self.input_placeholder = input_placeholder
		self.output_placeholder = output_placeholder
		self.dropout_placeholder = dropout_placeholder

		self.output_size = int(output_placeholder.get_shape()[1])

		self.image_width = int(image_width)
		self.image_height = int(image_height)
		self.image_size = int(image_width * image_height)

		self.kernel_size = int(kernel_size)
		self.conv1_features_count = int(conv1_features_count)
		self.conv2_features_count = int(conv2_features_count)

		self.fully_connected_units_count = int(fully_connected_units_count)
		
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

		# 2nd Layer (Convolution + ReLU + Max pooling)
		weights_conv_layer2 = mnist_model.weight_variable([self.kernel_size, self.kernel_size, self.conv1_features_count, self.conv2_features_count])
		biaises_conv_layer2 = mnist_model.bias_variable([self.conv2_features_count])
		hidden_conv_layer2 = tf.nn.relu(mnist_model.conv2d(hidden_max_pool_layer1, weights_conv_layer2) + biaises_conv_layer2)
		hidden_max_pool_layer2 = mnist_model.max_pool_2x2(hidden_conv_layer2)

		# 3rd Layer (Fully connected)
		convoluted_image_size = int(self.image_size/16)
		fc_size = int(convoluted_image_size * self.conv2_features_count)

		hidden_max_pool_layer2_flatten = tf.reshape(hidden_max_pool_layer2, [-1, fc_size])
		weights_fc_layer1 = mnist_model.weight_variable([fc_size, self.fully_connected_units_count])
		biaises_fc_layer1 = mnist_model.bias_variable([self.fully_connected_units_count])
		hidden_fc_layer1 = tf.nn.relu(tf.matmul(hidden_max_pool_layer2_flatten, weights_fc_layer1) + biaises_fc_layer1)

		## Dropout
		hidden_fc_layer1_drop = tf.nn.dropout(hidden_fc_layer1, self.dropout_placeholder)

		# 4th Layer (Fully connected)
		weights_fc_layer2 = mnist_model.weight_variable([self.fully_connected_units_count, self.output_size])
		biaises_fc_layer2 = mnist_model.bias_variable([self.output_size])

		return tf.matmul(hidden_fc_layer1_drop, weights_fc_layer2) + biaises_fc_layer2

	@define_scope
	def optimize(self):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.output_placeholder))
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		return optimizer.minimize(cross_entropy)

	@define_scope
	def accuracy(self):
		correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.output_placeholder, 1))
		return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Fonction pour "alléger" la construction du modèle
	def conv2d(inputTensor, weights):
		return tf.nn.conv2d(inputTensor, weights, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(inputTensor):
		return tf.nn.max_pool(inputTensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

def main():
	# MNIST loading
	mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

	# Parameters
	DIGIT_COUNT = 10
	IMAGE_SIZE = 784

	TRAINING_BATCH_SIZE = 50
	# Reduire si sur CPU (risque de prendre ~1h sinon)
	TRAINING_ITERATION_COUNT = 20000
	TESTING_BATCH_SIZE = 1000
	TESTING_ITERATION_COUNT = int(mnist.test.num_examples / TESTING_BATCH_SIZE)

	# Placeholders
	input_placeholder = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name="Input_flat_placeholder")
	output_placeholder = tf.placeholder(tf.float32, [None, DIGIT_COUNT], name="True_output_placeholder")
	dropout_placeholder = tf.placeholder(tf.float32, name="Dropout_placeholder")

	# Model
	model = mnist_model(input_placeholder, output_placeholder, dropout_placeholder)
	# Optimize : for training
	optimize = model.optimize
	# Accuracy : for testing
	accuracy = model.accuracy

	with tf.Session() as session:
		print("Initializing variables...")
		session.run(tf.global_variables_initializer())
		print("Variables initialized !")

		# Training
		for i in tqdm(range(TRAINING_ITERATION_COUNT)):
			batch_inputs, batch_outputs = mnist.train.next_batch(TRAINING_BATCH_SIZE)
			session.run(optimize, feed_dict={input_placeholder : batch_inputs, output_placeholder : batch_outputs, dropout_placeholder: 0.5})

		# Testing
		test_accuracy = 0
		for i in tqdm(range(TESTING_ITERATION_COUNT)):
			test_input, test_outputs = mnist.test.next_batch(TESTING_BATCH_SIZE)
			test_accuracy += session.run(accuracy, feed_dict={input_placeholder : test_input, output_placeholder : test_outputs, dropout_placeholder: 1.0})
		test_accuracy /= TESTING_ITERATION_COUNT
		print("Test accuracy = %g"%test_accuracy)

if __name__ == '__main__':
    main()
