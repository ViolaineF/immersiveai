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

    self.inference
    self.loss
    self.training

    self.evaluation

  @define_scope
  def inference(self):
    input_image_placeholder = tf.reshape(self.input_placeholder, [-1, self.image_width, self.image_height, 1])

    # 1st Layer (Convolution + ReLU + Max pooling)
    with tf.name_scope('conv1'):
      weights = mnist_model.weight_variable([self.kernel_size, self.kernel_size, 1, self.conv1_features_count])
      biases = mnist_model.bias_variable([self.conv1_features_count])
      hidden_conv_layer1 = tf.nn.relu(mnist_model.conv2d(input_image_placeholder, weights) + biases)
      hidden_max_pool_layer1 = mnist_model.max_pool_2x2(hidden_conv_layer1)

    # 2nd Layer (Convolution + ReLU + Max pooling)
    with tf.name_scope('conv2'):
      weights = mnist_model.weight_variable([self.kernel_size, self.kernel_size, self.conv1_features_count, self.conv2_features_count])
      biases = mnist_model.bias_variable([self.conv2_features_count])
      hidden_conv_layer2 = tf.nn.relu(mnist_model.conv2d(hidden_max_pool_layer1, weights) + biases)
      hidden_max_pool_layer2 = mnist_model.max_pool_2x2(hidden_conv_layer2)

    # 3rd Layer (Fully connected)
    with tf.name_scope('fully_connected1'):
      convoluted_image_size = int(self.image_size/16)
      fc_size = int(convoluted_image_size * self.conv2_features_count)

      hidden_max_pool_layer2_flatten = tf.reshape(hidden_max_pool_layer2, [-1, fc_size])
      weights = mnist_model.weight_variable([fc_size, self.fully_connected_units_count])
      biaises = mnist_model.bias_variable([self.fully_connected_units_count])
      hidden_fc_layer1 = tf.nn.relu(tf.matmul(hidden_max_pool_layer2_flatten, weights) + biaises)
  
      ## Dropout
    hidden_fc_layer1_drop = tf.nn.dropout(hidden_fc_layer1, self.dropout_placeholder)

    # 4th Layer (Fully connected)
    with tf.name_scope('fully_connected2'):
      weights_fc_layer2 = mnist_model.weight_variable([self.fully_connected_units_count, self.output_size])
      biaises_fc_layer2 = mnist_model.bias_variable([self.output_size])

      logits = tf.matmul(hidden_fc_layer1_drop, weights_fc_layer2) + biaises_fc_layer2

    return logits

  @define_scope
  def loss(self):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.inference, self.output_placeholder)
    return tf.reduce_mean(cross_entropy)

  @define_scope
  def training(self):
    tf.summary.scalar('loss', self.loss)
    optimizer = tf.train.AdamOptimizer(self.learning_rate)

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(self.loss, global_step=global_step)
    #train_op = optimizer.minimize(self.loss)
    return train_op

  @define_scope
  def evaluation(self):
    correct_inference = tf.equal(tf.argmax(self.inference, 1), tf.argmax(self.output_placeholder, 1))
    return tf.reduce_mean(tf.cast(correct_inference, tf.float32))

  # Fonction pour "alléger" la construction du modèle
  def conv2d(inputTensor, weights):
    return tf.nn.conv2d(inputTensor, weights, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(inputTensor):
    return tf.nn.max_pool(inputTensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  def weight_variable(shape, name='weights'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape, name='biases'):
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

  with tf.Graph().as_default():
    # Placeholders
    input_placeholder = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name="Input_flat_placeholder")
    output_placeholder = tf.placeholder(tf.float32, [None, DIGIT_COUNT], name="True_output_placeholder")
    dropout_placeholder = tf.placeholder(tf.float32, name="Dropout_placeholder")

    # Model
    model = mnist_model(input_placeholder, output_placeholder, dropout_placeholder)

    #loss = model.loss
    train_op = model.training
    loss = model.loss
    # evaluation : for testing
    evaluation = model.evaluation

    summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as session:
      summary_writer = tf.summary.FileWriter('/tmp/custom/MNIST_Exp/logs/', session.graph)

      print("Initializing variables...")
      session.run(init)
      print("Variables initialized !")

      # Training
      for i in tqdm(range(TRAINING_ITERATION_COUNT)):
        batch_inputs, batch_outputs = mnist.train.next_batch(TRAINING_BATCH_SIZE)
        feed_dict = {
          input_placeholder : batch_inputs,
          output_placeholder : batch_outputs,
          dropout_placeholder: 0.5
          }
        _, loss_value = session.run([train_op, loss], feed_dict=feed_dict)

        if i%100 == 0:
          summary_str = session.run(summary, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, i)
          summary_writer.flush()

        if i%1000 == 0 or (i + 1) == TRAINING_ITERATION_COUNT:
          checkpoint_file = '/tmp/custom/MNIST_Exp/logs/model.ckpt'
          saver.save(session, checkpoint_file, global_step=i)

      # Testing
      test_accuracy = 0
      for i in tqdm(range(TESTING_ITERATION_COUNT)):
        test_input, test_outputs = mnist.test.next_batch(TESTING_BATCH_SIZE)
        test_accuracy += session.run(evaluation, feed_dict={input_placeholder : test_input, output_placeholder : test_outputs, dropout_placeholder: 1.0})
      test_accuracy /= TESTING_ITERATION_COUNT
      print("Test accuracy = %g"%test_accuracy)

if __name__ == '__main__':
    main()
