import tensorflow as tf

from Reinforcement.ModelSkeleton import ModelSkeleton, define_scope
from Reinforcement.DQN.DQNConfig import DQNConfig

class DQNModel(ModelSkeleton):
  def __init__(self, config : DQNConfig):
    self.config = config
    
    self.input_size = config.input_width * config.input_height * config.input_depth
    self.input_width = config.input_width
    self.input_height = config.input_height
    self.input_depth = config.input_depth
    self.output_size = self.action_count = config.actions_count

    self.input_placeholder = tf.placeholder(tf.float32, (None, self.input_size))
    self.action_placeholder = tf.placeholder(tf.float32, (None, config.actions_count))
    self.target_placeholder = tf.placeholder(tf.float32, (None))

    self._placeholders = {
      "input" : self.input_placeholder,
      "action" : self.action_placeholder,
      "bidule" : self.target_placeholder
    }

    self.inference
    #self.loss
    #self.training
    #self.evaluation

  @define_scope
  def placeholders(self):
    return self._placeholders

  @define_scope
  def inference(self):
    input_as_4d_tensor = tf.reshape(self.input_placeholder, (-1, self.input_width, self.input_height, self.input_depth))

    with tf.name_scope("conv1"):
      convolution_weights_1 = tf.Variable(tf.truncated_normal([8, 8, self.input_depth, 32], stddev=0.01))
      convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[32]))
      convolution_layer1 = tf.nn.conv2d(input_as_4d_tensor, convolution_weights_1, strides = [1, 4, 4, 1], padding="SAME") + convolution_bias_1
      convolution_layer1_relu = tf.nn.relu(convolution_layer1)
      convolution_layer1_maxpool = tf.nn.max_pool(convolution_layer1_relu, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    with tf.name_scope("conv2"):
      convolution_weights_2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
      convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[64]))
      convolution_layer2 = tf.nn.conv2d(convolution_layer1_maxpool, convolution_weights_2, strides = [1, 2, 2, 1], padding="SAME") + convolution_bias_2
      convolution_layer2_relu = tf.nn.relu(convolution_layer2)
      convolution_layer2_maxpool = tf.nn.max_pool(convolution_layer2_relu, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    with tf.name_scope("conv3"):
      convolution_weights_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
      convolution_bias_3 = tf.Variable(tf.constant(0.01, shape=[64]))
      convolution_layer3 = tf.nn.conv2d(convolution_layer2_maxpool, convolution_weights_3, strides = [1, 1, 1, 1], padding="SAME") + convolution_bias_3
      convolution_layer3_relu = tf.nn.relu(convolution_layer3)
      convolution_layer3_maxpool = tf.nn.max_pool(convolution_layer3_relu, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
      convolution_layer3_flatten = tf.reshape(convolution_layer3_maxpool, [-1, 256])

    with tf.name_scope("feed_forward1"):
      feed_forward_weights_1 = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
      feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))
      feed_forward_layer1 = tf.matmul(convolution_layer3_flatten, feed_forward_weights_1) + feed_forward_bias_1
      feed_forward_layer1_relu = tf.nn.relu(feed_forward_layer1)

    with tf.name_scope("feed_forward2"):
      feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, self.action_count], stddev=0.01))
      feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[self.action_count]))
      output_layer = tf.matmul(feed_forward_layer1_relu, feed_forward_weights_2) + feed_forward_bias_2
    return output_layer

  @define_scope
  def loss(self):
    return super().loss()

  @define_scope
  def training(self):
    return super().training()

  @define_scope
  def evaluation(self):
    return super().evaluation()