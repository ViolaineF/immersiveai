import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

DIGIT_COUNT = 10

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
CONV_IMAGE_SIZE = 49

KERNELSIZE = 5
CONV1_FEATURES_COUNT = 32
CONV2_FEATURES_COUNT = 64
FC_LAYER_NEURONS_COUNT = 1024

LEARNING_RATE = 1e-4

BATCH_SIZE = 50

ITERATION_COUNT = 20000

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

def create_model():
    ### Placeholders (Data : MNIST images and corresponding numbers as labels)
    input_data_placeholder = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name="Input_Flat_Placeholder")
    input_image_placeholder = tf.reshape(input_data_placeholder, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
    output_data_placeholder = tf.placeholder(tf.float32, [None, DIGIT_COUNT], name="True_Output_Placeholder")

    ### First layer (Conv + ReLU + Max Pool)
    weights_conv_layer1 = weight_variable([KERNELSIZE, KERNELSIZE, 1, CONV1_FEATURES_COUNT])
    biaises_conv_layer1 = bias_variable([CONV1_FEATURES_COUNT])
    hidden_conv_layer1 = tf.nn.relu(conv2d(input_image_placeholder, weights_conv_layer1) + biaises_conv_layer1)
    hidden_max_pool_layer1 = max_pool_2x2(hidden_conv_layer1)
    
    ### Second layer (Conv + ReLU + Max Pool)
    weights_conv_layer2 = weight_variable([KERNELSIZE, KERNELSIZE, CONV1_FEATURES_COUNT, CONV2_FEATURES_COUNT])
    biaises_conv_layer2 = bias_variable([CONV2_FEATURES_COUNT])
    hidden_conv_layer2 = tf.nn.relu(conv2d(hidden_max_pool_layer1, weights_conv_layer2) + biaises_conv_layer2)
    hidden_max_pool_layer2 = max_pool_2x2(hidden_conv_layer2)

    ### Third layer (Fully Connected)
    hidden_max_pool_layer2_flatten = tf.reshape(hidden_max_pool_layer2, [-1, CONV_IMAGE_SIZE * CONV2_FEATURES_COUNT])
    weights_fc_layer1 = weight_variable([CONV_IMAGE_SIZE * CONV2_FEATURES_COUNT, FC_LAYER_NEURONS_COUNT])
    biaises_fc_layer1 = bias_variable([FC_LAYER_NEURONS_COUNT])
    hidden_fc_layer1 = tf.nn.relu(tf.matmul(hidden_max_pool_layer2_flatten, weights_fc_layer1) + biaises_fc_layer1)

    ### Dropout
    keep_prob = tf.placeholder(tf.float32)
    hidden_fc_layer1_drop = tf.nn.dropout(hidden_fc_layer1, keep_prob)

    ### Fourth layer (Fully Connected for Readout)
    weights_fc_layer2 = weight_variable([FC_LAYER_NEURONS_COUNT, DIGIT_COUNT])
    biaises_fc_layer2 = bias_variable([DIGIT_COUNT])

    predicted_output = tf.matmul(hidden_fc_layer1_drop, weights_fc_layer2) + biaises_fc_layer2
    
    return (input_data_placeholder, output_data_placeholder, predicted_output, keep_prob)

def main():
    print("Loading MNIST data ...")
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
    print("MNIST data loaded !")
    
    (input_data_placeholder, output_data_placeholder, predicted_output, keep_prob) = create_model()


##  for i in range(20000):
##      batch = mnist.train.next_batch(50)
##      if i%100 == 0:
##          train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
##          print("step %d, training accuracy %g"%(i, train_accuracy))
##      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
##
##  print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predicted_output, output_data_placeholder))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(predicted_output, 1), tf.argmax(output_data_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    session = tf.InteractiveSession()
    
    print("Initializing variables...")
    session.run(tf.global_variables_initializer())
    print("Variables initialized !")

    for i in tqdm(range(ITERATION_COUNT)):
        batch_inputs, batch_outputs = mnist.train.next_batch(BATCH_SIZE)
        #if i%100 == 0:
            #train_accuracy = accuracy.eval(feed_dict={input_data_placeholder : batch_inputs, output_data_placeholder : batch_outputs, keep_prob: 1.0})
            #print("Step %d/%d, training accuracy = %g"%(i, ITERATION_COUNT, train_accuracy))

        session.run(train_step, feed_dict={input_data_placeholder : batch_inputs, output_data_placeholder : batch_outputs, keep_prob: 0.5})
        
    test_accuracy = 0
    for i in tqdm(range(100)):
        test_input, test_outputs = mnist.test.next_batch(100)
        test_accuracy += accuracy.eval(feed_dict={input_data_placeholder : test_input, output_data_placeholder : test_outputs, keep_prob: 1.0})
    test_accuracy /= 100
    print("Test accuracy = %g"%test_accuracy)

    session.close()

if __name__ == '__main__':
    main()
