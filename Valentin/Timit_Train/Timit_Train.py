import numpy as np
import tflearn
import tensorflow as tf

from tflearn.data_utils import to_categorical, pad_sequences
from Timit_utils.TimitDatabase import TimitDatabase

def pad_sequences_2d(sequences, maxlen, width, value):
    tmp = []
    for sequence in sequences:
        tmp.append(np.reshape(sequence, [-1]))
    tmp = pad_sequences(tmp, maxlen = maxlen * width, value = value)
    tmp = np.reshape(tmp, [len(sequences), maxlen, width])
    return tmp

def to_categorical_2d(sequences, nb_classes):
    seq_len = len(sequences[0])
    tmp = np.reshape(sequences, [-1])
    tmp = to_categorical(tmp, nb_classes = nb_classes)
    tmp = np.reshape(tmp, [len(sequences), seq_len, nb_classes])
    return tmp

def shuffle(inputs, labels):
    order = np.random.shuffle(np.arange(len(inputs)))
    inputs = inputs[order]
    labels = inputs[order]
    return inputs, labels

database_path = r"C:\tmp\TIMIT"
database = TimitDatabase(database_path)

train_dataset = database.train_dataset
test_dataset = database.test_dataset

inputs_maxlen = 200
inputs_width = 40
labels_maxlen = 75
nb_classes = 62

print("Loading database")
train_mfcc, _, train_labels, _ = train_dataset.load_batch()
test_mfcc, _, test_labels, _ = test_dataset.load_batch()

print("Preprocessing : Padding inputs")
train_mfcc = pad_sequences_2d(train_mfcc, inputs_maxlen, inputs_width, 0)
test_mfcc = pad_sequences_2d(test_mfcc, inputs_maxlen, inputs_width, 0)

print("Preprocessing : Padding outputs")
train_labels = pad_sequences(train_labels, labels_maxlen)
test_labels = pad_sequences(test_labels, labels_maxlen)

print("Preprocessing : Labels -> One Hot")
train_labels = to_categorical_2d(train_labels, nb_classes = nb_classes)
test_labels = to_categorical_2d(test_labels, nb_classes = nb_classes)

print("Preprocessing : Shuffling")
shuffle(train_mfcc, train_labels)
shuffle(test_mfcc, test_labels)

conv_feat_count = 32
max_pool_size = 3
conv_result_size = conv_feat_count * int(inputs_width / max_pool_size)
reduction_size = 128

print("Building network")
input_data = tflearn.input_data([None, inputs_maxlen, inputs_width])
### CONV
net = tf.reshape(input_data, [-1, inputs_maxlen, inputs_width, 1])
net = tflearn.conv_2d(net, nb_filter = conv_feat_count, filter_size = [9, 9])
net = tflearn.max_pool_2d(net, [1, max_pool_size])
net = tflearn.reshape(net, [-1, inputs_maxlen * conv_result_size])
### Reduction
net = tflearn.fully_connected(net, inputs_maxlen * reduction_size, activation = "relu")
net = tflearn.reshape(net, [-1, inputs_maxlen, reduction_size])
### LSTM
net = tflearn.merge([input_data, net], mode = "concat", axis = 2)
net = tflearn.lstm(net, n_units = 256, dropout = 0.8, dynamic = False, return_seq = True)
net = tf.transpose(tf.stack(net), [1,0,2])
net = tflearn.lstm(net, n_units = 256, dropout = 0.8, dynamic = False, return_seq = False)
### DNN
net = tflearn.fully_connected(net, 512, activation = "relu")
net = tflearn.fully_connected(net, 512, activation = "relu")
net = tflearn.fully_connected(net, labels_maxlen * nb_classes, activation = "softmax")
net = tflearn.dropout(net, 0.5)
net = tflearn.reshape(net, [-1, labels_maxlen, nb_classes])
net = tflearn.regression(net, optimizer = "adam", learning_rate = 0.001, loss = "categorical_crossentropy")

print("Building model")
model = tflearn.DNN(net, tensorboard_verbose = 0)
print("Training")
model.fit(train_mfcc, train_labels, validation_set = (test_mfcc, test_labels), show_metric = True, batch_size = 8, n_epoch = 100)