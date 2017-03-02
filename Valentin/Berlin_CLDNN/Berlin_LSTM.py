import numpy as np
import tflearn

from tflearn.data_utils import to_categorical
from Berlin_utils.BerlinDatabase import BerlinDatabase, build_timit_database

MFCC_COUNT = 26
PADDING = 200

def shuffle(inputs, labels):
    order = np.random.shuffle(np.arange(len(inputs)))
    inputs = inputs[order]
    labels = inputs[order]
    return inputs, labels

build_timit_database(r"C:\tmp\Berlin", MFCC_COUNT, 0.025, 0.025)
database = BerlinDatabase(r"C:\tmp\Berlin", MFCC_COUNT)
print("Loading database")
database.load_batch(padding = PADDING)
print("Database loaded")

mfcc = database.mfcc_features
print("Preprocessing : Labels -> One Hot")
labels = to_categorical(database.emotion_tokens, 5)

shuffle(mfcc, labels)

train_mfcc = mfcc[:-100]
train_labels = labels[:-100]

test_mfcc = mfcc[-100:]
test_labels = labels[-100:]

net = tflearn.input_data([None, PADDING, MFCC_COUNT])
net = tflearn.lstm(net, n_units = 128, dropout = 0.8, dynamic = True)
net = tflearn.fully_connected(net, 5, activation = 'softmax')
net = tflearn.regression(net, optimizer = "adam", learning_rate = 0.001, loss = "categorical_crossentropy")

model = tflearn.DNN(net, tensorboard_verbose = 0)
model.fit(train_mfcc, train_labels, validation_set = (test_mfcc, test_labels), show_metric = True, batch_size = 8, n_epoch = 100)
result = model.evaluate(test_mfcc, test_labels, batch_size = 100)