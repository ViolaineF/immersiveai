import tensorflow as tf
import numpy as np
from tqdm import tqdm

from Timit_utils.TimitDatabase import TimitDatabase
from ModelSkeletons.SupervisedModel import SupervisedModel, define_scope
from LSTM_CTCConfig import LSTM_CTCConfig

class LSTM_CTCModel(SupervisedModel):
  def __init__(self, config : LSTM_CTCConfig):
    self.config = config

    self.build_placeholders()
    self.inference
    self.loss
    self.training

  def build_placeholders(self):
    self.input_placeholder = tf.placeholder(tf.float32, shape = [None, self.config.input_max_time, self.config.input_frame_size])
    self.input_lengths_placeholder = tf.placeholder(tf.int32, shape = [None])
    self.output_placeholder = tf.sparse_placeholder(tf.int32)
    self.output_lengths_placeholder = tf.placeholder(tf.int32, shape = [None])

  @define_scope
  def inference(self):
    # LSTM
    with tf.name_scope("LSTM"):
      cell = tf.nn.rnn_cell.LSTMCell(self.config.cell_size)
      cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.config.cell_count)
      rnn, _ = tf.nn.dynamic_rnn(cell, self.input_placeholder, self.input_lengths_placeholder, dtype = tf.float32)

      flatten_rnn = tf.reshape(rnn, shape = [-1, self.config.input_max_time * self.config.cell_size])

    # Hidden layers
    layer = flatten_rnn
    for i in range(self.config.fully_connected_count):
      with tf.name_scope("FullyConnected_n" + str(i)):
        previous_layer_size = int(layer.get_shape()[1])
        # WEIGHTS & BIASES #
        weights = tf.Variable(tf.random_normal(shape = [previous_layer_size, self.config.fully_connected_size], stddev = 0.1))
        biases = tf.Variable(tf.zeros(shape = [self.config.fully_connected_size]))
        # MATMUL + RELU #
        layer = tf.matmul(layer, weights) + biases
        layer = tf.nn.relu(layer)

    # Last layer
    # WEIGHTS & BIASES #
    weights = tf.Variable(tf.random_normal(shape = [self.config.fully_connected_size, self.config.output_max_time * self.config.output_frame_size],stddev = 0.1))
    biases = tf.Variable(tf.zeros(shape = [self.config.output_max_time * self.config.output_frame_size]))
    # MATMUL + RELU #
    layer = tf.matmul(layer, weights) + biases
    layer = tf.nn.relu(layer)
    
    logits = tf.reshape(layer, shape = [-1, self.config.output_max_time, self.config.output_frame_size])
    return logits

  @define_scope
  def loss(self):
    logits_timemajor = tf.transpose(self.inference, [1, 0, 2])
    ctc_loss = tf.nn.ctc_loss(logits_timemajor, self.output_placeholder, self.output_lengths_placeholder)
    return tf.reduce_mean(ctc_loss)

  @define_scope
  def training(self):
    optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
    return optimizer.minimize(self.loss)

  @define_scope
  def evaluation(self):
    pass
  
def tokens_for_sparse(sequences):
  eos_value = 9632
  tmp = []
  for seq_idx in range(len(sequences)):
    seq = sequences[seq_idx]
    for i in range(len(seq)):
      end_idx = i
      if seq[i] == eos_value:
        break
    tmp.append(seq[:end_idx])
      
  sequences = tmp

  indices = []
  values = []

  for n, seq in enumerate(sequences):
      indices.extend(zip([n]*len(seq), range(len(seq))))
      values.extend(seq)

  indices = np.asarray(indices, dtype=np.int64)
  values = np.asarray(values, dtype=np.int32)
  shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

  return indices, values, shape

def main():
  timit_database = TimitDatabase(r"G:\TIMIT")

  ##### PARAMETERS #######
  max_input_sequence_length = timit_database.get_max_mfcc_features_length()
  max_output_sequence_length = timit_database.get_max_phonemes_length()
  dictionary_size = timit_database.phonemes_dictionary_size

  train_data = timit_database.train_dataset
  eval_data = timit_database.test_dataset

  lstm_config = LSTM_CTCConfig()
  lstm_config.input_max_time = max_input_sequence_length
  lstm_config.input_frame_size = 40
  lstm_config.output_max_time = max_output_sequence_length
  lstm_config.output_frame_size = dictionary_size

  lstm_model = LSTM_CTCModel(lstm_config)

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  ##### SESSION #####
  with tf.Session(config = session_config) as session:
    init = tf.global_variables_initializer()
    session.run(init)

    for i in tqdm(range(1000)):
      batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = train_data.next_batch(11, one_hot=False)

      tmp = []
      for j in range(len(batch_outputs)):
        tmp.append(batch_outputs[j][:batch_output_lengths[j]])

      batch_outputs = np.array(tmp)
      batch_outputs = tokens_for_sparse(batch_outputs)

      feed_dict = \
      {
        lstm_model.input_placeholder : batch_inputs,
        lstm_model.input_lengths_placeholder : batch_input_lengths,
        lstm_model.output_placeholder : batch_outputs,
        lstm_model.output_lengths_placeholder : batch_output_lengths
      }

      session.run(lstm_model.training, feed_dict = feed_dict)

      if(i%100 == 0):
        print('######')
        print("Loss :", session.run(lstm_model.loss, feed_dict = feed_dict))
        print('######')

    ##### TEST #####
    batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = train_data.next_batch(1, False)
    feed_dict = \
    {
      lstm_model.input_placeholder : batch_inputs,
      lstm_model.input_lengths_placeholder : batch_input_lengths,
    }

    test = session.run(lstm_model.inference, feed_dict = feed_dict)
    test = session.run(tf.transpose(test, [1, 0, 2]))
    decoded, prob = tf.nn.ctc_beam_search_decoder(test, batch_output_lengths, 100, 5)

    decoded = session.run(decoded)
    prob = session.run(prob)

    results = []
    for decoded_path in decoded:
      phonemes_ids = decoded_path.values
      result_words = ""
      for idx in phonemes_ids:
        word = timit_database.id_to_phoneme_dictionary[idx]
        if word != "<EOS>":
          result_words += word + ' '
      results += [result_words]

    target = batch_outputs[0]
    target_words = ""
    for idx in target:
      word = timit_database.id_to_phoneme_dictionary[idx]
      if word != "<EOS>":
        target_words += word + ' '

    print("--------------")
    print("--------------")
    print("--------------")
    print("Target : \n", target_words, '\n')
    print("Results")
    for result_sentence in results:
      print(result_sentence)
    #print(prob)
    print("--------------")
    print("--------------")
    print("--------------")

main()