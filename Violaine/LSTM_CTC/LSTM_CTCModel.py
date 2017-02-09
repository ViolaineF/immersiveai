import tensorflow as tf
import numpy as np
from tqdm import tqdm

from Timit_utils.TimitDatabase import TimitDatabase
from ModelSkeletons.SupervisedModel import SupervisedModel, define_scope
from LSTM_CTCConfig import LSTM_CTCModelConfig

class LSTM_CTCModel(SupervisedModel):
    def __init__(self, config : LSTM_CTCModelConfig):
        self.config = config

        self.graph = tf.Graph()
        self.graph.as_default()

        self.build_placeholders()
        self.inference
        self.loss
        self.training
        self.decoded_inference

    def build_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape = [None, self.config.input_max_time, self.config.input_frame_size], name = "input_placeholder")
        self.input_lengths_placeholder = tf.placeholder(tf.int32, shape = [None], name = "input_lengths_placeholder")
        self.output_placeholder = tf.sparse_placeholder(tf.int32, name = "output_placeholder")
        self.output_lengths_placeholder = tf.placeholder(tf.int32, shape = [None], name = "output_placeholder")

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
        tf.summary.scalar('loss', self.loss)
        global_step = tf.Variable(0, name='global_step', trainable=False)

        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        return optimizer.minimize(self.loss, global_step = global_step)

    @define_scope
    def evaluation(self):
        pass

    @define_scope
    def decoded_inference(self):
        inference_timemajor = tf.transpose(self.inference, [1, 0, 2])
        decoded, prob = tf.nn.ctc_greedy_decoder(inference_timemajor, self.output_lengths_placeholder)
        return decoded
 
    @staticmethod
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

    def train(self, train_data, iteration_count : int, batch_size : int):
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True

        self.session = tf.Session(config = session_config)

        self.init_model()
        for i in tqdm(range(iteration_count)):
            feed_dict = self.train_step(train_data, batch_size)
            if((i + 1)%100 == 0):
                summary_str = self.session.run(self.summary, feed_dict = feed_dict)
                self.summary_writer.add_summary(summary_str, i)
                self.summary_writer.flush()
                print('######')
                print("Loss :", self.session.run(self.loss, feed_dict = feed_dict))
                print('######')

            if((i + 1)%1000 == 0):
                self.saver.save(self.session, self.config.checkpoints_path + r"\network", global_step=i)

        self.test(train_data)

    def init_model(self):
        self.summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.config.checkpoints_path + "/logs", self.session.graph)
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.config.checkpoints_path)

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        else: 
            self.session.run(tf.global_variables_initializer())

    def train_step(self, database, batch_size : int):
        train_data = database.train_dataset
        batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = train_data.next_batch(batch_size, one_hot=False)

        tmp = []
        for j in range(len(batch_outputs)):
            tmp.append(batch_outputs[j][:batch_output_lengths[j]])

        batch_outputs_feed = np.array(tmp)
        batch_outputs_feed = LSTM_CTCModel.tokens_for_sparse(batch_outputs_feed)

        feed_dict = \
        {
            self.input_placeholder : batch_inputs,
            self.input_lengths_placeholder : batch_input_lengths,
            self.output_placeholder : batch_outputs_feed,
            self.output_lengths_placeholder : batch_output_lengths
        }

        self.session.run(self.training, feed_dict = feed_dict)
        return feed_dict

    def test(self, database):
        ##### TEST #####
        test_data = database.test_dataset
        batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = test_data.next_batch(1, False)

        tmp = []
        for j in range(len(batch_outputs)):
            tmp.append(batch_outputs[j][:batch_output_lengths[j]])

        batch_outputs_feed = np.array(tmp)
        batch_outputs_feed = LSTM_CTCModel.tokens_for_sparse(batch_outputs_feed)

        feed_dict = \
        {
            self.input_placeholder : batch_inputs,
            self.input_lengths_placeholder : batch_input_lengths,
            self.output_placeholder : batch_outputs_feed,
            self.output_lengths_placeholder : batch_output_lengths
        }

        decoded = self.session.run(self.decoded_inference, feed_dict = feed_dict)

        results = []
        for decoded_path in decoded:
            phonemes_ids = decoded_path.values
            result_words = ""
            for idx in phonemes_ids:
                word = database.id_to_phoneme_dictionary[idx]
                if word != "<EOS>":
                    result_words += word + ' '
            results += [result_words]

        target = batch_outputs[0]
        target_words = ""
        for idx in target:
            word = database.id_to_phoneme_dictionary[idx]
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

        eval = self.session.run(tf.edit_distance(tf.cast(self.decoded_inference[0], tf.int32), self.output_placeholder), feed_dict = feed_dict)
        print(eval)

if __name__ == '__main__':
    model_config = LSTM_CTCModelConfig("timit_model_config.ini")
    model = LSTM_CTCModel(model_config)

    timit_database = TimitDatabase(r"C:\tmp\TIMIT")
    train_data = timit_database.train_dataset

    model.train(train_data, 0, 11)