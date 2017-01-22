import os
from tqdm import tqdm
import tensorflow as tf

from CLDNNModel import CLDNNModel, CLDNNConfig
from SpeechDataUtils import SpeechDataUtils, SpeechDataSet

class CLDNN():
  def __init__(self):
    self._init_config()
    self._init_data()
    self._init_graph()
    self._init_session()

  def _init_config(self):
    self.features_count = 40
    self.max_input_sequence_length = 150
    self.max_output_sequence_length = 22

    self.librispeech_path = r"E:\tmp\LibriSpeech"

    self.model_config_path = "ModelConfig.ini"
    self.summary_base_path = r"E:\tmp\custom\CLDNN_CTC"
    self.summary_logs_path = self.summary_base_path + r"\logs"
    self.checkpoints_path = self.summary_base_path + r"\checkpoints"
    self.log_every_x_steps = 100
    self.save_every_x_steps = 1000

    if not os.path.exists(self.summary_base_path):
      os.mkdir(self.summary_base_path)
    if not os.path.exists(self.summary_logs_path):
      os.mkdir(self.summary_logs_path)
    if not os.path.exists(self.checkpoints_path):
      os.mkdir(self.checkpoints_path)

  def _init_data(self):
    self.data = SpeechDataUtils(librispeech_path = self.librispeech_path, bucket_size = self.max_input_sequence_length)
    self.train_data = self.data.train
    self.eval_data = self.data.eval

    self.dictionary_size = self.data.dictionary_size
    self.eval_iterations_count = self.eval_data.batch_total_count

  def _init_graph(self):
    self.graph = tf.Graph()
    self.graph.as_default()

    self.cldnn_config = CLDNNConfig(self.max_input_sequence_length, self.features_count, 
                                     self.max_output_sequence_length, self.dictionary_size,
                                     self.model_config_path)
    self.cldnn_model = CLDNNModel(self.cldnn_config)

    self.inference_op = self.cldnn_model.inference
    self.loss_op = self.cldnn_model.loss
    self.train_op = self.cldnn_model.training
    self.eval_op = self.cldnn_model.evaluation

    self.summary = tf.summary.merge_all()
    self.saver = tf.train.Saver()

  def _init_session(self):
    self.session_config = tf.ConfigProto()
    self.session_config.gpu_options.allow_growth = True

    self.session = tf.Session(config = self.session_config)

    self.summary_writer = tf.summary.FileWriter(self.summary_logs_path, self.session.graph)

    checkpoint = tf.train.get_checkpoint_state(self.checkpoints_path)
    if checkpoint and checkpoint.model_checkpoint_path:
      print("Found existing checkpoint in", checkpoint.model_checkpoint_path)
      self.saver.restore(self.session, checkpoint.model_checkpoint_path)
      print("Loaded checkpoints from", checkpoint.model_checkpoint_path)
    else:
      print("No checkpoint found. Starting from scratch.")
      init_op = tf.global_variables_initializer()
      print("Initializing variables...")
      self.session.run(init_op)
      print("Variables initialized !")

  def train(self, training_iteration_count : int, batch_size : int):
    for i in tqdm(range(training_iteration_count)):
      batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = self.train_data.next_batch(batch_size, one_hot = False)

      #print(batch_output_lengths)
      #print(batch_outputs)
      #input()

      feed_dict = {
          self.cldnn_model.input_placeholder : batch_inputs,
          self.cldnn_model.input_lengths_placeholder : batch_input_lengths,
          self.cldnn_model.output_placeholder : batch_outputs,
          self.cldnn_model.output_lengths_placeholder : batch_output_lengths
          }

      _, loss_value = self.session.run([self.train_op, self.loss_op], feed_dict = feed_dict)

      if i%self.log_every_x_steps == 0:
        summary_str = self.session.run(self.summary, feed_dict = feed_dict)
        self.summary_writer.add_summary(summary_str, i)
        self.summary_writer.flush()

      if (i%self.save_every_x_steps == 0) or ((i + 1) == training_iteration_count):
        checkpoint_file = self.checkpoints_path + "model.ckpt"
        print("\nAt step", i, "loss = ", loss_value,'\n')
        self.saver.save(self.session, checkpoint_file, global_step = i)

  def evaluate(self, evaluation_iteration_count : int, batch_size : int):
    pass

def main():
  cldnn = CLDNN()
  cldnn.train(12500, 1)

if __name__ == '__main__':
  main()