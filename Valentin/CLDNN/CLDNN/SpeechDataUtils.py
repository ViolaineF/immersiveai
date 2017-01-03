import os
import numpy as np
from threading import Thread
from asyncio import Queue
from asyncio import StreamReader
from tqdm import tqdm

import time

class SpeechDataUtils(object):
  def __init__(self, librispeech_path = r"D:\tmp\LibriSpeech\mp3"):
    self.librispeech_path = librispeech_path
    self.preprocess_order = get_preprocess_order(librispeech_path)

    self.features_count = 40
    self.dictionnary = get_words_dictionnary(librispeech_path)
    self.dictionnary_size = len(self.dictionnary)

    self.preloaded_batches = Queue()

  def preload_batch(self, batch_size : int):
    thread = Thread(target=self._preload_batch_coroutine, args=[batch_size])
    thread.start()
    return thread

  def _preload_batch_coroutine(self, batch_size : int):
    # (Input, Lengths, Output)

    inputs = []
    lengths = []
    outputs = []

    for i in range(batch_size):

      input_data = np.zeros((100, 40))
      length_data = 1
      output_data = np.zeros((10)).astype(int)

      inputs += [input_data]
      lengths += [length_data]
      outputs += [output_data]

    batch = (inputs, lengths, outputs)
    self.preloaded_batches.put_nowait(batch)

  def get_preloaded_batch(self, batch_size_if_empty):
    if self.preloaded_batches.empty():
      thread = self.preload_batch(batch_size_if_empty)
      thread.join()

    batch = self.preloaded_batches.get_nowait()
    return batch

  def has_preloaded_batch(self):
    return not self.preloaded_batches.empty()

def get_preprocess_order(librispeech_path : str):
  preprocess_order_file = open(os.path.join(librispeech_path, "process_order.txt"), 'r')

  preprocess_order = preprocess_order_file.readlines()
  for i in range(len(preprocess_order)):
    # Removing '\n' at the end with :   [:-1]
    tmp = preprocess_order[i][:-1]
    # and splitting into [index, npy_file_path, sentence_file_path] with :   split(',')
    preprocess_order[i] = tmp.split(',')

  preprocess_order_file.close()

  return preprocess_order

def get_sequences_lengths(librispeech_path : str, save_to_file = True, sort_lengths = True, force_process = False):
  # Retrieve existing file
  sentence_lengths_file = os.path.join(librispeech_path, "sentence_lengths.npy")
  sentence_lengths = None

  if not force_process and os.path.exists(sentence_lengths_file):
    sentence_lengths = np.load(sentence_lengths_file)
    print("Found existing list of sequences lengths with", len(sentence_lengths), "sentences.")
    return sentence_lengths

  preprocess_order = get_preprocess_order(librispeech_path)
  # If force to (re)process or if file is not found, process
  for file_index in tqdm(range(len(preprocess_order))):
    entry = preprocess_order[file_index]
    full_path = librispeech_path + entry[1]
    mfcc = np.load(full_path)

    sentence_count = len(mfcc)
    for sentence_index in range(sentence_count):
      sentence_length = len(mfcc[sentence_index])
      sentence_info = [file_index, sentence_index, sentence_length]

      if sentence_lengths is None:
        sentence_lengths = np.array(sentence_info)
      else:
        sentence_lengths = np.vstack([sentence_lengths, sentence_info])
  
  sentence_lengths = sentence_lengths[sentence_lengths[:,2].argsort()]

  if save_to_file:
    np.save(sentence_lengths_file, sentence_lengths)

  return sentence_lengths

def get_batches_files(librispeech_path : str):
  pass

def get_words_dictionnary(librispeech_path : str):
  # Retrieve existing file
  dictionnary_file_path = os.path.join(librispeech_path, "dictionnary.txt")
  dictionnary = dict()

  dictionnary_file = open(dictionnary_file_path, 'r')
  lines = dictionnary_file.readlines()

  for i in range(len(lines)):
    line = lines[i]
    id, word = line.split(' ')
    dictionnary[id] = word

  return dictionnary

def create_batches_of_sequences(librispeech_path : str, batch_size = 5000, buckets = (150, 250, 500, 1000, 1500, 2000)):
  preprocess_order = get_preprocess_order(librispeech_path)
  sentence_lengths = get_sequences_lengths(librispeech_path)

  sentence_count = len(sentence_lengths)
  i = 0
  batch_id = 0

  batches_infos = ""

  while i < sentence_count:
    batch_of_sentence_infos = sentence_lengths[i:min(i + batch_size, sentence_count - 1)]
    batch = []
    batch_sentence_lengths = []
    max_length_in_batch = 0
    # Main Loop : Gathering data + determination of max length
    print("Starting batch n°" + str(batch_id) + "...")
    for j in tqdm(range(len(batch_of_sentence_infos))):
      [file_index, sentence_index, sentence_length] = batch_of_sentence_infos[j]
      file_path = librispeech_path + preprocess_order[file_index][1]
      sentences_mfcc_in_file = np.load(file_path)
      sentence_mfcc = sentences_mfcc_in_file[sentence_index]
      # Max Length
      max_length_in_batch = max(max_length_in_batch, len(sentence_mfcc))
      # Data
      batch.append(sentence_mfcc)
      batch_sentence_lengths.append(sentence_length)

    # Choice of bucket (and padding size)
    padded_length = max_length_in_batch
    for bucket in buckets:
      if(padded_length <= bucket):
        padded_length = bucket
        break
    if(padded_length > buckets[-1]):
      padded_length = buckets[-1]

    # Padding
    for j in tqdm(range(len(batch_of_sentence_infos))):
      sentence_mfcc = batch[j]
      pad_length = padded_length - len(sentence_mfcc)
      batch[j] = np.lib.pad(sentence_mfcc, ((0, pad_length), (0,0)), 'constant', constant_values=0)

    # Saving (data and metadata)
    batch = np.array(batch)
    batch_file_path = r"\batches\batch_" + str(batch_id) + "_l" + str(padded_length) + ".npy"
    np.save(librispeech_path + batch_file_path, batch)

    batch_sentence_lengths = np.array(batch_sentence_lengths)
    batch_sentence_lengths_file_path = r"\batches\batch_sentence_lengths_" + str(batch_id) + "_l" + str(padded_length) + ".npy"
    np.save(librispeech_path + batch_sentence_length_file_path, batch_sentence_lengths)

    batches_infos += padded_length + ' ' + batch_file_path + ' ' + batch_sentence_length_file_path + '\n'

    # Iteration
    i += batch_size
    batch_id += 1

  batches_infos_output_file = open(os.path.join(librispeech_path, "batches_infos.txt"), 'w')
  batches_infos_output_file.write(batches_infos)        
  batches_infos_output_file.close()

def main():
  librispeech_path = r"D:\tmp\LibriSpeech"
  create_batches_of_sequences(librispeech_path)

if __name__ == '__main__':
  main()
  #sentence_lengths = np.load(r"D:\tmp\LibriSpeech\mp3\sentence_lengths.npy")



  #result = ""

  #for i in tqdm(range(len(sentence_lengths))):
  #  result += str(sentence_lengths[i][2])
  #  if (i + 1) < len(sentence_lengths):
  #    result += '\n'
  
  #f = open(r"D:\tmp\LibriSpeech\mp3\sentence_lengths.csv", 'w')
  #f.write(result)
  #f.close()