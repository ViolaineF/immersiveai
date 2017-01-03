import os
import numpy as np
from threading import Thread
from asyncio import Queue
from asyncio import StreamReader
from tqdm import tqdm

import time

class SpeechDataUtils(object):
  def __init__(self, librispeech_path = r"D:\tmp\LibriSpeech"):
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

def get_mfcc_lengths(librispeech_path : str, save_to_file = True, sort_lengths = True, force_process = False):
  # Retrieve existing file
  mfcc_lengths_file = os.path.join(librispeech_path, "mfcc_lengths.npy")
  mfcc_lengths = None

  if not force_process and os.path.exists(mfcc_lengths_file):
    mfcc_lengths = np.load(mfcc_lengths_file)
    print("Found existing list of mfcc lengths with", len(mfcc_lengths), "sentences.")
    return mfcc_lengths

  preprocess_order = get_preprocess_order(librispeech_path)
  # If force to (re)process or if file is not found, process
  for file_index in tqdm(range(len(preprocess_order))):
    entry = preprocess_order[file_index]
    full_path = librispeech_path + entry[1]
    mfcc = np.load(full_path)

    sentence_count = len(mfcc)
    for sentence_index in range(sentence_count):
      mfcc_length = len(mfcc[sentence_index])
      sentence_info = [file_index, sentence_index, mfcc_length]

      if mfcc_lengths is None:
        mfcc_lengths = np.array(sentence_info)
      else:
        mfcc_lengths = np.vstack([mfcc_lengths, sentence_info])
  
  mfcc_lengths = mfcc_lengths[mfcc_lengths[:,2].argsort()]

  if save_to_file:
    np.save(mfcc_lengths_file, mfcc_lengths)

  return mfcc_lengths

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
    if word.endswith('\n'):
      word = word[:-1]
    dictionnary[word] = int(id)

  dictionnary_file.close()

  return dictionnary

def get_transcript_from_file_and_index(transcript_file_path : str, sentence_index : int):
    transcript_file = open(transcript_file_path, 'r')
    transcript_lines = transcript_file.readlines()
    transcript_line = transcript_lines[sentence_index]
    transcript_file.close()

    # Parsing
    split_index = transcript_line.find(' ') + 1

    words_in_line = transcript_line[split_index:]
    words_in_line = words_in_line.split(' ')
    words_count_in_line = len(words_in_line)

    transcript = ' '

    for word_index in range(words_count_in_line):
      word = words_in_line[word_index]

      if word.endswith("\'S"):
        word = word[:-2]
      if word.endswith("\'"):
        word = word[:-1]

      transcript += word
      if (word_index + 1) != words_count_in_line:
        transcript += ' '

    return transcript, words_count_in_line

def get_mfcc_from_file_and_index(sentences_mfcc_file_path : str, sentence_index : int):
  sentences_mfcc_in_file = np.load(sentences_mfcc_file_path)
  sentence_mfcc = sentences_mfcc_in_file[sentence_index]

  return sentence_mfcc

def get_best_bucket(length : int, buckets : list):
  for bucket in buckets:
    if(length <= bucket):
      length = bucket
      break
  if(length > buckets[-1]):
    length = buckets[-1]
  return length

def save_batch(librispeech_path : str, batch_id : int, mfcc_padded_length : int, mfcc_batch, 
               mfcc_batch_lengths : list, transcripts_batch : list, transcripts_batch_lengths : list, tokenized_transcripts_batch : list):
  batch_directory = librispeech_path + r"\batches"
  ### MFCC
  mfcc_batch = np.array(mfcc_batch)
  batch_file_path = "mfcc_batch" + str(batch_id) + "_l" + str(mfcc_padded_length) + ".npy"
  batch_file_path = os.path.join(batch_directory, batch_file_path)
  np.save(batch_file_path, mfcc_batch)

  ### Lengths
  mfcc_batch_lengths = np.array(mfcc_batch_lengths)
  mfcc_batch_lengths_file_path = "mfcc_batch_lengths_" + str(batch_id) + "_l" + str(mfcc_padded_length) + ".npy"
  mfcc_batch_lengths_file_path = os.path.join(batch_directory, mfcc_batch_lengths_file_path)
  np.save(mfcc_batch_lengths_file_path, mfcc_batch_lengths)

  ### Transcript
  transcripts_batch_file_path = "transcripts_batch_" + str(batch_id) + "_l" + str(mfcc_padded_length) + ".txt"
  transcripts_batch_file_path = os.path.join(batch_directory, transcripts_batch_file_path)
  transcripts_batch_output_file = open(transcripts_batch_file_path, 'w')
  transcripts_batch_output_file.writelines(transcripts_batch)        
  transcripts_batch_output_file.close()

  transcripts_batch_lengths = np.array(transcripts_batch_lengths)
  transcripts_batch_lengths_file_path = "transcripts_batch_lengths_" + str(batch_id) + "_l" + str(mfcc_padded_length) + ".npy"
  transcripts_batch_lengths_file_path = os.path.join(batch_directory, transcripts_batch_lengths_file_path)
  np.save(transcripts_batch_lengths_file_path, transcripts_batch_lengths)

  tokenized_transcripts_batch = np.array(tokenized_transcripts_batch)
  tokenized_transcripts_batch_file_path = "tokenized_transcripts_batch_" + str(batch_id) + "_l" + str(mfcc_padded_length) + ".npy"
  tokenized_transcripts_batch_file_path = os.path.join(batch_directory, tokenized_transcripts_batch_file_path)
  np.save(tokenized_transcripts_batch_file_path, tokenized_transcripts_batch)

  return batch_file_path, mfcc_batch_lengths_file_path, transcripts_batch_file_path, transcripts_batch_lengths_file_path, tokenized_transcripts_batch_file_path

def tokenize_transcript(dictionary : dict, transcript : str, split_char = ' '):
  words = transcript.split(split_char)
  
  transcript_length = len(words)
  result = []
  for i in range(transcript_length):
    word = words[i]
    if word.endswith('\n'):
      word = word[:-1]
    if word in dictionary:
      result.append(dictionary[word])
  return result

def create_batches_of_sequences(librispeech_path : str, batch_size = 5000, buckets = (150, 250, 500, 1000, 1500, 2000)):
  preprocess_order = get_preprocess_order(librispeech_path)
  mfcc_lengths = get_mfcc_lengths(librispeech_path)
  word_dictionary = get_words_dictionnary(librispeech_path)

  sentence_count = len(mfcc_lengths)
  i = 0
  batch_id = 0

  batches_infos = ""

  while i < sentence_count:
    batch_of_mfcc_infos = mfcc_lengths[i:min(i + batch_size, sentence_count - 1)]

    # MFCC
    mfcc_batch = []
    mfcc_batch_lengths = []

    # Transcripts
    transcripts_batch = []
    transcripts_batch_lengths = []
    tokenized_transcripts_batch = []

    max_mfcc_length_in_batch = 0

    test_ratio = 0

    # Main Loop : Gathering data + determination of max length
    print("Starting batch n°" + str(batch_id) + "...")

    for j in tqdm(range(len(batch_of_mfcc_infos))):
      [file_index, sentence_index, mfcc_length] = batch_of_mfcc_infos[j]

      ### MFCC
      sentences_mfcc_file_path = librispeech_path + preprocess_order[file_index][1]
      sentence_mfcc = get_mfcc_from_file_and_index(sentences_mfcc_file_path, sentence_index)
      mfcc_batch.append(sentence_mfcc)

      ### Length
      max_mfcc_length_in_batch = max(max_mfcc_length_in_batch, mfcc_length)
      mfcc_batch_lengths.append(mfcc_length)

      ### Text
      transcript_file_path = librispeech_path + preprocess_order[file_index][2]
      transcript_file_path = transcript_file_path.replace("seg", "trans")
      transcript, transcript_length = get_transcript_from_file_and_index(transcript_file_path, sentence_index)
      transcripts_batch += [transcript]
      transcripts_batch_lengths += [transcript_length]
      tokenized_transcripts_batch += [tokenize_transcript(word_dictionary, transcript)]

      test_ratio += mfcc_length / transcript_length

    print("test ratio", batch_id, "=", test_ratio / len(batch_of_mfcc_infos))

    # Choice of bucket (and padding size)
    mfcc_padded_length = get_best_bucket(max_mfcc_length_in_batch, buckets)

    # Padding
    for j in tqdm(range(len(batch_of_mfcc_infos))):
      sentence_mfcc = mfcc_batch[j]
      pad_length = mfcc_padded_length - len(sentence_mfcc)
      mfcc_batch[j] = np.lib.pad(sentence_mfcc, ((0, pad_length), (0,0)), 'constant', constant_values=0)

    # Saving (data and metadata)
    batch_file_path, mfcc_batch_lengths_file_path, transcripts_batch_file_path, transcripts_batch_lengths_file_path, tokenized_transcripts_batch_file_path = save_batch(librispeech_path, batch_id, mfcc_padded_length, mfcc_batch, mfcc_batch_lengths, transcripts_batch, transcripts_batch_lengths, tokenized_transcripts_batch)
    batches_infos += str(mfcc_padded_length) + ' ' + batch_file_path + ' ' + mfcc_batch_lengths_file_path + ' ' + transcripts_batch_file_path + ' ' + transcripts_batch_lengths_file_path + ' ' + tokenized_transcripts_batch_file_path + '\n'

    # Iteration
    i += batch_size
    batch_id += 1

  batches_infos_output_file = open(os.path.join(librispeech_path, "batches_infos.txt"), 'w')
  batches_infos_output_file.write(batches_infos)        
  batches_infos_output_file.close()

def main():
  librispeech_path = r"E:\LibriSpeech"
  create_batches_of_sequences(librispeech_path, batch_size = 5000)

if __name__ == '__main__':
  main()
