class TimitSample(object):
  def __init__(self, sample_name : str):
    self.sample_name = sample_name
    self.audio_file_name = sample_name + ".WAV"
    self.phonemes_file_name = sample_name + ".PHN"
    self.sentence_file_name = sample_name + ".TXT"
    self.words_file_name = sample_name + ".WRD"
    self.mfcc_file_name = sample_name + ".NPY"