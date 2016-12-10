from pydub import AudioSegment

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

import scipy.io.wavfile as wav
import os
import numpy as np

class SpeechData(object):
    def __init__(self, filename : str):

        self.mp3_filename = filename

        if ".mp3" in filename:
            filename = filename.replace(".mp3", ".wav")
        else:
            filename += ".wav"
        self.wav_filename = filename

    def __enter__(self):
        audio_file_mp3 = AudioSegment.from_mp3(self.mp3_filename)
        audio_file_wav = audio_file_mp3.export(self.wav_filename, format = "wav")

        (rate,sig) = wav.read(self.wav_filename)
        mfcc_feat = mfcc(sig, rate)

        d_mfcc_feat = delta(mfcc_feat, 2)
        self.fbank_feat = logfbank(sig, rate, nfilt = 40)
        return self

    def __exit__(self, type, value, traceback):
        os.remove(self.wav_filename)

    def save_fbank_as_binary(self, filename=None):
        if filename is None:
            if ".mp3" in self.mp3_filename:
                filename = self.mp3_filename.replace(".mp3", ".npy")
            else:
                filename += ".npy"

        np.save(filename, self.fbank_feat)