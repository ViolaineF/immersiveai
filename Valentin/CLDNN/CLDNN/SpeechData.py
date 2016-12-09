from pydub import AudioSegment

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

import scipy.io.wavfile as wav

class SpeechData(object):
    def __init__(self, filename : str):

        audio_file_mp3 = AudioSegment.from_mp3(filename)

        if ".mp3" in filename:
            filename = filename.replace(".mp3", ".wav")
        else:
            filename += ".wav"

        audio_file_wav = audio_file_mp3.export(filename, format = "wav")

        (rate,sig) = wav.read(filename)
        mfcc_feat = mfcc(sig, rate)

        d_mfcc_feat = delta(mfcc_feat, 2)
        self.fbank_feat = logfbank(sig, rate, nfilt = 40)



