from pydub import AudioSegment

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

import os
import numpy as np
from tqdm import tqdm
import gc

class SpeechData(object):
    def __init__(self, filename : str, times : list):
        self.filename = filename
        self.times = times

        # Opening and reading of the audio file
        audio_file = AudioSegment.from_mp3(filename)

        channels_count = audio_file.channels
        samples = audio_file.get_array_of_samples()
        samples_count = int(len(samples) / channels_count)

        samples = np.reshape(samples, (samples_count, channels_count))
        rate = audio_file.frame_rate

        # Features extracting
        features = []
        for [start, end] in times:
            sentence = samples[start:end]

            mfcc_feat = mfcc(sentence, rate)
            d_mfcc_feat = delta(mfcc_feat, 2)
            fbank_feat = logfbank(sentence, rate, nfilt = 40)

            features.append(fbank_feat)

        self.features = np.array(features)

    def save_fbank_as_binary(self, filename=None):
        if filename is None:
            if self.filename.endswith(".mp3") :
                filename = self.filename.replace(".mp3", ".npy")
            else:
                filename += ".npy"

        np.save(filename, self.features)

    @staticmethod
    def process_all_in_directory(dirpath:str, frame_rate = 16000):
        if not os.path.isdir(dirpath):
            print(dirpath + " is not a valid directory. Aborting.")
            return

        if not os.path.isdir(dirpath):
            print(dirpath + " is not a valid directory. Aborting.")
            return

        # Subfunction for gathering all files names
        def walk_into_librispeech(dirpath : str):
            def get_times_files_in_dir(root : str):
                files_in_directory = os.listdir(root)

                base_file = None
                intro_file = None

                for file in files_in_directory:
                    if file.endswith(".seg.txt") and not file.endswith(".sents.seg.txt"):
                        if "intro" in file:
                            intro_file = file
                        else:
                            base_file = file

                assert (base_file != None), "Couldn't find xx-xx.seg.txt inside " + dirpath
                return (base_file, intro_file)

            audio_files = []
            times_files = []
            times_files_intro = []

            for root, directories, filenames in os.walk(dirpath):
                for filename in filenames:
                    if filename.endswith(".mp3"):
                        ## Adding mp3 file to list
                        audio_files.append(os.path.join(root, filename))
                        ## Looking for *.seg.txt next to mp3 file (intro.seg.txt too, if present)
                        (base_file, intro_file) = get_times_files_in_dir(root)
                        times_files.append(os.path.join(root, base_file))
                        if intro_file is not None:
                            times_files_intro.append(os.path.join(root, intro_file))

            return (audio_files, times_files, times_files_intro)

        # Subfunction to extract times from files
        def get_all_times(times_files : list, frame_rate = 16000):
            all_times = []
            for times_file in times_files:
                file = open(times_file)
                all_text = file.read()

                lines = str.split(all_text, '\n')
                times = []
                for line in lines:
                    if line is '':
                        continue

                    infos = str.split(line, ' ')
                    if(len(infos) < 3):
                        continue

                    start = int(float(infos[1]) * frame_rate)
                    end = int(float(infos[2]) * frame_rate)

                    times.append([start, end])
                all_times.append(times)
            return all_times

        (audio_files, times_files, times_files_intro) = walk_into_librispeech(dirpath)

        files_count = len(audio_files) # should be equal to the number of time files
        all_times = get_all_times(times_files)

        for i in tqdm(range(files_count)):
            audio_file = audio_files[i]
            times = all_times[i]
            speech_data = SpeechData(audio_file, times)
            speech_data.save_fbank_as_binary()
            gc.collect()

SpeechData.process_all_in_directory(r"H:\LibriSpeech\mp3")