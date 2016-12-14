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
            filename = SpeechData.get_npy_filename(self.filename)

        np.save(filename, self.features)

    @staticmethod
    def get_npy_filename(filename : str):
        if filename.endswith(".mp3") :
            filename = filename.replace(".mp3", ".npy")
        else:
            filename += ".sents.npy"

        return filename

    @staticmethod
    def process_all_in_directory(dirpath:str, frame_rate = 16000, check_for_existing_npy_file = False):
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
                    if file.endswith(".sents.seg.txt"):
                    #if file.endswith(".seg.txt") and not file.endswith(".sents.seg.txt"):
                        if "intro" in file:
                            intro_file = file
                        else:
                            base_file = file

                assert (base_file != None), "Couldn't find xx-xx.sents.seg.txt inside " + dirpath
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

        print("Searching mp3 files in", dirpath)
        (audio_files, times_files, times_files_intro) = walk_into_librispeech(dirpath)

        files_count = len(audio_files) # should be equal to the number of time files
        print("Found", files_count, "mp3 files,", len(times_files), "alignement files (plus", len(times_files_intro),"introduction aligment files) in", dirpath)
        print("Processing alignement files...")
        all_times = get_all_times(times_files)
        print("Alignement files processed !")

        print("Writing preprocess order file...")
        process_order = ""
        for i in range(files_count):
            process_order += str(i) + ',' + audio_files[i] + ',' + times_files[i]
            if i != (files_count - 1) :
                process_order += '\n'
        process_order_file = open(os.path.join(dirpath, "process_order.txt"), 'w')
        process_order_file.write(process_order)
        process_order_file.close()
        print("Preprocess order file written")

        print("Starting main pre process loop ...")
        for i in tqdm(range(files_count)):
            audio_file = audio_files[i]
            times = all_times[i]

            if check_for_existing_npy_file:
                npy_filename = SpeechData.get_npy_filename(audio_file)
                if os.path.exists(npy_filename):
                    continue

            speech_data = SpeechData(audio_file, times)
            speech_data.save_fbank_as_binary()
            gc.collect()
        print("Job done !")

    @staticmethod
    def is_spoken_text_filename(filename : str):
        #return filename.endswith(".trans.txt") and not filename.endswith(".sents.trans.txt")
        return filename.endswith(".sents.trans.txt")

    @staticmethod
    def gather_dictionnary(dirpath : str):
        spoken_text_files = []

        print("Looking for spoken books files (*.sents.trans.txt)")
        for root, directories, filenames in os.walk(dirpath):
            for filename in filenames:
                if SpeechData.is_spoken_text_filename(filename):
                    spoken_text_files.append(os.path.join(root,filename))
        print("Found", len(spoken_text_files), "spoken books files")
        
        dictionnary = []
        print("Building dictionnary ...")
        for i in tqdm(range(len(spoken_text_files))):
            filename = spoken_text_files[i]
            file = open(filename, "r")
            all_text = file.read()

            lines = all_text.split('\n')
            for line in lines:
                split_index = line.find(' ') + 1
                if split_index < 1:
                    continue
                words_in_line = line[split_index:]
                words_in_line = words_in_line.split(' ')

                for word in words_in_line:
                    if word.endswith("\'S"):
                        word = word[:-2]
                    if word.endswith("\'"):
                        word = word[:-1]
                    if not word in dictionnary:
                        dictionnary.append(word)
        print("Finished building dictionnary, found", len(dictionnary), "different words")
        
        output_string = ""
        for i in tqdm(range(len(dictionnary))):
            output_string += str(i) + ' ' + dictionnary[i]
            if i < (len(dictionnary) - 1):
                output_string += '\n'

        print("Saving dictionnary under", os.path.join(dirpath, "dictionnary.txt"))

        output_file = open(os.path.join(dirpath, "dictionnary.txt"), 'w')
        output_file.write(output_string)        
        output_file.close()      

#SpeechData.process_all_in_directory(r"F:\LibriSpeech\mp3", check_for_existing_npy_file = True)
SpeechData.gather_dictionnary(r"F:\LibriSpeech\mp3")
