import tensorflow as tf
import os

from BerlinDatabase import BerlinDatabase

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class BerlinTFRecords(object):
    def __init__(self, database : BerlinDatabase):
        self.database = database
        self.database_path = database.database_path
        if self.database is None:
            raise Exception("No database was provided")

    def convert_to_tfrecords(self, validation_size = 50):
        self.database.load_batch()
        if self.database.samples_count <= validation_size:
            raise Exception("Validation size (%d) is bigger than database size (%d)" % (validation_size, self.database.samples_count))
        mfcc = self.database.mfcc_features
        mfcc_lengths = self.database.mfcc_features_lengths
        labels = self.database.emotion_tokens

        train = (
            mfcc[:-validation_size],
            mfcc_lengths[:-validation_size],
            labels[:-validation_size])

        valid = (
            mfcc[-validation_size:],
            mfcc_lengths[-validation_size:],
            labels[-validation_size:])

        self.convert_dataset_to_tfrecords(train, "train")
        self.convert_dataset_to_tfrecords(valid, "valid")

    def convert_dataset_to_tfrecords(self, dataset, dataset_name):
        mfcc = dataset[0]
        mfcc_lengths = dataset[1]
        labels = dataset[2]
        samples_count = len(mfcc)
        if len(mfcc_lengths) != samples_count:
            raise ValueError("MFCCs samples size (%d) does not match MFCCs lengths samples size (%d)" % (samples_count, len(mfcc_lengths)))
        if len(labels) != samples_count:
            raise ValueError("MFCCs samples size (%d) does not match labels samples size (%d)" % (samples_count, len(labels)))

        filename = os.path.join(self.database_path, dataset_name + ".tfrecords")
        print("Converting dataset : %s (writing to %s)" % (dataset_name, filename))
        writer = tf.python_io.TFRecordWriter(filename)
        for i in range(samples_count):
            mfcc_raw = mfcc[i].tostring()
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'mfcc_raw' : _bytes_feature(mfcc_raw),
                        'mfcc_lengths' : _int64_feature(mfcc_lengths[i]),
                        'labels' : _int64_feature(labels[i])
                        }
                    )
                )
            writer.write(example.SerializeToString())
        writer.close()

if __name__ == "__main__":
    database_path = r"C:\tmp\Berlin"
    database = BerlinDatabase(database_path)
    converter = BerlinTFRecords(database)
    converter.convert_to_tfrecords()