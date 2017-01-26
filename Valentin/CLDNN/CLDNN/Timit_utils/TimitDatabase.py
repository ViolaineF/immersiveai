import os

class TimitDatabase(object):
    def __init__(self, timit_database_path : str):
      self.timit_database_path = timit_database_path
      self.train_dataset_path = os.path.join(timit_database_path, "TRAIN")
      self.test_dataset_path = os.path.join(timit_database_path, "TEST")

    def list_all_samples(self) -> dict:
      train_samples_names = TimitDatabase.list_all_samples_inside(self.train_dataset_path)
      test_samples_names = TimitDatabase.list_all_samples_inside(self.test_dataset_path)

      print(len(train_samples_names), len(test_samples_names))
      
      samples_list = \
      {
        "train" : train_samples_names,
        "test" : test_samples_names
      }

    @staticmethod
    def list_all_samples_inside(path : str):
      samples_names = []
      base_path_len = len(path)
      for root, dirs, files in os.walk(path):
        if len(files) == 0:
          continue
        sample_root = root[base_path_len:]
        for file in files:
          if not file.endswith(".TXT"):
            continue
          samples_names.append(sample_root + '\\' + file[:-4])
      return samples_names


a = TimitDatabase(r"E:\tmp\TIMIT")
a.list_all_samples()