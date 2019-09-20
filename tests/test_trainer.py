import unittest
import pickle
import numpy as np
from sklearn.datasets import fetch_kddcup99
from anomaly_detection.trainer import Trainer, InsufficientTrainingDataError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


class TestTrainer(unittest.TestCase):
    def setUp(self):
        with open("data/train.pickle", mode="rb") as f:
            self.train_data = pickle.load(f)

    def test_train_and_save(self):
        trainer = Trainer()
        trainer.train(self.train_data["features"])
        trainer.save("tmp.model")

    def test_train_with_insufficient_data(self):
        trainer = Trainer()
        with self.assertRaises(InsufficientTrainingDataError):
            trainer.train([])

    def test_train_with_small_data(self):
        trainer = Trainer()
        trainer.train([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        trainer.save("tmp.model")
#####ori#######
    def check_data(self):
        self.features, self.labels = fetch_kddcup99(subset="http", return_X_y=True)
        assert self.features[1] > 2

    def test_closs_validation(self):
        trainer = Trainer()
        kf = KFold(n_splits=3)
        self.features, self.labels = fetch_kddcup99(subset="http", return_X_y=True)
        self.labels = list(map(lambda label: 0 if label == b"normal." else 1, self.labels))
        self.labels = np.array(self.labels)
        for train_index, test_index in kf.split(self.features, self.labels):
            train_data = self.features[train_index]
            test_data = self.features[test_index]
            train_label = self.labels[train_index]
            test_label = self.labels[test_index]
            trainer.train(train_data)
            result = trainer.model.predict(test_data)
            accuracy = accuracy_score(test_label, result)
            print("正解率＝", accuracy)
            assert accuracy > 0.8
    def test_SVM(self):
        trainer = Trainer()
        kf = KFold(n_splits=3)
        self.features, self.labels = fetch_kddcup99(subset="http", return_X_y=True)
        self.labels = list(map(lambda label: 0 if label == b"normal." else 1, self.labels))
        self.labels = np.array(self.labels)
        for train_index, test_index in kf.split(self.features, self.labels):
            train_data = self.features[train_index]
            test_data = self.features[test_index]
            train_label = self.labels[train_index]
            test_label = self.labels[test_index]
            trainer.train_SVM(train_data)
            result = trainer.model.predict(test_data)
            result = [0 if i == 1 else i for i in result]
            result = [1 if i == -1 else i for i in result]
            accuracy = accuracy_score(test_label, result)
            print("正解率＝", accuracy)
            assert accuracy > 0.8
#####ori#######
