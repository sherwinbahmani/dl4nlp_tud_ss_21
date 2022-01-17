import numpy as np

class TopKAccuracy(object):
    def __init__(self, k: int = 50) -> None:
        super(TopKAccuracy, self).__init__()
        self.k = k
        self.best = 0

    def compute(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        tps = 0
        for pred, label in zip(predictions[:, :self.k], labels):
            if label in pred:
                tps +=1
        return tps / len(labels)
