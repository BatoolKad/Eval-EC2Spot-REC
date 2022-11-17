"""
Class model used to prepare each machine learning model for drawing REC

"""
import numpy as np
from Models.rec_plot.rec_plot.rec import RegressionErrorCharacteristic


def calculate_rec(y_true, y_pred):
    return RegressionErrorCharacteristic(y_true, y_pred)


class Model:
    def __init__(self, model_name, color="black", precision=3):
        self.model_name = model_name
        self.color = color
        self.precision = precision
        self.predictions = []
        self.targets = []

    def prepare(self):
        prediction_s = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        self.reg_metrics = RegressionErrorCharacteristic(targets, prediction_s)