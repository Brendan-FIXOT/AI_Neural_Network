import numpy as np

class LossFunction :
    @staticmethod
    def mse(Y_true, Y_pred) :
        return np.mean((Y_true - Y_pred) ** 2)

    @staticmethod
    def mse_derivative(Y_true, Y_pred) :
        return (2 / Y_true.shape[0]) * (Y_pred - Y_true)