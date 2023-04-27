from keras.layers import GRU
from . import Model


class Gru(Model):
    def __init__(self, input_shape):
        super(Gru, self).__init__(GRU, input_shape=input_shape)

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def summary(self):
        self.model.summary()

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        self.history = self.model.history

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
