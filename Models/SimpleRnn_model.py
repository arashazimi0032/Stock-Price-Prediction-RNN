from keras.layers import SimpleRNN
from . import Model


class SimpleRnn(Model):
    def __init__(self, input_shape):
        super(SimpleRnn, self).__init__(SimpleRNN, input_shape=input_shape)

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def summary(self):
        self.model.summary()

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        self.history = self.model.history

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
