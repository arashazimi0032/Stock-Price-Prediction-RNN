from keras.models import Sequential
from keras.layers import Dense


class Model:
    def __init__(self, recurrent_layer, input_shape):
        self.history = None

        self.model = Sequential()

        self.model.add(recurrent_layer(64, return_sequences=True, input_shape=input_shape))

        self.model.add(recurrent_layer(64, return_sequences=True))

        self.model.add(recurrent_layer(64, return_sequences=True))

        self.model.add(recurrent_layer(64))

        self.model.add(Dense(1))

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def summary(self):
        self.model.summary()

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        self.history = self.model.history

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
