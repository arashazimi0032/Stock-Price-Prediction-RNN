import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def min_max_scaler(train, test):
    scaler = MinMaxScaler()
    train_sc = scaler.fit_transform(train)
    test_sc = scaler.transform(test)
    return train_sc, test_sc, scaler


def train_sequence_generator(train, sequence_length=60):
    x_train = []
    y_train = []

    for i in range(sequence_length, len(train)):
        x_train.append(train[i - sequence_length: i])
        y_train.append(train[i])

    return np.array(x_train), np.array(y_train)


def test_sequence_generator(train, test, sequence_length=60):
    total_data = np.concatenate([train, test])
    inputs = total_data[len(total_data) - len(test) - sequence_length:]

    return train_sequence_generator(inputs, sequence_length=sequence_length)


def mse(y_test, pred):
    return np.sqrt(mean_squared_error(y_test, pred))


def prepare_data(train_path, test_path, sequence_length=60):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train = train_data['Open'].values.reshape(-1, 1)
    test = test_data['Open'].values.reshape(-1, 1)

    train, test, scaler = min_max_scaler(train, test)

    x_train, y_train = train_sequence_generator(train, sequence_length=sequence_length)
    x_test, y_test = test_sequence_generator(train, test, sequence_length=sequence_length)

    return x_train, x_test, y_train, y_test, scaler
