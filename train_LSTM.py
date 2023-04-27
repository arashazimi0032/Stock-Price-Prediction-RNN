import matplotlib.pyplot as plt
from Utils import *
from Models import Lstm
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

train_path = r'D:\E\freelancer projects\Stock-Price-Prediction-RNN\data\Google_Stock_Price_Train.csv'
test_path = r'D:\E\freelancer projects\Stock-Price-Prediction-RNN\data\Google_Stock_Price_Test.csv'

model_check_point = ModelCheckpoint(filepath=r'D:\E\freelancer projects\Stock-Price-Prediction-RNN\Models\LSTM.h5',
                                    )

scheduler = LearningRateScheduler(lstm_scheduler)

sequence_length = 120

X_train, X_test, y_train, y_test, scaler = prepare_data(train_path, test_path, sequence_length)

lstm = Lstm(input_shape=(X_train.shape[1], X_train.shape[2]))

lstm.compile(loss='mse', optimizer='adam')

lstm.summary()

lstm.fit(X_train, y_train, batch_size=32, epochs=70, validation_data=(X_test, y_test), callbacks=[model_check_point,
                                                                                                  scheduler])


pred = lstm.predict(X_test)

print('MSE: ', mse(scaler.inverse_transform(y_test), scaler.inverse_transform(pred)))

pd.DataFrame(lstm.history.history).plot()
plt.title('LSTM')

plt.figure(figsize=(15, 9))
plt.plot(pred, 'b', label='Predict')
plt.plot(y_test, 'r', label='Truth')
plt.legend()
plt.title('LSTM prediction')

plt.show()
