import matplotlib.pyplot as plt
from Utils import *
from Models import SimpleRnn
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

train_path = r'D:\E\freelancer projects\Stock-Price-Prediction-RNN\data\Google_Stock_Price_Train.csv'
test_path = r'D:\E\freelancer projects\Stock-Price-Prediction-RNN\data\Google_Stock_Price_Test.csv'

model_check_point = ModelCheckpoint(filepath=r'D:\E\freelancer projects\Stock-Price-Prediction-RNN\Models\SimpleRNN.h5',
                                    )

scheduler = LearningRateScheduler(simple_rnn_scheduler)

sequence_length = 120

X_train, X_test, y_train, y_test, scaler = prepare_data(train_path, test_path, sequence_length)

rnn = SimpleRnn(input_shape=(X_train.shape[1], X_train.shape[2]))

rnn.compile(loss='mse', optimizer='adam')

rnn.summary()

rnn.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[model_check_point,
                                                                                                 scheduler])

pred = rnn.predict(X_test)

print('MSE: ', mse(scaler.inverse_transform(y_test), scaler.inverse_transform(pred)))

pd.DataFrame(rnn.history.history)[['loss', 'val_loss']].plot()
plt.title('Simple RNN')

plt.figure(figsize=(15, 9))
plt.plot(pred, 'b', label='Predict')
plt.plot(y_test, 'r', label='Truth')
plt.legend()
plt.title('Simple RNN prediction')

plt.show()
