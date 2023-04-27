import matplotlib.pyplot as plt
from Utils import *
from Models import Gru
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

train_path = r'D:\E\freelancer projects\Stock-Price-Prediction-RNN\data\Google_Stock_Price_Train.csv'
test_path = r'D:\E\freelancer projects\Stock-Price-Prediction-RNN\data\Google_Stock_Price_Test.csv'

model_check_point = ModelCheckpoint(filepath=r'D:\E\freelancer projects\Stock-Price-Prediction-RNN\Models\GRU.h5',
                                    )

scheduler = LearningRateScheduler(gru_scheduler)

sequence_length = 120

X_train, X_test, y_train, y_test, scaler = prepare_data(train_path, test_path, sequence_length)

gru = Gru(input_shape=(X_train.shape[1], X_train.shape[2]))

gru.compile(loss='mse', optimizer='adam')

gru.summary()

gru.fit(X_train, y_train, batch_size=32, epochs=40, validation_data=(X_test, y_test), callbacks=[model_check_point,
                                                                                                 scheduler])


pred = gru.predict(X_test)

print('MSE: ', mse(scaler.inverse_transform(y_test), scaler.inverse_transform(pred)))

pd.DataFrame(gru.history.history).plot()
plt.title('GRU')

plt.figure(figsize=(15, 9))
plt.plot(pred, 'b', label='Predict')
plt.plot(y_test, 'r', label='Truth')
plt.legend()
plt.title('GRU prediction')

plt.show()
