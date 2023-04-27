# Stock Price Prediction RNN

This repository is an academic exercise for the Google stock price prediction project on the Google_Stock_Price Dataset.
The Google_Stock_Price includes the daily price of Google stocks from 2012 to 2016.

In this project, using the data of the past 120 days and using different types of recurrent networks 
(Simple RNN, LSTM and GRU), the stock price for the next 20 days has been estimated.

Prediction results of every network is shown below:
#### Simple RNN Prediction
<img src="./images/simple%20rnn%20prediction.png"/>

#### GRU Prediction
<img src="./images/gru%20prediction.png"/>

#### LSTM Prediction
<img src="./images/lstm%20prediction.png"/>

#### The MSE Results
|  | Simple RNN | GRU     | LSTM    |
|-----|---------|---------|---------|
|MSE| 15.1917 | 12.8776 | 10.9656 |

## Requirements
- python 3.9
- Tensorflow == 2.11.0
- pandas == 1.5.3
- numpy == 1.24.2
- matplotlib == 3.6.3
- keras~=2.11.0
- scikit-learn~=1.1.1

## License
This repository is released under [Apache License V2](http://www.apache.org/licenses/LICENSE-2.0). To develop,
publication and use it, please follow the terms of this license.
