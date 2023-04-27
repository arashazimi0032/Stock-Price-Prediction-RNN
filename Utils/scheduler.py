
def simple_rnn_scheduler(epoch, lr):
    if epoch < 20:
        return lr
    elif epoch < 30:
        return 0.0003
    elif epoch < 40:
        return 0.0001
    else:
        return 0.00005


def lstm_scheduler(epoch, lr):
    if epoch < 40:
        return lr
    elif epoch < 50:
        return 0.0003
    elif epoch < 60:
        return 0.0001
    else:
        return 0.00005


def gru_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return 0.0003
    elif epoch < 30:
        return 0.0001
    else:
        return 0.00005
