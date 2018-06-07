import numpy as np
from keras.layers import LSTM
from keras.models import Sequential, load_model

from phased_lstm_keras.PhasedLSTM import PhasedLSTM


def main():
    n_neurons = 32
    input_shape = (100, 2)
    output_size = 10
    X = np.random.random((n_neurons, *input_shape))
    Y = np.random.random((n_neurons, output_size))

    model_lstm = Sequential()
    model_lstm.add(LSTM(output_size, input_shape=input_shape))
    model_lstm.compile('rmsprop', 'mse')
    model_lstm.save('model_lstm.h5')
    model_lstm = load_model('model_lstm.h5')
    model_lstm.summary()

    model_phasedlstm = Sequential()
    model_phasedlstm.add(PhasedLSTM(output_size, input_shape=input_shape))
    model_phasedlstm.compile('rmsprop', 'mse')
    model_phasedlstm.save('model_plstm.h5')
    model_phasedlstm = load_model('model_plstm.h5')
    model_phasedlstm.summary()

    model_lstm.fit(X, Y)
    model_phasedlstm.fit(X, Y)


if __name__ == "__main__":
    main()
