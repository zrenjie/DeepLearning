
import tflearn
from tflearn.data_utils import pad_sequences, to_categorical
from tflearn.datasets import imdb
from _nsis import out
from tensorflow.contrib.slim.python.slim import learning

train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
X_train, Y_train = train
X_test, Y_test = train

X_train = pad_sequences(X_train, maxlen=100, value=0.1)
X_test = pad_sequences(X_test, maxlen=100, value=0.1)

Y_train = to_categorical(Y_train, nb_classes=2)
Y_test = to_categorical(Y_test, nb_classes=2)

RNN = tflearn.input_data([None, 100])
RNN = tflearn.embedding(RNN, input_dim=10000, output_dim=128)

RNN = tflearn.lstm(RNN, 128, dropout=0.8)
RNN = tflearn.fully_connected(RNN, 2, activation='softmax')
RNN = tflearn.regression(RNN, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

model = tflearn.DNN(RNN, tensorboard_verbose=0)
model.fit(X_train, Y_train, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=32)


if __name__ == '__main__':
    print('Length of train samples: ', len(X_train) )
    print('Length of test samples: ', len(X_test))
    print(X_train[:5])
    print(Y_train[:5])