
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import cifar10
import numpy as np

import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[np.where(y_train == 1)[0], :, :, :]
x_test = x_test[np.where(y_test == 1)[0], :, :, :]

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


x_train_n = x_train + 0.5 * np.random.normal(loc=0.0, scale=0.4, size=x_train.shape)
x_test_n = x_test + 0.5 * np.random.normal(loc=0.0, scale=0.4, size=x_test.shape)

x_train_n = np.clip(x_train_n, 0.,1.)
x_test_n = np.clip(x_test_n, 0.,1.)

inp_img = Input(shape=(32,32,3))
img = Conv2D(32, (3,3),activation='relu', padding='same')(inp_img)
img = MaxPooling2D((2,2),padding='same')(img)
img = Conv2D(32, (3,3),activation='relu', padding='same')(img)
img = UpSampling2D((2,2))(img)
decoded = Conv2D(3, (3,3), activation='sigmoid', padding='same')(img)
autoencoder = Model(inp_img, decoded)

# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#   
# tensorboard = TensorBoard(log_dir='tmp/cifar10',
#                           histogram_freq=0, write_graph=True, write_images=True)
# model_saver = ModelCheckpoint(
#     filepath='tmp/cifar10/cpt',
#     verbose=0,
#     period=2)
#  
# autoencoder.fit(x_train_n, x_train,
#                 epochs=50,
#                 batch_size=64,
#                 shuffle=True,
#                 validation_data=(x_test_n,x_test),
#                 callbacks=[tensorboard, model_saver])




Model.load_weights(autoencoder, 'tmp/cifar10/cpt')
  
predicted = autoencoder.predict([x_train_n[:1]])
plt.subplot(121)
plt.imshow(x_train_n[0])
plt.subplot(122)
plt.imshow(predicted[0])
plt.show()




