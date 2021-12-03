import tf_cnn
import tensorflow as tf
from tensorflow import keras

# This code is based on Fukushima1980

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

cnn = tf_cnn.CnnModel()
cnn.fit_model(x_train, y_train, x_test, y_test)
