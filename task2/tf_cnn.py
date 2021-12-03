from qlib.model.base import Model
from typing import Text, Union
from qlib.data.dataset import Dataset
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from qlib.data.dataset.handler import DataHandlerLP
from tensorflow.keras import models


class CnnModel(Model):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.activation = 'relu'
        self.filters = 32
        self.kernel_size = 3
        self.dense1_par = 128
        self.dense2_par = 10
        self.epochs = 5
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def fit(self, dataset: Dataset):
        self.dense2_par = 1
        self.loss = tf.keras.losses.MeanSquaredError()
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        self.fit_model(x_train, y_train, x_valid, y_valid)

    def fit_model(self, x_train, y_train, x_valid, y_valid):
        model = models.Sequential([
            Conv2D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation),
            Flatten(),
            Dense(self.dense1_par),
            Dense(self.dense2_par)

        ])
        model.compile(optimizer='adam',
                      loss=self.loss,
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=self.epochs, validation_data=(x_valid, y_valid), verbose=2)
        model.summary()
        model.save('MyModel')

    def predict(self, dataset: Dataset, segment: Union[Text, slice] = "test") -> object:
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        model_trained = models.load_model('MyModel')
        probability_model = tf.keras.Sequential([model_trained,
                                                 tf.keras.layers.Softmax()])
        predictions = probability_model.predict(x_test)
        return predictions
