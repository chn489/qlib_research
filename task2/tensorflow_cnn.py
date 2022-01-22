from qlib.model.base import Model
from typing import Text, Union
from qlib.data.dataset import DatasetH, Dataset
from tensorflow.keras import layers, models, regularizers, optimizers, losses
from qlib.data.dataset.handler import DataHandlerLP
import pandas as pd


class CNN(Model):
    def __init__(self,
                 epochs=15,
                 n_factors=158,
                 loss=losses.MeanSquaredError()):
        super(CNN, self).__init__()
        self.cnn_model = None
        self.n_factors = n_factors
        self.epochs = epochs
        self.loss = loss

    def fit(self, dataset: Dataset):
        dtrain, dvalid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )

        dtrain = dtrain.fillna(method='bfill', axis=0).fillna(0)
        dvalid = dvalid.fillna(method='bfill', axis=0).fillna(0)

        x_train, y_train = dtrain["feature"], dtrain["label"]
        x_valid, y_valid = dvalid["feature"], dvalid["label"]

        # print(x_valid, '\n', y_valid)

        x_train = x_train.values.reshape(-1, self.n_factors, 1)
        x_valid = x_valid.values.reshape(-1, self.n_factors, 1)

        self.cnn_model = models.Sequential([
            layers.Conv1D(filters=32, kernel_size=3, kernel_regularizer=regularizers.l2(0.001), activation='relu',
                          padding='same'),
            layers.MaxPooling1D(data_format='channels_first'),
            layers.Conv1D(filters=64, kernel_size=3, kernel_regularizer=regularizers.l2(0.001), activation='relu',
                          padding='same'),
            layers.MaxPooling1D(data_format='channels_first'),
            layers.Conv1D(filters=128, kernel_size=3, kernel_regularizer=regularizers.l2(0.001), activation='relu',
                          padding='same'),
            layers.MaxPooling1D(data_format='channels_first'),
            # layers.Conv1D(filters=256, kernel_size=3, activation='relu',kernel_regularizer=regularizers.l2(0.001), padding='same'),
            # layers.MaxPooling1D(data_format='channels_first'),
            layers.Flatten(),
            # layers.Dense(256, activation='relu',kernel_initializer='random_normal', kernel_regularizer=regularizers.l2(0.001)),
            # layers.BatchNormalization(),
            layers.Dense(128, activation='relu', kernel_initializer='random_normal',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            # layers.Dense(64, activation='relu',kernel_initializer='random_normal',kernel_regularizer=regularizers.l2(0.001)),
            # layers.BatchNormalization(),
            layers.Dense(32, activation='relu', kernel_initializer='random_normal',
                         kernel_regularizer=regularizers.l2(0.001)),
            # layers.BatchNormalization(),
            layers.Dense(1)
        ])
        self.cnn_model.compile(optimizer=optimizers.Adam(0.001, clipnorm=1),
                               loss=self.loss, )
        self.cnn_model.fit(x_train, y_train, epochs=self.epochs, batch_size=512, validation_data=(x_valid, y_valid),
                           verbose=1)
        self.cnn_model.summary()
        self.cnn_model.save('MyModel')

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test") -> object:
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I).fillna(method='bfill',
                                                                                                 axis=0).fillna(0)
        index = x_test.index
        values = x_test.values.reshape(-1, self.n_factors, 1)
        # print(values, '\n')
        pred = list(self.cnn_model.predict(values))
        # print(pred, '\n', index)
        return pd.Series(pred, index=index)
