from qlib.model.base import Model
from typing import Text, Union
from qlib.data.dataset import DatasetH, Dataset
from tensorflow.keras import layers, models, regularizers, optimizers, losses, metrics
from qlib.data.dataset.handler import DataHandlerLP
import pandas as pd

def cov(x, y):
    x_bar = x.mean()
    y_bar = y.mean()
    cov_xy = 0
    for i in range(0, len(x)):
        cov_xy += (x[i] - x_bar) * (y[i] - y_bar)
    cov_xy = cov_xy / len(x)
    return cov_xy


def pearson_corr(x, y):
    x_std = x.std()
    y_std = y.std()
    cov_xy = cov(x, y)
    corr = cov_xy / (x_std * y_std)
    return corr

def dropz(df):
    idx = []
    for c in df.columns:
        if df[c].values.sum() == 0:
            idx.append(c)
    return df.drop(idx, axis=1)

class CNN(Model):
    def __init__(self,
                 epochs=15,
                 model=None,
                 dtrain=None,
                 dvalid=None,
                 dtest=None,
                 loss=losses.MeanSquaredError()):
        super(CNN, self).__init__()
        self.xgb = None
        self.cnn_model = model
        self.cnn_2 = None
        self.dtrain = dtrain
        self.dvalid = dvalid
        self.dtest = dtest
        self.epochs = epochs
        self.loss = loss

    def fit(self, dataset: Dataset):
        if self.dtrain is None:
            dtrain, dvalid = dataset.prepare(
                ["train", "valid"],
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )
        self.dtrain = dtrain.fillna(method='ffill', axis=1).fillna(0)
        self.dtrain = dropz(self.dtrain)
        self.dvalid = dvalid.fillna(method='ffill', axis=1).fillna(0)
        self.dvalid = dropz(self.dvalid)

        x_train, y_train = self.dtrain["feature"], self.dtrain["label"]
        #x_train = dropz(x_train)

        x_valid, y_valid = self.dvalid["feature"], self.dvalid["label"]
        #x_valid = dropz(x_valid)

        x_train = x_train.values.reshape(-1, x_train.shape[1], 1)
        # print(x_train.shape)
        x_valid = x_valid.values.reshape(-1, x_valid.shape[1], 1)

        if self.cnn_model is None:
            rmse = metrics.RootMeanSquaredError('rmse')
            self.cnn_model = models.Sequential([
                layers.Conv1D(64, kernel_size=8, strides=3, activation='leaky_relu'),
                layers.Dense(64, activation='swish'),
                layers.MaxPooling1D(pool_size=2, strides=2),

                layers.Conv1D(128, kernel_size=3, strides=2, activation='leaky_relu'),
                layers.Conv1D(128, kernel_size=2, strides=1, activation='leaky_relu'),
                layers.LayerNormalization(),
                layers.Dense(128, activation='swish'),
                layers.MaxPooling1D(pool_size=2, strides=1),

                layers.Conv1D(256, kernel_size=2, strides=1, activation='leaky_relu'),
                layers.Conv1D(256, kernel_size=2, strides=1, activation='leaky_relu'),
                layers.Dense(256, activation='swish', kernel_regularizer=regularizers.l2(0.0003)),
                layers.MaxPooling1D(pool_size=2, strides=1),

                layers.Flatten(),
                layers.Dense(512, activation='swish', kernel_regularizer=regularizers.l2(0.0005)),
                layers.Dropout(0.5),
                layers.Dense(1)
            ])
            self.cnn_model.compile(optimizer=optimizers.Adam(0.0005),
                                   loss=self.loss, metrics=[rmse, 'mape'])
            self.cnn_model.fit(x_train, y_train, epochs=self.epochs, batch_size=256, validation_data=(x_valid, y_valid),
                               verbose=1)
        self.cnn_model.summary()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test") -> object:
        if self.dtest is not None:
            x_test, y_test =self.dtest['feature'], self.dtest['label']
        else:
            self.dtest = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
            self.dtest = dropz(self.dtest)
            x_test, y_test = self.dtest['feature'], self.dtest['label']

        x_test = x_test.fillna(method='ffill', axis=1).fillna(0)
        y_test = y_test.fillna(method='ffill', axis=1).fillna(0)

        index = x_test.index
        values = x_test.values.reshape(-1, x_test.shape[1], 1)
        pred = list(self.cnn_model.predict(values))
        pred = pd.DataFrame(pred, columns=['score'],index=index)
        print("pearson_corr:", pearson_corr(pred.values, y_test.values))
        return pred
