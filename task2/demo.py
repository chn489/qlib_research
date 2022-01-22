import tensorflow_cnn
import qlib
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config

provider_uri = "E:/qilb研究/.qlib_data/cn_data"  # target_dir
if __name__ == "__main__":
    qlib.init(provider_uri=provider_uri, region=REG_CN, expression_cache=None, dataset_cache=None)
    market = "csi300"
    benchmark = "SH000300"
    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": market,
    }
    task = {
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2008-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2017-01-01", "2020-08-01"),
                },
            },
        },
    }
    model = tensorflow_cnn.CNN(epochs=10)
    dataset = init_instance_by_config(task["dataset"])
    model.fit(dataset)
    prediction = model.predict(dataset)
    print(prediction)
