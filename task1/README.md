task1的任务是跑通程序，分为数据准备和搭建workflow两个部分。  
其中搭建workflow时，如果程序在本地的pycharm运行，需要将代码放在if __name__ == "__main__":下面，否则会报错；  
另外初始化时，需要在qlib.init()中增加参数expression_cache=None, dataset_cache=None，否则会收到来自redis的报错。  
task1的代码参照了官网给出的example,在准备好数据并且完成初始化之后，使用LightGBM训练模型并且回测。
