task2的任务是基于指定论文复现代码，并搭建以CNN为主干网络的模型。  
我将它分成了2个部分：第一个是基于tensorflow搭建的卷积神经网络API（即tf_cnn)，第二个是使用tf_cnn复现Fukushima1980的demo（数据集为mnist)。  
如果您希望使用自己的数据进行训练，那么使用tf_cnn的fit_model（）即可训练模型；如果希望使用qlib准备好的数据，那么请使用tf_cnn的fit（）。
