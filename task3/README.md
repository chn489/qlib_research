task3的主要目的是将qlib技术文档中的各个部分串通起来，一遍实现构造数据、特征分析、特征预处理、训练模型、回测模型、回测结果分析和模型上线等功能。  
代码包括一个名为workflow的API，可以实现例如数据获取、初始化、配置config文件和task、训练模型和回测模型等功能；  
还有一个demo作为使用例。这里选用了线性回归中的LASSO回归，因子为Alpha158,通过使用workflow的函数，只需要输入对应的参数就可以自动完成数据读取、初始化、创建task并且准备数据和训练模型，
创建port_analysis_config并且回测等功能。  


对代码的一些说明：由于qlib的一些bug(例如qlib.contrib.report.analysis_position第19行中的ic和_rank_ic_需要改成列表形式，否则会报错：ValueError:If using all scalar values, you must pass an index、来自同一份文件第57行_ic_df.index=_ic_df.index.strftime()，运行回测和分析的代码时会报错：AttributeError:'RangeIndex' object has no attribute strftime),有些地方需要手动修改qlib的库文件，有些地方暂时不知道如何修改（例如第二个bug），只能放弃部分功能（这里放弃了画图功能）
