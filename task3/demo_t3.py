import workflow
from qlib.config import REG_CN

if __name__ == "__main__":
    provider = "E:/qilb研究/.qlib_data/cn_data"
    workflow.init(provider=provider, region=REG_CN)
    data_handler_config = workflow.data_handler_config(start_time='2008-01-01', end_time='2020-08-01',
                                                       fit_start_time='2008-01-01', fit_end_time='2014-12-31',
                                                       instruments='csi300')
    model_kwargs = {
        'estimator': 'lasso',
        'alpha': 0.5,
    }
    task_model = workflow.task_model(model_class='LinearModel', module_path='qlib.contrib.model.linear',
                                     m_kwargs=model_kwargs)
    dataset_segments = workflow.create_segments(train_start='2008-01-01', train_end='2014-12-31',
                                                valid_start='2015-01-01', valid_end='2016-12-31',
                                                test_start='2017-01-01', test_end='2020-08-01')
    dataset_kwargs = workflow.create_dataset_kwarg(handler_class='Alpha158', kwargs=data_handler_config,
                                                   segment=dataset_segments)
    task_dataset = workflow.task_dataset(dataset_class='DatasetH', module_path='qlib.data.dataset',
                                         d_kwargs=dataset_kwargs)
    task = workflow.create_task(task_model, task_dataset)
    model, dataset = workflow.prepare(task)

    history = workflow.train_model(task, model, dataset)
    analysis = workflow.create_port_analysis_config(freq='day', model=model, dataset=dataset, benchmark="SH000300")
    workflow.backtest(analysis, history, dataset)
