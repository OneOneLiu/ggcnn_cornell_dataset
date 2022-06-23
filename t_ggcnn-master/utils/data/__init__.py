def get_dataset(dataset_name):
    if dataset_name == 'cornell':
        from .cornell_data import CornellDataset
        return CornellDataset
    elif dataset_name == 'jacquard':
        from .jacquard_data import JacquardDataset
        return JacquardDataset
    elif dataset_name == 'jacquard_t':
        from .jacquard_data_t import JacquardDataset
        return JacquardDataset
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))