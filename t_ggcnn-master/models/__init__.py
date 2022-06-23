def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_prob':
        from .ggcnn2_prob import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_patch':
        from .ggcnn2_patch import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_m_patch':
        from .ggcnn2_multi_patches import GGCNN2
        return GGCNN2
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
