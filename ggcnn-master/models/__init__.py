def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_f':
        from .ggcnn2_filter import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_prob':
        from .ggcnn2_prob import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_prob_v2':
        from .ggcnn2_prob_v2 import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_prob_v3':
        from .ggcnn2_prob_v3 import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_patch':
        from .ggcnn2_patch import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_4f':
        from .ggcnn2_4filter import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_4prob':
        from .ggcnn2_prob_4_filter import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_4patch':
        from .ggcnn2_patch_4_filter import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_patch_v2':
        from .ggcnn2_patch_v2 import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_patch_v3':
        from .ggcnn2_patch_v3 import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_patch_v4':
        from .ggcnn2_patch_v4 import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_patch_v6':
        from .ggcnn2_patch_v6 import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_patch_v7':
        from .ggcnn2_patch_v7 import GGCNN2
        return GGCNN2
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
