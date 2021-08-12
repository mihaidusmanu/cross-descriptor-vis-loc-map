from lib.networks import MLP


def create_network_for_feature(feature, config, use_cuda):
    use_bn = config['use_bn']
    emb_dim = config['emb_dim']
    emb_l2_norm = config['emb_l2_norm']
    emb_last_activation = config['emb_last_activation']
    descriptor_dim = config[feature]['descriptor_dim']
    hidden_dims = config[feature]['hidden_dims']
    l2_norm = config[feature]['l2_norm']
    last_activation = config[feature]['last_activation']
    
    # Define encoder.
    encoder = MLP(
        num_channels=[descriptor_dim] + hidden_dims + [emb_dim], use_cuda=use_cuda,
        use_bn=use_bn, last_activation=emb_last_activation, l2_norm=emb_l2_norm
    )

    # Define decoder.
    decoder = MLP(
        num_channels=[emb_dim] + hidden_dims[:: -1] + [descriptor_dim], use_cuda=use_cuda,
        use_bn=use_bn, last_activation=last_activation, l2_norm=l2_norm
    )

    return encoder, decoder