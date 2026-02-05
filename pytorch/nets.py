"""
Create neural networks.
"""

import math

import torch.nn as nn
from dlk.nets.autoencoder import Autoencoder
from dlk.nets.conv1d import ConvNet, ConvResNet
from dlk.nets.efficientnet import EfficientNet1D
from dlk.nets.mlp import MLPNet, MLPNet_MultIn, MLPResNet
from dlk.nets.transformer1d import ChannelWiseTransformerNet, TransformerNet
from dlk.nets.unet import DecoderNet1d_2021 as DecoderConvNet
from dlk.nets.unet import EncoderNet1d_2021 as EncoderConvNet
from dlk.nets.unet import UNet1d_2021 as UNet

from utils import NetworkType

# --------------------------------------
# Neural Networks for Inverse Maps
# --------------------------------------


def _get_activation(name):
    if "relu" == name.casefold():
        return nn.ReLU()
    elif "gelu" == name.casefold():
        return nn.GELU()
    elif "silu" == name.casefold() or "swish" == name.casefold():
        return nn.SiLU()
    else:
        raise ValueError("Unknown name for activation function: " + name)


def _get_conv1d_size(in_length, kernel, stride=1, padding=0, dilation=1):
    return int((in_length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


def _create_MLPNet(input_channels, input_size, output_size, net_params, logger):
    logger.info(
        f"create MLPNet_MultIn({input_channels * input_size}, {output_size}, ...)"
    )
    return MLPNet_MultIn(
        input_channels * input_size,
        output_size,
        hidden_layers_sizes=net_params["dense_layer_sizes"],
        hidden_layers_activation=_get_activation(net_params["activation_fn"]),
        use_dropout=net_params.get("dropout", False),
        output_layer_activation=None,
    )


def _create_MLPResNet(
    input_channels,
    input_size,
    output_size,
    net_params,
    logger,
    hidden_input_size=0,
):
    activation_fn = _get_activation(net_params["activation_fn"])
    embed_size = net_params.get("embedding_size", 1)
    flattened_input_size = (
        input_channels * input_size,
        net_params["residual_blocks_sizes"][0][0],
    )
    # configure hidden inputs to residual blocks
    if hidden_input_size:
        residual_blocks_sizes = list(net_params["residual_blocks_sizes"])
        if isinstance(hidden_input_size, int):
            hidden_input_size = [hidden_input_size] * len(residual_blocks_sizes)
        assert len(hidden_input_size) == len(residual_blocks_sizes)
        for i in range(len(residual_blocks_sizes)):
            residual_blocks_sizes[i][0] += hidden_input_size[i]
    else:
        residual_blocks_sizes = net_params["residual_blocks_sizes"]
    # create net
    logger.info(f"create MLPNet_MultIn({flattened_input_size}, {output_size}, ...)")
    return MLPResNet(
        flattened_input_size,
        output_size,
        embedding_size=embed_size,
        attention_blocks_n_heads=net_params.get("attention_layers_n_heads"),
        attention_blocks_activation_size=embed_size * 4,
        attention_blocks_activation=activation_fn,
        residual_blocks_sizes=net_params["residual_blocks_sizes"],
        residual_blocks_activation=activation_fn,
        use_dropout=net_params.get("dropout", False),
        output_layer_activation=None,
    )


def _create_convNet(input_channels, input_size, output_size, net_params, logger):
    activation_fn = _get_activation(net_params["activation_fn"])
    use_dropout = net_params.get("dropout", False)
    kernel = net_params.get("conv_layer_kernel", 3)
    stride = net_params.get("conv_layer_stride", 2)
    padding = net_params.get("conv_layer_padding", 0)
    n_conv_layers = len(net_params["conv_layer_sizes"])
    # set parameters of convolution layers
    hidden_conv_layers_kernels = n_conv_layers * [kernel]
    hidden_conv_layers_kwargs = {"stride": stride, "padding": padding}
    # calculate length of features after convolutional layers
    n_channels = input_channels * net_params["conv_layer_sizes"][-1]
    n_features = input_size
    for _ in range(n_conv_layers):
        n_features = _get_conv1d_size(n_features, kernel, stride, padding)
    flattened_input_size = n_channels * n_features
    # create net
    logger.info(f"create ConvNet({input_channels}, ..., {output_size}, ...)")
    return ConvNet(
        # convolutional layers
        input_channels,
        hidden_conv_layers_channels_mult=net_params["conv_layer_sizes"],
        hidden_conv_layers_kernels=hidden_conv_layers_kernels,
        hidden_conv_layers_activation=activation_fn,
        hidden_conv_layers_kwargs=hidden_conv_layers_kwargs,
        # dense layers
        hidden_dense_input_size=flattened_input_size,
        hidden_dense_layers_sizes=net_params["dense_layer_sizes"],
        hidden_dense_layers_activation=activation_fn,
        # output layer
        output_size=output_size,
        output_layer_activation=None,
        # other
        use_dropout=use_dropout,
    )


def _create_convResNet(
    input_channels,
    input_size,
    output_size,
    net_params,
    logger,
    mlb_kwargs={},
    mlp_block_hidden_input_size=0,
):
    activation_fn = _get_activation(net_params["activation_fn"])
    use_dropout = net_params.get("dropout", False)
    kernel = net_params.get("conv_layer_kernel", 3)
    stride = net_params.get("conv_layer_stride", 2)
    padding = net_params.get("conv_layer_padding", 1)
    # padding_mode = net_params.get("conv_layer_padding_mode", "replicate")
    n_conv_layers = len(net_params["conv_layer_sizes"])
    # set parameters of convolution block
    mlb_kwargs = dict(mlb_kwargs)
    mlb_kwargs.update(
        {
            "padding": 1,
            "padding_mode": "replicate",
            "stride": 2,
        }
    )
    conv_resnet_params = {
        "channels_mult": net_params["conv_layer_sizes"],
        "kernels": n_conv_layers * [kernel],
        "activation": activation_fn,
        "use_dropout": use_dropout,
        "mlb_kwargs": mlb_kwargs,
    }
    # calculate length of features after convolutional layers
    n_channels = input_channels * net_params["conv_layer_sizes"][-1]
    n_features = input_size
    for _ in range(n_conv_layers):
        n_features = _get_conv1d_size(n_features, kernel, stride, padding)
    flattened_input_size = (
        n_channels * n_features,
        net_params["residual_blocks_sizes"][0][0],
    )
    # configure hidden inputs to residual blocks
    if mlp_block_hidden_input_size:
        residual_blocks_sizes = list(net_params["residual_blocks_sizes"])
        if isinstance(mlp_block_hidden_input_size, int):
            mlp_block_hidden_input_size = [mlp_block_hidden_input_size] * len(
                residual_blocks_sizes
            )
        assert len(mlp_block_hidden_input_size) == len(residual_blocks_sizes)
        for i in range(len(residual_blocks_sizes)):
            residual_blocks_sizes[i][0] += mlp_block_hidden_input_size[i]
    else:
        residual_blocks_sizes = net_params["residual_blocks_sizes"]
    # set parameters of MLP block
    mlp_resnet_params = {
        "input_size": flattened_input_size,
        "output_size": output_size,
        "residual_blocks_sizes": residual_blocks_sizes,
        "residual_blocks_activation": activation_fn,
        "use_dropout": use_dropout,
        "output_layer_activation": None,
    }
    # create net
    logger.info(f"create ConvResNet({input_channels}, ...)")
    return ConvResNet(
        input_channels,
        conv_resnet_params=conv_resnet_params,
        mlp_resnet_params=mlp_resnet_params,
    )


def _create_efficientNet(input_channels, input_size, output_size, net_params, logger):
    use_dropout = net_params.get("dropout", False)
    logger.info(
        f"create EfficientNet1D({input_channels}, {input_size}, {output_size}, ...)"
    )
    return EfficientNet1D(
        input_channels=input_channels,
        input_length=input_size,
        num_classes=output_size,
        dropout_connect=use_dropout if use_dropout else 0.0,
        dropout_head=use_dropout if use_dropout else 0.0,
    )


def _create_transformerNet(input_channels, input_size, output_size, net_params, logger):
    patch_size = net_params.get("patch_size", input_size // 10)
    embed_size = net_params.get("embedding_size")
    attn_n_heads = net_params.get("attention_layers_n_heads")
    use_dropout = net_params.get("dropout", False)
    if 1 == input_channels:
        logger.info(f"create TransformerNet({input_size}, {output_size}, ...)")
        return TransformerNet(
            input_seq_size=input_size,
            output_size=output_size,
            patch_size=patch_size,
            embedding_size=embed_size,
            attn_n_heads=attn_n_heads,
            dropout=use_dropout if use_dropout else 0.0,
        )
    else:
        logger.info(
            f"create ChannelWiseTransformerNet({input_channels}, {input_size}, {output_size}, ...)"
        )
        return ChannelWiseTransformerNet(
            input_channels=input_channels,
            input_seq_size=input_size,
            output_size=output_size,
            patch_size=patch_size,
            embedding_size=embed_size,
            attn_n_heads=attn_n_heads,
            dropout=use_dropout if use_dropout else 0.0,
        )


def create_network(params, logger):
    # get network options
    net_params = params["net"]
    net_type = NetworkType.get_from_name(net_params["type"])
    logger.info(f"Network type: {net_params['type']}, key: {net_type}")
    # set input and output sizes
    assert 2 == len(params["data"]["num_features"])
    input_channels, input_size = params["data"]["num_features"]
    output_size = params["data"]["num_targets"]
    # create network
    if NetworkType.MLPNET == net_type:
        net = _create_MLPNet(
            input_channels, input_size, output_size, net_params, logger
        )
    elif NetworkType.MLPRESNET == net_type:
        net = _create_MLPResNet(
            input_channels, input_size, output_size, net_params, logger
        )
    elif NetworkType.CONVNET == net_type:
        net = _create_convNet(
            input_channels, input_size, output_size, net_params, logger
        )
    elif NetworkType.CONVRESNET == net_type:
        net = _create_convResNet(
            input_channels, input_size, output_size, net_params, logger
        )
    elif NetworkType.EFFICIENTNET == net_type:
        net = _create_efficientNet(
            input_channels, input_size, output_size, net_params, logger
        )
    elif NetworkType.TRANSFORMERNET == net_type:
        net = _create_transformerNet(
            input_channels, input_size, output_size, net_params, logger
        )
    else:
        raise NotImplementedError(f"Type {net_type} is not implemented")
    # return network
    return net


# --------------------------------------
# Autoencoder Networks
# --------------------------------------


def create_enc_dec(params, logger):
    e_net_params = params["e_net"]
    d_net_params = params["d_net"]
    e_net_type = NetworkType.get_from_name(e_net_params["type"])
    d_net_type = NetworkType.get_from_name(d_net_params["type"])
    logger.info(f"Encoder type: {e_net_params['type']}, key: {e_net_type}")
    logger.info(f"Decoder type: {d_net_params['type']}, key: {d_net_type}")
    input_size = math.prod(params["data"]["num_features"])
    latent_size = params["data"]["latent_size"]
    output_size = input_size
    input_channels = params["data"]["num_features"][0]
    latent_channels = params["data"]["num_features"][0]
    output_channels = params["data"]["num_features"][0]
    # create encoder network
    if NetworkType.MLPNET == e_net_type:
        e_net = MLPNet(
            input_size,
            latent_size,
            hidden_layers_sizes=e_net_params["dense_layer_sizes"],
            hidden_layers_activation=_get_activation(e_net_params["activation_fn"]),
            use_dropout=e_net_params["dropout"],
        )
    elif NetworkType.MLPRESNET == e_net_type:
        e_net = MLPResNet(
            input_size,
            latent_size,
            residual_blocks_sizes=e_net_params["residual_blocks_sizes"],
            residual_blocks_activation=_get_activation(e_net_params["activation_fn"]),
            use_dropout=e_net_params["dropout"],
        )
    elif NetworkType.CONVNET == e_net_type:
        e_net = EncoderConvNet(
            input_channels,
            latent_channels,
            internal_channels=e_net_params["conv_internal_channels"],
            channel_mult=e_net_params["conv_channel_mult"],
        )
    else:
        raise NotImplementedError()
    # create decoder network
    if NetworkType.MLPNET == d_net_type:
        d_net = MLPNet(
            latent_size,
            output_size,
            hidden_layers_sizes=d_net_params["dense_layer_sizes"],
            hidden_layers_activation=_get_activation(d_net_params["activation_fn"]),
            use_dropout=d_net_params["dropout"],
            output_layer_activation=None,
        )
    elif NetworkType.MLPRESNET == d_net_type:
        d_net = MLPResNet(
            latent_size,
            output_size,
            residual_blocks_sizes=d_net_params["residual_blocks_sizes"],
            residual_blocks_activation=_get_activation(d_net_params["activation_fn"]),
            use_dropout=d_net_params["dropout"],
        )
    elif NetworkType.CONVNET == d_net_type:
        d_net = DecoderConvNet(
            latent_channels,
            output_channels,
            internal_channels=d_net_params["conv_internal_channels"],
            channel_mult=d_net_params["conv_channel_mult"],
        )
    else:
        raise NotImplementedError()
    # return networks
    return e_net, d_net


def create_ae(params, logger):
    e_net, d_net = create_enc_dec(params, logger)

    def _output_transf(y):
        return y.reshape(-1, *params["data"]["num_features"])

    return Autoencoder(e_net, d_net, output_layer_transformation=_output_transf)


# --------------------------------------
# UNet
# --------------------------------------


def create_unet(params, logger):
    logger.info(f"create UNet(1, 1, ...)")
    return UNet(
        1,
        1,
        internal_channels=params["net"]["conv_internal_channels"],
        channel_mult=params["net"]["conv_channel_mult"],
    )
