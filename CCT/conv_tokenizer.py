import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, conv_args, pooling_args,
                 activation, use_pool):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(**conv_args)
        self.activation = activation() if activation is not None else nn.Identity()
        self.pooling = nn.MaxPool2d(**pooling_args) if use_pool else nn.Identity

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pooling(x)

        return x


class ConvolutionTokenizer(nn.Module):
    def __init__(self, kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 num_layers=1,
                 in_channels=3,
                 out_channels=64,
                 hidden_channels=64,
                 activation=None,
                 use_pool=True,
                 conv_bias=False):
        super(ConvolutionTokenizer, self).__init__()
        num_filters = [in_channels] + \
                      [hidden_channels for _ in range(num_layers - 1)] + \
                      [out_channels]

        self.conv_layers = nn.Sequential(
            *[
                ConvBlock(
                    conv_args=dict(
                        in_channels=num_filters[idx],
                        out_channels=num_filters[idx + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=conv_bias
                    ),
                    pooling_args=dict(
                        kernel_size=pooling_kernel_size,
                        stride=pooling_stride,
                        padding=pooling_padding
                    ),
                    use_pool=use_pool,
                    activation=activation
                )
                for idx in range(num_layers)
            ],
            nn.Flatten(start_dim=2)
        )
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(layers):
        if isinstance(layers, nn.Conv2d):
            nn.init.kaiming_normal_(layers.weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        """Output the length of the tokenized image"""
        return self.forward(
            torch.zeros((1, n_channels, height, width))
        ).size(1)

    def forward(self, x):
        # Before: Shape of x: (batch_size, in_channels, width, height)
        # After: Shape of x: (batch_size, out_channels, num_patches)
        x = self.conv_layers(x)

        # Shape of x: (batch_size, num_patches, out_channels)
        # out_channels can be considered as embed_dim
        # which should be input into the transformers
        x = x.transpose(1, 2)

        return x
