import torch.nn as nn


class ConvHeadPooling(nn.Module):
    """Adapted from
    https://github.com/naver-ai/pit/blob/9d97a62e6a2a72a86685003998fcae700f952e18/pit.py#L54-L69
    """
    def __init__(self, in_feature: int, out_feature: int,
                 stride: int):
        super(ConvHeadPooling, self).__init__()

        # Depth wise convolution
        self.conv = nn.Conv2d(
            in_channels=in_feature, out_channels=out_feature,
            kernel_size=stride + 1, padding=stride // 2,
            stride=stride, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):
        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token
