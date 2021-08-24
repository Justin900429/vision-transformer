import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothing(nn.Module):
    """Label smoothing

    Attributes
    ----------
    smoothing_rate: float
        Rate for the smoothing, should be bounded in (0, 1)

    Notes
    -----
    Examples:
    > y = (0, 0, 0, 0, 1)
    > smoothing_rate = 0.1
    > y_new = (0.02, 0.02, 0.02, 0.02, 0.92)

    """
    def __init__(self,
                 smoothing_rate: float = 0.1,
                 mode: str = "mean"):
        super(LabelSmoothing, self).__init__()
        assert mode in ["mean", "sum", "none"], f"Not supported mode for {mode}." \
                                                f" Should be in ['mean', 'sum', 'none']"
        self.mode = mode
        self.smoothing_rate = smoothing_rate

    def forward(self, predict, target):
        # Shape of predict: (batch_size, num_classes)
        log_probs = F.log_softmax(predict, dim=-1)

        # Vanilla cross entropy loss
        NLL_loss = -log_probs.gather(dim=-1,
                                     index=target.unsqueeze(-1))
        NLL_loss = NLL_loss.squeeze(1)

        # Add the smoothing loss to it
        smoothing_loss = -log_probs.mean(dim=-1)
        total_loss = (1 - self.smoothing_rate) * NLL_loss + self.smoothing_rate * smoothing_loss

        if self.mode == "mean":
            return total_loss.mean()
        elif self.mode == "sum":
            return total_loss.sum()
        else:
            return total_loss
