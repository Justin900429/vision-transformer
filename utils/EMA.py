"""Adapted from
https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4
"""
class EMA:
    def __init__(self, model, ratio):
        self.model = model
        self.ratio = ratio
        self.shadow = dict()

    def register(self):
        for name, param in self.model.named_parameters:
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Something went wrong in updating model. " \
                                            f"{name} not in the backup list"
                new_average = (1.0 - self.ratio) * param.data + self.ratio * self.shadow[name]
                self.shadow[name] = new_average.clone()

    @property
    def shadow(self):
        return self.ratio
