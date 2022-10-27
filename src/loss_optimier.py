import torch.optim as optim
from lib.pytorch_lib.metrics_learning.metric_strategy_gpu import ArcFaceLossAdaptiveMargin



def optimizer_fun(model_parameter,lr):
    optimizer =optim.Adam(model_parameter,lr)
    return optimizer


# loss func
class criterion_cla:
    def __init__(self, margin, s, out_dim):
        self.margin = margin
        self.s = s
        self.out_din = out_dim

    def arc(self):
        arc = ArcFaceLossAdaptiveMargin(self.margin, self.out_din)
        return arc

    def loss(self, logits_m, target):
        arc = self.arc()
        loss_m = arc(logits_m, target,self.out_din)
        return loss_m

