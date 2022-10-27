import torch.nn as nn
import geffnet
from resnest.torch import resnest101
from lib.pytorch_lib.metrics_learning.metric_strategy_gpu import Swish_module,ArcMarginProduct_subcenter



# Effnet_GLDv2
class Effnet_GLDv2(nn.Module):
     def __init__(self, enet_type, out_dim):
        super(Effnet_GLDv2, self).__init__()
        self.enet = geffnet.create_model(enet_type.replace('-', '_'), pretrained=True)
        feature = self.enet.classifier.in_features
        self.feat = nn.Linear(feature, 512)
        self.switch = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.feat(x)
        x = self.switch(x)
        logits_m = self.metric_classify(x)
        return logits_m


class ResNet101_GLDv2(nn.Module):
    def __init__(self, enet_tpye, out_dim):
        super(ResNet101_GLDv2, self).__init__()
        self.enet = resnest101(pretrained=True)
        feature = self.enet.fc.in_features
        self.feat = nn.Linear(feature, 512)
        self.swish = Swish_module()
        self.metric_classifiy = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.feat(x)
        x = self.swish(x)
        logits_m = self.metric_classifiy(x)
        return logits_m


