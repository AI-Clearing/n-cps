# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict
from config import config
from base_model import resnet50, resnet101

class Network(nn.Module):
    def __init__(self, num_classes, criterion, norm_layer, pretrained_model=None, num_networks=2, resnet_type='resnet50'):
        super(Network, self).__init__()
        assert num_networks > 1, 'At least 2 networks are necessary!'
        self.branches = nn.ModuleList([
            SingleNetwork(num_classes, criterion, norm_layer, pretrained_model, resnet_type=resnet_type) for _ in range(num_networks)
            ])

    def forward(self, data, step=1):
        if not self.training:
            return self.forward_eval(data)
        return self.branches[step-1](data)
    
    def forward_eval(self, data):
        if config.eval_mode is None or config.eval_mode == 'single':
            return self.branches[0](data)
        else:
            preds_list = []
            for branch in self.branches:
                preds_list.append(branch(data))
            preds = torch.stack(preds_list)  # (n, b, c, w, h)
            if config.eval_mode == 'max_confidence':
                return preds.max(dim=0)[0]
            elif config.eval_mode == 'max_confidence_softmax':
                return F.softmax(preds, dim=2).max(dim=0)[0]
            elif config.eval_mode == 'soft_voting':
                return F.softmax(preds, dim=2).sum(dim=0)
            elif config.eval_mode == 'hard_voting':
                mode_tensor = preds.argmax(dim=2).mode(dim=0)[0]
                return F.one_hot(mode_tensor, num_classes=preds.shape[2]).reshape(preds.shape[1:])
            elif config.eval_mode == 'max_confidence_overlap':
                overlap = ((self.branches[0](data).argmax(dim=1) == preds.max(dim=0)[0].argmax(dim=1)).sum() / preds.max(dim=0)[0].argmax(dim=1).numel()).item()
                raise Exception(f'Not implemented yet - eval_mode: {config.eval_mode}')
            else:
                raise Exception(f'No such eval_mode: {config.eval_mode}')

class SingleNetwork(nn.Module):
    def __init__(self, num_classes, criterion, norm_layer, pretrained_model=None, resnet_type='resnet50'):
        super(SingleNetwork, self).__init__()
        if resnet_type == 'resnet50':
            self.backbone = resnet50(pretrained_model, norm_layer=norm_layer,
                                    bn_eps=config.bn_eps,
                                    bn_momentum=config.bn_momentum,
                                    deep_stem=True, stem_width=64)
        elif resnet_type == 'resnet101':
            self.backbone = resnet101(pretrained_model, norm_layer=norm_layer,
                                    bn_eps=config.bn_eps,
                                    bn_momentum=config.bn_momentum,
                                    deep_stem=True, stem_width=64)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.head = Head(num_classes, norm_layer, config.bn_momentum)
        self.business_layer = []
        self.business_layer.append(self.head)
        self.criterion = criterion

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        self.business_layer.append(self.classifier)

    def forward(self, data):
        blocks = self.backbone(data)
        v3plus_feature = self.head(blocks)      # (b, c, h, w)
        b, c, h, w = v3plus_feature.shape

        pred = self.classifier(v3plus_feature)

        b, c, h, w = data.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            return v3plus_feature, pred
        return pred

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)       # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)  # add activation layer

        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool

class Head(nn.Module):
    def __init__(self, classify_classes, norm_act=nn.BatchNorm2d, bn_momentum=0.0003):
        super(Head, self).__init__()

        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [6, 12, 18], norm_act=norm_act)

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            norm_act(48, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )

    def forward(self, f_list):
        f = f_list[-1]
        f = self.aspp(f)

        low_level_features = f_list[0]
        low_h, low_w = low_level_features.size(2), low_level_features.size(3)
        low_level_features = self.reduce(low_level_features)

        f = F.interpolate(f, size=(low_h, low_w), mode='bilinear', align_corners=True)
        f = torch.cat((f, low_level_features), dim=1)
        f = self.last_conv(f)

        return f


if __name__ == '__main__':
    model = Network(40, criterion=nn.CrossEntropyLoss(),
                    pretrained_model=None,
                    norm_layer=nn.BatchNorm2d)
    left = torch.randn(2, 3, 128, 128)
    right = torch.randn(2, 3, 128, 128)

    print(model.backbone)

    out = model(left)
    print(out.shape)
