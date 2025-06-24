from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn

class SSD300(nn.Module):
    def __init__(self, num_classes):
        super(SSD300, self).__init__()
        self.num_classes = num_classes
        vgg_features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        vgg_features[16] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        vgg_features[23] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.vgg_part1 = nn.ModuleList(vgg_features[:23])
        self.vgg_part2 = nn.ModuleList(vgg_features[23:30])
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.aux_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, padding=0), nn.ReLU(True), nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, padding=0), nn.ReLU(True), nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, padding=0), nn.ReLU(True), nn.Conv2d(128, 256, kernel_size=3, padding=0), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, padding=0), nn.ReLU(True), nn.Conv2d(128, 256, kernel_size=3, padding=0), nn.ReLU(True))
        ])
        self.boxes_per_loc = [4, 6, 6, 6, 4, 4]
        self.loc_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        in_channels = [512, 1024, 512, 256, 256, 256]
        for i in range(len(in_channels)):
            self.loc_convs.append(nn.Conv2d(in_channels[i], self.boxes_per_loc[i] * 4, kernel_size=3, padding=1))
            self.cls_convs.append(nn.Conv2d(in_channels[i], self.boxes_per_loc[i] * num_classes, kernel_size=3, padding=1))
        self.init_weights()

    def init_weights(self):
        layers_to_init = [self.conv6, self.conv7, self.aux_convs, self.loc_convs, self.cls_convs]
        for layer_group in layers_to_init:
            for m in layer_group.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        features, locs, confs = [], [], []
        for layer in self.vgg_part1: x = layer(x)
        features.append(x)
        for layer in self.vgg_part2: x = layer(x)
        x = self.pool5(x)
        x = F.relu(self.conv6(x), inplace=True)
        x = F.relu(self.conv7(x), inplace=True)
        features.append(x)
        for conv_block in self.aux_convs:
            x = conv_block(x)
            features.append(x)
        for i, feature in enumerate(features):
            loc = self.loc_convs[i](feature).permute(0, 2, 3, 1).contiguous()
            locs.append(loc.view(loc.size(0), -1))
            conf = self.cls_convs[i](feature).permute(0, 2, 3, 1).contiguous()
            confs.append(conf.view(conf.size(0), -1))
        locs = torch.cat(locs, 1).view(locs[0].size(0), -1, 4)
        confs = torch.cat(confs, 1).view(confs[0].size(0), -1, self.num_classes)
        return locs, confs

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers(self.vgg_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers, in_channels = [], 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)