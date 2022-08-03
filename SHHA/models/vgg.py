import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

# the fusion module
class Stack(nn.Module):
    def __init__(self):
        super(Stack, self).__init__()
        self.output_layer = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        ).to(device='cuda')
           
    def forward(self, answers):
        x = self.output_layer(answers)
        return torch.abs(x)

# the forward adapter
class Adapter1(nn.Module):
    def __init__(self):
        super(Adapter1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=4)
        ).to(device='cuda')

    def forward(self, x):
        x = self.layer(x)
        return x

class Adapter2(nn.Module):
    def __init__(self):
        super(Adapter2, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=5, padding=2, stride=4)
        ).to(device='cuda')

    def forward(self, x):
        x = self.layer(x)
        return x

class Adapter3(nn.Module):
    def __init__(self):
        super(Adapter3, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=7, padding=3, stride=4)
        ).to(device='cuda')

    def forward(self, x):
        x = self.layer(x)
        return x
        

class VGG(nn.Module):
    def __init__(self, flag): 
        # vgg19
        super(VGG, self).__init__()

        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.layer1_relu = nn.ReLU(inplace=True)

        self.layer2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.layer2_relu = nn.ReLU(inplace=True)
        self.layer2_maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.layer3_relu = nn.ReLU(inplace=True)

        self.layer4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.layer4_relu = nn.ReLU(inplace=True)
        self.layer4_maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.layer5_relu = nn.ReLU(inplace=True)

        self.layer6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.layer6_relu = nn.ReLU(inplace=True)

        self.layer7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.layer7_relu = nn.ReLU(inplace=True)

        self.layer8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.layer8_relu = nn.ReLU(inplace=True)
        self.layer8_maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.layer9_relu = nn.ReLU(inplace=True)

        self.layer10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer10_relu = nn.ReLU(inplace=True)

        self.layer11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer11_relu = nn.ReLU(inplace=True)

        self.layer12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer12_relu = nn.ReLU(inplace=True)
        self.layer12_maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer13_relu = nn.ReLU(inplace=True)

        self.layer14 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer14_relu = nn.ReLU(inplace=True)

        self.layer15 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer15_relu = nn.ReLU(inplace=True)

        self.layer16 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer16_relu = nn.ReLU(inplace=True)

        # the backward adapter
        if flag == 0:
            self.de_pred5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

            self.de_pred1 = nn.Sequential(
                nn.Conv2d(512 + 64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        elif flag == 1:
            self.de_pred5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=5, padding=2),
                nn.ReLU(inplace=True)
            )

            self.de_pred1 = nn.Sequential(
                nn.Conv2d(512 + 64, 64, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.ReLU(inplace=True)
            )
        else:
            self.de_pred5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=7, padding=3),
                nn.ReLU(inplace=True)
            )

            self.de_pred1 = nn.Sequential(
                nn.Conv2d(512 + 64, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=7, padding=3),
                nn.ReLU(inplace=True)
            )

    def run_default_model(self):
        self.encoder_1 = nn.Sequential(
          self.layer1,
          self.layer1_relu,

          self.layer2,
          self.layer2_relu,
          self.layer2_maxpooling
        )

        self.encoder_2 = nn.Sequential(
          self.layer3,
          self.layer3_relu,

          self.layer4,
          self.layer4_relu,
          self.layer4_maxpooling
        )

        self.encoder_3 = nn.Sequential(
          self.layer5,
          self.layer5_relu,

          self.layer6,
          self.layer6_relu,

          self.layer7,
          self.layer7_relu,

          self.layer8,
          self.layer8_relu,
          self.layer8_maxpooling
        )
        
        self.encoder_4 = nn.Sequential(
          self.layer9,
          self.layer9_relu,

          self.layer10,
          self.layer10_relu,

          self.layer11,
          self.layer11_relu,

          self.layer12,
          self.layer12_relu,
          self.layer12_maxpooling
        )

        self.encoder_5 = nn.Sequential(
          self.layer13,
          self.layer13_relu,

          self.layer14,
          self.layer14_relu,

          self.layer15,
          self.layer15_relu,

          self.layer16,
          self.layer16_relu
        )

    def forward(self, x):
        input_size = x.size()
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x4 = self.encoder_4(x3)
        x5 = self.encoder_5(x4)

        x = self.de_pred5(x5)
        x = F.interpolate(x, size=x1.size()[2:])
        
        x = torch.cat([x1, x], 1)
        x = self.de_pred1(x)

        return torch.abs(x)


def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
  'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19(flag):
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(flag)
    model_parameters = model_zoo.load_url(model_urls['vgg19'])
    model.layer1.load_state_dict({"weight": model_parameters["features.0.weight"], "bias": model_parameters["features.0.bias"]})
    model.layer2.load_state_dict({"weight": model_parameters["features.2.weight"], "bias": model_parameters["features.2.bias"]})
    model.layer3.load_state_dict({"weight": model_parameters["features.5.weight"], "bias": model_parameters["features.5.bias"]})
    model.layer4.load_state_dict({"weight": model_parameters["features.7.weight"], "bias": model_parameters["features.7.bias"]})
    model.layer5.load_state_dict({"weight": model_parameters["features.10.weight"], "bias": model_parameters["features.10.bias"]})
    model.layer6.load_state_dict({"weight": model_parameters["features.12.weight"], "bias": model_parameters["features.12.bias"]})
    model.layer7.load_state_dict({"weight": model_parameters["features.14.weight"], "bias": model_parameters["features.14.bias"]})
    model.layer8.load_state_dict({"weight": model_parameters["features.16.weight"], "bias": model_parameters["features.16.bias"]})
    model.layer9.load_state_dict({"weight": model_parameters["features.19.weight"], "bias": model_parameters["features.19.bias"]})
    model.layer10.load_state_dict({"weight": model_parameters["features.21.weight"], "bias": model_parameters["features.21.bias"]})
    model.layer11.load_state_dict({"weight": model_parameters["features.23.weight"], "bias": model_parameters["features.23.bias"]})
    model.layer12.load_state_dict({"weight": model_parameters["features.25.weight"], "bias": model_parameters["features.25.bias"]})
    model.layer13.load_state_dict({"weight": model_parameters["features.28.weight"], "bias": model_parameters["features.28.bias"]})
    model.layer14.load_state_dict({"weight": model_parameters["features.30.weight"], "bias": model_parameters["features.30.bias"]})
    model.layer15.load_state_dict({"weight": model_parameters["features.32.weight"], "bias": model_parameters["features.32.bias"]})
    model.layer16.load_state_dict({"weight": model_parameters["features.34.weight"], "bias": model_parameters["features.34.bias"]})
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    model.run_default_model()
    return model

