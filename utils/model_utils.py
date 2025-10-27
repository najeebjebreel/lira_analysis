import torch.nn as nn
import torch.nn.functional as F
import timm


class FCN(nn.Module):
    """
    A generic fully connected network for tabular datasets (e.g., Purchase, Location, Texas).
    It exposes intermediate representations useful for privacy attacks like LiRA or RMIA.
    """

    def __init__(self, input_dim, num_classes, hidden_dims=[1024, 512, 256, 128],
                 dropout_rate=0.0, activation='tanh'):
        super().__init__()
        self.activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'gelu': nn.GELU
        }
        if activation not in self.activations:
            raise ValueError(f"Unsupported activation: {activation}")
        act_fn = self.activations[activation]
        layers, prev_dim = [], input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.dropout(x)
        return self.classifier(x)


# ----------------- WideResNet custom -----------------

class WideBasicBlock(nn.Module):
    """WideResNet basic block."""

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.equal_in_out = (in_planes == out_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = drop_rate
        if not self.equal_in_out:
            self.conv_shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        if not self.equal_in_out:
            x = self.relu1(self.bn1(x))
            out = self.conv1(x)
        else:
            out = self.conv1(self.relu1(self.bn1(x)))
        out = self.relu2(self.bn2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        shortcut = x if self.equal_in_out else self.conv_shortcut(x)
        return out + shortcut


class WideNetworkBlock(nn.Module):
    """Stack of WideResNet blocks."""

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super().__init__()
        layers = [block(in_planes if i == 0 else out_planes,
                        out_planes,
                        stride if i == 0 else 1,
                        drop_rate)
                  for i in range(nb_layers)]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """WideResNet implementation."""

    def __init__(self, depth, num_classes, width=2, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, 'Depth must be 6n+4'
        n = (depth - 4) // 6
        channels = [16, 16 * width, 32 * width, 64 * width]

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = WideNetworkBlock(n, channels[0], channels[1], WideBasicBlock, 1, drop_rate)
        self.block2 = WideNetworkBlock(n, channels[1], channels[2], WideBasicBlock, 2, drop_rate)
        self.block3 = WideNetworkBlock(n, channels[2], channels[3], WideBasicBlock, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity()
        self.fc = nn.Linear(channels[3], num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        return self.fc(out)


# ----------------- Model factory with CIFAR optimizations -----------------

def get_model(num_classes, **kwargs):
    """
    Create a model using timm for vision architectures, or custom FCN/WideResNet.

    Args:
        architecture (str): Name in timm registry (e.g., 'resnet18', 'vit_base_patch16_224')
        num_classes (int)
        **kwargs:
          - pretrained (bool)
          - input_channels (int)
          - drop_rate (float)
          - cifar_stem (bool): replace 7x7 stem+pool with 3x3 conv, no pool
          - input_dim, hidden_dims, activation (for FCN)
    Returns:
        nn.Module
    """
    architecture = kwargs.get('architecture', 'resnet18')
    pretrained = kwargs.get('pretrained', False)
    drop_rate = kwargs.get('drop_rate', 0.0)
    in_chans = kwargs.get('input_channels', 3)
    cifar_stem = kwargs.get('cifar_stem', False)

    # Tabular FCN
    if architecture in ['fcn']:
        input_dim = kwargs.get('input_dim', 600)
        hidden_dims = kwargs.get('hidden_dims', [1024, 512, 256, 128])
        activation = kwargs.get('activation', 'tanh')
        return FCN(input_dim, num_classes, hidden_dims, dropout_rate=drop_rate, activation=activation)

    # Custom WideResNet
    if architecture.startswith('wrn28-'):
        width = int(architecture.split('-')[-1])
        return WideResNet(depth=28, num_classes=num_classes, width=width, drop_rate=drop_rate)

    # timm vision models
    model = timm.create_model(
        architecture,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_rate
    )


    if cifar_stem and hasattr(model, 'conv_stem'):
            model.conv_stem = nn.Conv2d(
                                      in_chans, model.conv_stem.out_channels,
                                      kernel_size=3, stride=1, padding=1, bias=False
                                    )
            
    # CIFAR stem optimization: replace 7x7 conv + pool with 3x3 conv
    if cifar_stem and hasattr(model, 'conv1'):
        model.conv1 = nn.Conv2d(
            in_chans, model.conv1.out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        if hasattr(model, 'maxpool'):
            model.maxpool = nn.Identity()

    return model
