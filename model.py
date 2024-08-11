import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

from config import Config

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


aux_params = dict(
    pooling="avg",  # one of 'avg', 'max'
    dropout=0.5,  # dropout ratio, default is None
    activation="sigmoid",  # activation function, default is None
    classes=Config.num_classes,  # define number of output labels
)


def get_model(device, model_name="UNet++"):
    if model_name == "UNet++":
        model = smp.UnetPlusPlus(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=3,
            aux_params=aux_params,
            classes=1,
        ).to(device)
    elif model_name == "MAnet":
        model = smp.MAnet(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=3,
            aux_params=aux_params,
            classes=1,
        ).to(device)
    elif model_name == "FPN":
        model = smp.FPN(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=3,
            aux_params=aux_params,
            classes=1,
        ).to(device)
    elif model_name == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=3,
            aux_params=aux_params,
            classes=1,
        ).to(device)
    elif model_name == "PSPNet":
        model = smp.PSPNet(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=3,
            aux_params=aux_params,
            classes=1,
        ).to(device)
    elif model_name == "Linknet":
        model = smp.Linknet(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=3,
            aux_params=aux_params,
            classes=1,
        ).to(device)
    elif model_name == "PAN":
        model = smp.PAN(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=3,
            aux_params=aux_params,
            classes=1,
        ).to(device)
    else:
        print("No model name found!\n")
    return model

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()