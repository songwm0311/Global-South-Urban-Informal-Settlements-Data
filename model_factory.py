"""Model construction helpers for informal-settlement segmentation."""

from models.UnetFamily.MSUNET import msunet
from models.UnetFamily.U2NET import u2net
from models.UnetFamily.Unet import UNet
from models.UnetFamily.Unet2PLUS import UNet2plus
from models.UnetFamily.Unet3PLUS import Unet3Plus


MODEL_NAMES = (
    "UNet",
    "UNet2plus",
    "Unet3Plus",
    "UNet_3Plus_DeepSup",
    "UNet_3Plus_DeepSup_CGM",
    "u2net",
    "U2NETP",
    "msunet",
)


def build_model(name="Unet3Plus", in_channels=13, n_classes=1):
    """Build one of the segmentation architectures bundled with the repository."""
    if name == "UNet":
        return UNet.UNet(
            in_channels=in_channels,
            n_classes=n_classes,
            feature_scale=4,
            is_deconv=True,
            is_batchnorm=True,
        )
    if name == "UNet2plus":
        return UNet2plus.UNet_2Plus(
            in_channels=in_channels,
            n_classes=n_classes,
            feature_scale=4,
            is_deconv=True,
            is_batchnorm=True,
            is_ds=True,
        )
    if name == "Unet3Plus":
        return Unet3Plus.UNet_3Plus(
            in_channels=in_channels,
            n_classes=n_classes,
            feature_scale=4,
            is_deconv=True,
            is_batchnorm=True,
        )
    if name == "UNet_3Plus_DeepSup":
        return Unet3Plus.UNet_3Plus_DeepSup(
            in_channels=in_channels,
            n_classes=n_classes,
            feature_scale=4,
            is_deconv=True,
            is_batchnorm=True,
            other=False,
        )
    if name == "UNet_3Plus_DeepSup_CGM":
        return Unet3Plus.UNet_3Plus_DeepSup_CGM(
            in_channels=in_channels,
            n_classes=n_classes,
            feature_scale=4,
            is_deconv=True,
            is_batchnorm=True,
            other=False,
        )
    if name == "u2net":
        return u2net.U2NET(in_ch=in_channels, out_ch=n_classes, other=False)
    if name == "U2NETP":
        return u2net.U2NETP(in_ch=in_channels, out_ch=n_classes, other=False)
    if name == "msunet":
        return msunet.U_Net(img_ch=in_channels, output_ch=n_classes)
    raise ValueError("Unknown model {!r}; choose from {}".format(name, ", ".join(MODEL_NAMES)))


def primary_output(output):
    """Return the highest-resolution tensor from deep-supervision models."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output
