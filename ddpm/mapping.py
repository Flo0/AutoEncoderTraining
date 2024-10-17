from unet_impl import OneResUNet, TwoResUNet, AttentionUNet

MODEL_NAME_MAPPING = {
    "OneResUNet": OneResUNet,
    "TwoResUNet": TwoResUNet,
    "AttentionUNet": AttentionUNet,
}
