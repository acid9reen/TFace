import argparse
from typing import Literal
from typing import get_args

import torch

from generate_pseudo_labels.extract_embedding.model import R50
from generate_pseudo_labels.extract_embedding.model import MobileFaceNet


Backbone = Literal["MFN", "R_50"]


class Torch2OnnxNamespace(argparse.Namespace):
    backbone: Backbone
    model_path: str


def parse_args() -> Torch2OnnxNamespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("backbone", help="Backbone [MFN, R_50]")
    parser.add_argument("model_path", help="Path to torch model checkpoint")

    return parser.parse_args(namespace=Torch2OnnxNamespace())


def network(backbone: Backbone, checkpoint_path: str):
    if backbone == "MFN":  # MobileFaceNet
        net = MobileFaceNet(
            [112, 112], 512, output_name="GDC", use_type="Qua"
        ).to("cpu")
    elif backbone == "R_50":
        net = R50([112, 112], use_type="Qua").to("cpu")
    else:
        raise ValueError(
            f"Unknown backbone '{backbone}', "
            f"please pick one of these: {get_args(Backbone)}"
        )

    net_dict = net.state_dict()
    data_dict = {
        key.replace("module.", ""): value
        for key, value in torch.load(checkpoint_path, map_location="cpu").items()
    }
    net_dict.update(data_dict)
    net.load_state_dict(net_dict)
    net.eval()

    return net


def main(args: Torch2OnnxNamespace) -> None:
    net = network(args.backbone, args.model_path)

    dummy_input = torch.randn(1, 3, 112, 112, device="cpu")

    torch.onnx.export(
        net,
        dummy_input,
        f"{args.backbone}_sdd_fiqa.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names = ["input"],
        output_names = ["output"],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
