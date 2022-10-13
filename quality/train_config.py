import os

import torch
import torchvision.transforms as T


class Config:
    # <<---- dataset ---->>
    dataset_name = ""
    img_list = r"./generate_pseudo_labels/annotations/quality_pseudo_labels.txt"


    # <<---- data preprocess ---->>
    transform = T.Compose([
        T.Resize((112, 112)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


    # <<---- network settings ---->>
    # [MFN, R_50]
    backbone = "MFN"
    finetuning_model = None


    # <<---- training settings ---->>
    seed = 0
    epoch = 20
    # ['L1', 'L2', 'SmoothL1']
    loss = "SmoothL1"
    lr = 0.0001
    stepLR = [5, 10]
    weight_decay = 0.0005


    # <<---- device configuration ---->>
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 70
    pin_memory = True
    multi_GPUs = [0]
    num_workers = 6


    # <<---- save settings ---->>
    checkpoints = os.path.join(
        r"./checkpoints/Quality_Regression/",
        dataset_name,
        backbone,
    )
    checkpoints_name = backbone
    saveModel_epoch = 1


    # <<---- other ---->>
    display = 100


    def to_dict(self) -> dict[str, float | str]:
        return {
            "seed": self.seed,
            "num epochs": self.epoch,
            "lr": self.lr,
            "weight decay": self.weight_decay,
            "step lr": str(self.stepLR),
            "loss": self.loss,
            "backbone": self.backbone,
        }


config = Config()
