import torch
import torchvision.transforms as T


class Config:
    # dataset
    img_list = r"./generate_pseudo_labels/annotations/quality_pseudo_labels.txt"
    finetuning_model = None

    # save settings
    checkpoints = r"./checkpoints/MS1M_Quality_Regression/S1"
    checkpoints_name = "MFN"

    # data preprocess
    transform = T.Compose([
        T.Resize((112, 112)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # training settings
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 0
    multi_GPUs = [0]
    # [MFN, R_50]
    backbone = "MFN"
    pin_memory = True
    num_workers = 6
    batch_size = 70
    epoch = 20
    lr = 0.0001
    stepLR = [5, 10]
    weight_decay = 0.0005
    display = 100
    saveModel_epoch = 1
    # ['L1', 'L2', 'SmoothL1']
    loss = "SmoothL1"

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
