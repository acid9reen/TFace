import os

import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config_test import config as conf
from dataset.dataset_txt import load_data as load_data_txt
from model import model_mobilefaceNet
from model import ModelProtocol
from model import OnnxModelAdapter
from model import R50


def setup_dataset():
    """
    Dataset setup
    Bulid a dataloader for training
    """

    dataloader, class_num = load_data_txt(conf, label=False, train=False)
    return dataloader, class_num


def setup_backbone() -> ModelProtocol:
    """
    Backbone setup
    Load a Backbone for training, support MobileFaceNet(MFN) and ResNet50(R50)
    """

    # MobileFaceNet
    if (backbone := conf.backbone) == "MFN":
        net = model_mobilefaceNet.MobileFaceNet(
            [112, 112],
            conf.embedding_size,
            output_name="GDC",
            use_type="Rec",
        ).to(device)

    # ResNet50
    elif backbone == "R_50":
        net = R50([112, 112], use_type="Rec").to(device)
    # Onnx model
    else:
        return OnnxModelAdapter(
            conf.eval_model,
            int(conf.device[conf.device.rfind(':') + 1:]),
            conf.gpu_mem_limit,
        )

    # load trained model weights
    if conf.eval_model != None:
        net_dict = net.state_dict()
        eval_dict = torch.load(conf.eval_model, map_location=device)
        eval_dict = {k.replace("module.", ""): v for k, v in eval_dict.items()}
        same_dict = {k: v for k, v in eval_dict.items() if k in net_dict}
        net_dict.update(same_dict)
        net.load_state_dict(net_dict)

    # if use multi-GPUs
    if device != "cpu" and len(multi_GPUs) > 1:
        net = nn.DataParallel(net, device_ids=multi_GPUs)

    net.eval()

    return net


def calculate_cosine_distance(feats1, feats2):
    """
    Computing cosine distance
    For similarity
    """

    cos = np.dot(feats1, feats2) / (np.linalg.norm(feats1) * np.linalg.norm(feats2))

    return cos


def npy2txt(img_list, feats_nplist, outfile) -> None:
    """
    For save embeddings to txt file
    """

    allFeats = np.load(feats_nplist)

    print(np.shape(allFeats))

    with open(img_list, "r") as f:
        for index, value in tqdm(enumerate(f)):
            imgPath = value.split()[0]
            feats = allFeats[index]
            feats = " ".join(map(str, feats))

            # ouput to the txt
            print(imgPath + " " + feats, file=outfile)


if __name__ == "__main__":
    """
    This method is to extract features from face dataset
    and save to numpy file
    """

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = conf.device
    multi_GPUs = conf.multi_GPUs
    net = setup_backbone()
    outfile = conf.outfile
    dataloader, class_num = setup_dataset()
    count = 0

    # Create outfile folder if
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    with open(conf.img_list, "r") as f:
        txtContent = f.readlines()

        # computer the number of samples
        sample_num = len(txtContent)
        print(f"Number of samples = {sample_num}")

    feats = np.zeros([sample_num, conf.embedding_size])

    with torch.no_grad():
        for datapath, data in tqdm(dataloader, total=len(dataloader)):
            data = data.to(device)
            embeddings = F.normalize(net(data), p=2, dim=1).cpu().numpy().squeeze()
            start_idx = count * conf.batch_size
            end_idx = (count + 1) * conf.batch_size

            try:
                # save embeddings of one iteration
                feats[start_idx:end_idx, :] = embeddings
            except IndexError:
                # save embeddings of the final iteration
                feats[start_idx:, :] = embeddings

            count += 1

        np.save(outfile, feats)
        checkfeats = np.load(outfile)

        print(f"Shape of calculated embeddings {np.shape(checkfeats)}")
        print(f"Feats saved to {outfile}")
