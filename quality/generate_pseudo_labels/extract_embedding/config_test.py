import torch
import torchvision.transforms as T


Gb = int


class Config:
    # <<---- dataset ---->>
    data_root = r"C:\Users\Ruslan\repos\face_image_quality_test\lfw-deepfunneled"
    img_list = r"C:\Users\Ruslan\repos\TFace\quality\generate_pseudo_labels\DATA.labelpath"
    eval_model = r"C:\Users\Ruslan\repos\face_image_quality_test\misc\rec_model_vggfaces2.onnx"
    outfile = r"C:\Users\Ruslan\repos\TFace\quality\generate_pseudo_labels\feats_npy\Embedding_Features.npy"

    # <<---- data preprocess ---->>
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Transformations for original authors' models:
    # transform = T.Compose([
    #     T.Resize((112, 112)),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])

    # <<---- network settings ---->>
    # [MFN, R_50, Onnx]
    backbone = "Onnx"

    # Do not touch, this is for original authors' models
    embedding_size = 512

    # <<---- evaluation settings ---->>
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    multi_GPUs = [0]
    batch_size = 60
    pin_memory = True
    num_workers = 4
    gpu_mem_limit: Gb = 8


config = Config()
