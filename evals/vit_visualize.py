import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch
import numpy as np
import cv2
import pickle
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from kornia.geometry.transform import Resize
import torch.nn.functional as F
from config import TrainConfigViT as Config
from dataio.dataloader import BraTS18Binary, probe_data_folder
from timm.models.vision_transformer import _create_vision_transformer


activation = {}


def get_attn_softmax(name):
    def hook(model, input, output):
        with torch.no_grad():
            input = input[0]
            B, N, C = input.shape
            qkv = (
                model.qkv(input)
                    .detach()
                    .reshape(B, N, 3, model.num_heads, C // model.num_heads)
                    .permute(2, 0, 3, 1, 4)
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * model.scale
            attn = attn.softmax(dim=-1)
            activation[name] = attn

    return hook


# expects timm vis transformer model
def add_attn_vis_hook(model):
    handles = []
    for idx, module in enumerate(list(model.blocks.children())):
        handle = module.attn.register_forward_hook(get_attn_softmax(f"attn{idx}"))
        handles.append(handle)
    return handles


def remove_attn_vis_hook(handles):
    for handle in handles:
        handle.remove()


def get_mask(im, att_mat):
    # Average the attention weights across all heads.
    # att_mat,_ = torch.max(att_mat, dim=1)
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1)).to(device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    # v = aug_att_mat[1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
    mask = cv2.resize(mask / mask.max(), im.shape[2:])
    # result = (mask * im[1:].numpy()).astype("uint8")
    return mask, joint_attentions, grid_size


def show_attention_map(model, im):
    add_attn_vis_hook(model)
    logits = model(im.unsqueeze(0))

    attn_weights_list = list(activation.values())

    mask, joint_attentions, grid_size = get_mask(im, torch.cat(attn_weights_list))
    # plt.imshow(im[0].detach(), cmap="binary")
    # plt.imshow(mask, cmap="jet", alpha=0.4)

    # probs = torch.nn.Softmax(dim=-1)(logits)
    # top5 = torch.argsort(probs, dim=-1, descending=True)
    # print("Prediction Label and Attention Map!\n")
    # for idx in top5[0, :5]:
    #     print(f'{probs[0, idx.item()]:.5f} : {idx.item()}', end='')

    # for i, v in enumerate(joint_attentions):
    #     # Attention from the output token to the input space.
    #     mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    #     mask = cv2.resize(mask / mask.max(), im.shape[2:])[..., np.newaxis]
    #     result = (mask * im).astype("uint8")
    #
    #     fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    #     ax1.set_title('Original')
    #     ax2.set_title('Attention Map_%d Layer' % (i + 1))
    #     _ = ax1.imshow(im)
    #     _ = ax2.imshow(result)

    # plt.show()
    return mask


if __name__ == "__main__":
    import os
    import sys
    import timm

    base_dir = "/hdd0/projects/regex/logs/vit_baseline/20210520_17:29:31"
    model_file = "vit_baseline__best__epoch=026_score=0.9759.pt"
    # model_file = "vit_multiclass_baseline__ckpt__epoch=301_score=0.7137.pt"
    config_file = "params.p"

    raw_params = Config()
    # Serialize transforms
    train_transforms = raw_params.config.pop("train_transforms")
    test_transforms = raw_params.config.pop("test_transforms")
    raw_params.config["train_transforms"] = {"train_transforms": train_transforms.__str__()}
    raw_params.config["test_transforms"] = {"test_transforms": test_transforms.__str__()}

    mask_transforms = Resize((256, 256))
    params = raw_params.config

    params["weights"] = os.path.join(base_dir, model_file)

    # Set seed
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    # Load dataset
    train_metadata, val_metadata, _ = probe_data_folder(params["data_path"],
                                                        train_frac=params["train_frac"],
                                                        bad_files=params["bad_files"],
                                                        subsample_frac=params["subsample_frac"])

    # Dataset
    val_metadata = [(image, mask) for (image, mask) in val_metadata if "y=1" in image]
    # val_metadata = val_metadata[:100]   #DEBUG
    val_dataset = BraTS18Binary(params["data_path"],
                                val_metadata,
                                transforms=test_transforms,
                                mask_transforms=mask_transforms,
                                prefetch_data=params["prefetch_data"],
                                get_mask=True,
                                shuffle=False)
    # Dataloader
    val_loader = DataLoader(dataset=val_dataset,
                            num_workers=1,
                            pin_memory=True,
                            batch_size=1,
                            shuffle=False)


    vit_params = {
        "pretrained": params["pretrained"],
        "img_size": params["spatial_dim"],
        "patch_size": params["patch_size"],
        "in_chans": params["channels"],
        "num_classes": params["num_classes"],
        "embed_dim": params["dim"],
        "depth": params["depth"],
        # "num_heads ": params["heads"],
    }
    model = _create_vision_transformer("vit_base_patch16_224", **vit_params)
    # Cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    baselines = [
        "/hdd0/projects/regex/logs/vit_baseline/20210522_10:59:10/vit_baseline__best__epoch=012_score=0.9684.pt",
        "/hdd0/projects/regex/logs/vit_baseline/20210520_17:29:31/vit_baseline__best__epoch=026_score=0.9759.pt",
        "/hdd0/projects/regex/logs/vit_baseline/20210522_14:43:11/vit_baseline__best__epoch=016_score=0.9801.pt",
        "/hdd0/projects/regex/logs/vit_baseline/20210522_16:36:01/vit_baseline__best__epoch=012_score=0.9717.pt",
        "/hdd0/projects/regex/logs/vit_baseline/20210522_18:28:37/vit_baseline__best__epoch=019_score=0.9761.pt",
    ]

    for k, baseline in enumerate(baselines):
        print(f"*** Processing baseline {k} ***")
        vit_attrs = []
        weights = torch.load(baseline)
        model.load_state_dict(weights, strict=True)
        # model = model.to("cpu")
        for i, (image, label) in tqdm(enumerate(val_loader), total=len(val_loader)):
            handles = add_attn_vis_hook(model)
            image = image.to(device)
            logits = model(image)
            attn_weights_list = list(activation.values())
            mask, joint_attentions, grid_size = get_mask(image, torch.cat(attn_weights_list))
            mask = torch.tensor(mask)[None, ...]
            vit_attrs.append(mask.detach())
            remove_attn_vis_hook(handles)
            activation = {}


        for level in [3, 4]:
            with open(f'ensemble_attrs/k={k}_vit_attrs_level={level}.pkl', 'wb') as f:
                pickle.dump(vit_attrs, f)

    # pass
    # model_names = timm.list_models("vit*")
    # image = torch.randn(3, 224, 224)
    # image = "/home/tomron27/Desktop/download.jpeg"
    # model_name = "vit_base_patch16_224"
    # m = timm.create_model(model_name, pretrained=True)
    # shape = eval(model_name[-3:])
    # show_attention_map(m, image, shape)