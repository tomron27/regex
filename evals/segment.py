import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pickle
from pyomo.environ import log, Var, Constraint, ConcreteModel, Objective, SolverFactory

from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, average_precision_score

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from torchvision.transforms import InterpolationMode
# from kornia.geometry.transform import Resize
from kornia.augmentation import Normalize

from config import TrainConfig, GPU_ID
from dataio.dataloader import probe_data_folder, BraTS18Binary
# from models.resnet import get_resnet50_attn_classifier
from models.unet import get_unet_encoder_classifier
from models.attention import SumPool
figsize = (24, 24)

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# Ignore pytorch warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_dataset_and_base_model():
    base_dir = "/hdd0/projects/regex/logs/unet_encoder_4attn_marginals/20210513_15:39:29"
    model_file = "unet_encoder_4attn_marginals__best__epoch=009_score=0.9632.pt"

    config_file = "params.p"

    config_handler = open(os.path.join(base_dir, config_file), 'rb')
    params = pickle.load(config_handler)
    params["learnable_attn"] = False
    params["learnable_marginals"] = False
    params["attn_kl"] = False
    params["weights"] = model_path = os.path.join(base_dir, model_file)

    transforms = Compose([
        Normalize(mean=torch.tensor([149.56119, 165.83559, 166.13501, 112.61901]),
                  std=torch.tensor([636.8766, 653.8386, 759.8256, 718.83594])),
        Resize((256, 256), interpolation=InterpolationMode.NEAREST)])

    mask_transforms = Resize((256, 256), interpolation=InterpolationMode.NEAREST)

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
                                transforms=transforms,
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

    # Cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet_encoder_classifier(**params)
    model = model.to(device)
    model.eval()

    return val_dataset, val_loader, model


def save_images_and_masks():
    res = []
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        res.append((image[0].numpy(), target.item(), mask[0].numpy()))
    with open(f'images_and_masks.pkl', 'wb') as f:
        pickle.dump(res, f)


def get_solver(tau3, tau4):
    factor = tau3.shape[0] // tau4.shape[0]
    # 2D indices
    indices3 = [(i, j) for i in range(tau3.shape[0]) for j in range(tau3.shape[1])]
    indices4 = [(i, j) for i in range(tau4.shape[0]) for j in range(tau4.shape[1])]

    # Window indices mapping {mu4 -> mu3_1,1, m3_1,2, ... }
    windows = {}
    for i in range(tau3.shape[0]):
        for j in range(tau3.shape[1]):
            floor_i, floor_j = int(i / factor), int(j / factor)
            if (floor_i, floor_j) not in windows:
                windows[(floor_i, floor_j)] = [(i, j)]
            else:
                windows[(floor_i, floor_j)].append((i, j))

    # Init model
    M = ConcreteModel()

    # Variables
    M.mu3 = Var(indices3, bounds=(0.0, 1.0))
    M.mu4 = Var(indices4, bounds=(0.0, 1.0))

    # Objective expressions
    kl_obj3_expr = sum(M.mu3[i] * (log(M.mu3[i]) - np.log(tau3[i])) for i in indices3)
    kl_obj4_expr = sum(M.mu4[i] * (log(M.mu4[i]) - np.log(tau4[i])) for i in indices4)

    M.obj = Objective(expr=kl_obj3_expr + kl_obj4_expr)

    ### Constraints
    # probability constraints
    mu3_sum_const_expr = sum(M.mu3[i] for i in indices3) == 1.0
    mu4_sum_const_expr = sum(M.mu4[i] for i in indices4) == 1.0

    M.mu3_sum_const = Constraint(expr=mu3_sum_const_expr)
    M.mu4_sum_const = Constraint(expr=mu4_sum_const_expr)

    # window constraints
    for key in windows.keys():
        setattr(M, f"window_const_{key}", Constraint(expr=M.mu4[key] == sum([M.mu3[val] for val in windows[key]])))

    solver = SolverFactory('ipopt')
    solver.solve(M, tee=True)

    mu3 = np.zeros(tau3.shape)
    for i in indices3:
        mu3[i] = M.mu3[i].value

    mu4 = np.zeros(tau4.shape)
    for i in indices4:
        mu4[i] = M.mu4[i].value

    return mu3, mu4


def get_deeplift_attr(level=4, downsample=True, normalize=True):
    attrs = []
    from captum.attr import DeepLift
    attributer = DeepLift(model, multiply_by_inputs=False)
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        image = image.to(device)
        attr = attributer.attribute(image, target=target[0])
        if normalize:
            attr -= attr.view(attr.shape[0], attr.shape[1], -1).min(dim=2)[0][:, :, None, None]
            attr /= attr.view(attr.shape[0], attr.shape[1], -1).max(dim=2)[0][:, :, None, None]
            attr = attr.sum(dim=1)
            attr = torch.softmax(attr.view(1, -1), dim=1).reshape(attr.shape)
        if downsample:
            if level == 4:
                attr = SumPool(8)(attr)
            elif level == 3:
                attr = SumPool(4)(attr)
            elif level == 2:
                attr = SumPool(2)(attr)
            elif level == 1:
                pass
            else:
                raise NotImplementedError
        attrs.append(attr.detach().cpu())
    return attrs


def get_gradcam_attr(level=4, normalize=True):
    attrs = []
    from captum.attr import LayerGradCam
    if level == 4:
        layer = model.encoder4
    elif level == 3:
        layer = model.encoder3
    elif level == 2:
        layer = model.encoder2
    elif level == 1:
        layer = model.encoder1
    else:
        raise NotImplementedError
    attributer = LayerGradCam(model, layer=layer)
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        image = image.to(device)
        attr = attributer.attribute(image, target=target[0])
        if normalize:
            attr -= attr.view(attr.shape[0], attr.shape[1], -1).min(dim=2)[0][:, :, None, None]
            attr /= attr.view(attr.shape[0], attr.shape[1], -1).max(dim=2)[0][:, :, None, None]
            attr = attr.sum(dim=1)
            attr = torch.softmax(attr.view(1, -1), dim=1).reshape(attr.shape)
        attrs.append(attr.detach().cpu())
    return attrs


def get_lrp_attr(level=4, normalize=True, downsample=True):
    attrs = []
    from captum.attr import LRP
    attributer = LRP(model)
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        image = image.to(device)
        attr = attributer.attribute(image, target=target[0])
        if normalize:
            attr -= attr.view(attr.shape[0], attr.shape[1], -1).min(dim=2)[0][:, :, None, None]
            attr /= attr.view(attr.shape[0], attr.shape[1], -1).max(dim=2)[0][:, :, None, None]
            attr = attr.sum(dim=1)
            attr = torch.softmax(attr.view(1, -1), dim=1).reshape(attr.shape)
        if downsample:
            if level == 4:
                attr = SumPool(8)(attr)
            elif level == 3:
                attr = SumPool(4)(attr)
            elif level == 2:
                attr = SumPool(2)(attr)
            elif level == 1:
                pass
            else:
                raise NotImplementedError
        attrs.append(attr.detach().cpu())
    return attrs


def get_inxgrad_attr(level=4, normalize=True, downsample=True):
    attrs = []
    from captum.attr import InputXGradient
    attributer = InputXGradient(model)
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        image = image.to(device)
        attr = attributer.attribute(image, target=target[0])
        if normalize:
            attr -= attr.view(attr.shape[0], attr.shape[1], -1).min(dim=2)[0][:, :, None, None]
            attr /= attr.view(attr.shape[0], attr.shape[1], -1).max(dim=2)[0][:, :, None, None]
            attr = attr.sum(dim=1)
            attr = torch.softmax(attr.view(1, -1), dim=1).reshape(attr.shape)
        if downsample:
            if level == 4:
                attr = SumPool(8)(attr)
            elif level == 3:
                attr = SumPool(4)(attr)
            elif level == 2:
                attr = SumPool(2)(attr)
            elif level == 1:
                pass
            else:
                raise NotImplementedError
        attrs.append(attr.detach().cpu())
    return attrs


def get_marginal_attr(level=4):
    base_dir = "/hdd0/projects/regex/logs/vit_baseline/20210520_17:29:31"
    model_file = "vit_baseline__best__epoch=026_score=0.9759.pt"
    config_file = "params.p"

    config_handler = open(os.path.join(base_dir, config_file), 'rb')
    params = pickle.load(config_handler)
    params["learnable_attn"] = True
    params["learnable_marginals"] = True
    params["attn_kl"] = True
    params["weights"] = model_path = os.path.join(base_dir, model_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet_encoder_classifier(**params)
    model = model.to(device)
    model.eval()

    attns = []
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        image = image.to(device)
        outputs, attn, marginals = model(image)
        attns.append(attn[level-1].detach().cpu())
    return attns


def get_baseline_attr(level=4):

    base_dir = "/hdd0/projects/regex/logs/unet_encoder_4attn/20210513_19:14:05"
    model_file = "unet_encoder_4attn__best__epoch=008_score=0.9640.pt"
    # model_file = "unet_encoder_4attn__best__epoch=106_score=0.9653.pt"
    config_file = "params.p"

    config_handler = open(os.path.join(base_dir, config_file), 'rb')
    params = pickle.load(config_handler)
    params["learnable_attn"] = True
    params["learnable_marginals"] = False
    params["attn_kl"] = False
    params["weights"] = model_path = os.path.join(base_dir, model_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet_encoder_classifier(**params)
    model = model.to(device)
    model.eval()

    attns = []
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        image = image.to(device)
        outputs, attn, _ = model(image)
        attns.append(attn[level-1].detach().cpu())
    return attns


def get_solver_attr(level=3):
    base_dir = "/hdd0/projects/regex/logs/unet_encoder_4attn/20210513_19:14:05"
    model_file = "unet_encoder_4attn__best__epoch=008_score=0.9640.pt"
    config_file = "params.p"

    config_handler = open(os.path.join(base_dir, config_file), 'rb')
    params = pickle.load(config_handler)
    params["learnable_attn"] = True
    params["learnable_marginals"] = False
    params["attn_kl"] = False
    params["weights"] = model_path = os.path.join(base_dir, model_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet_encoder_classifier(**params)
    model = model.to(device)
    model.eval()

    attns = []
    if level not in (3,):   # VERY slow for 1,2; not possible for 4
        return attns

    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        image = image.to(device)
        outputs, attn, _ = model(image)
        p1, p2 = attn[level-1], attn[level]
        mu1, mu2 = get_solver(p1.detach().cpu().numpy()[0], p2.detach().cpu().numpy()[0])
        attns.append(torch.tensor(mu1).unsqueeze(0))
    return attns


def get_ap(mask, heatmap):
    mask = mask[0]
    heatmap = heatmap[0]
    if heatmap.shape != mask.shape:
        heatmap = resize_filter(torch.tensor(heatmap).unsqueeze(0)).squeeze(0) / ((heatmap.shape[0] / mask.shape[0]) ** 2)
    heatmap = heatmap.numpy()
    return average_precision_score(mask.ravel(), heatmap.ravel())


def get_q_iou(mask, heatmap, q=0.025):
    if heatmap.shape != mask.shape:
        heatmap = resize_filter(torch.tensor(heatmap).unsqueeze(0)).squeeze(0) / ((heatmap.shape[0] / mask.shape[0]) ** 2)
    heatmap = heatmap.numpy()
    masked_heatmap = (heatmap > np.quantile(heatmap, 1 - q))
    mask = mask.numpy().astype(np.bool)
    intersect = (mask * masked_heatmap).sum()
    union = (mask + masked_heatmap).sum()
    return intersect / union


def get_ap_stats(attrs):
    ap_list = []
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (_, mask) = sample
        ap = get_ap(mask, attrs[j])
        ap_list.append(ap)
        # if j % 100 == 99:
            # plt.imshow(image[0, 2], cmap="gray")
            # plt.imshow(mask[0], alpha=0.4)
            # plt.show()
            # plt.imshow(attrs[j][0])
            # plt.show()
    return sum(ap_list) / len(ap_list)


def get_q_iou_stats(attrs, q=0.025):
    qiou_list = []
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (_, mask) = sample
        qiou = get_q_iou(mask, attrs[j], q=q)
        qiou_list.append(qiou)
        # if j % 100 == 99:
        #     plt.imshow(image[0, 2], cmap="gray")
        #     plt.imshow(mask[0], alpha=0.4)
        #     plt.show()
        #     plt.imshow(attrs[j][0])
        #     plt.show()
    return sum(qiou_list) / len(qiou_list)


if __name__ == "__main__":
    # Base model and dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset, val_loader, model = get_dataset_and_base_model()

    save_images = False
    calc_attrs = False
    calc_stats = True
    levels = [3, 4]
    # levels = [1, ]
    # levels = [4, ]
    resize_filter = Resize((256, 256), interpolation=InterpolationMode.NEAREST)

    if save_images:
        save_images_and_masks()
    for level in levels:
        result = {}
        print(f"****** Level {level} *****")
        if calc_attrs:
            print("*** Calculating attributions ***")
            gc_attrs = get_gradcam_attr(level=level)
            with open(f'gc_attrs_level={level}.pkl', 'wb') as f:
                pickle.dump(gc_attrs, f)
            inxgrad_attrs = get_inxgrad_attr(level=level)
            with open(f'inxgrad_attrs_level={level}.pkl', 'wb') as f:
                pickle.dump(inxgrad_attrs, f)
            dl_attrs = get_deeplift_attr(level=level)
            with open(f'dl_attrs_level={level}.pkl', 'wb') as f:
                pickle.dump(dl_attrs, f)
            lrp_attrs = get_lrp_attr(level=level)
            with open(f'lrp_attrs_level={level}.pkl', 'wb') as f:
                pickle.dump(lrp_attrs, f)
            baseline_attrs = get_baseline_attr(level=level)
            with open(f'baseline_attrs_level={level}.pkl', 'wb') as f:
                pickle.dump(baseline_attrs, f)
            marginal_attrs = get_marginal_attr(level=level)
            with open(f'marginal_attrs_level={level}.pkl', 'wb') as f:
                pickle.dump(marginal_attrs, f)
            solver_attrs = get_solver_attr(level=level)
            with open(f'solver_attrs_level={level}.pkl', 'wb') as f:
                pickle.dump(solver_attrs, f)


        if calc_stats:
            print("*** Calculating stats ***")
            with open(f'gc_attrs_level={level}.pkl', 'rb') as f:
                gc_attrs = pickle.load(f)
            with open(f'inxgrad_attrs_level={level}.pkl', 'rb') as f:
                inxgrad_attrs = pickle.load(f)
            with open(f'dl_attrs_level={level}.pkl', 'rb') as f:
                dl_attrs = pickle.load(f)
            with open(f'lrp_attrs_level={level}.pkl', 'rb') as f:
                lrp_attrs = pickle.load(f)
            with open(f'vit_attrs_level={level}.pkl', 'rb') as f:
                vit_attrs = pickle.load(f)
            with open(f'baseline_attrs_level={level}.pkl', 'rb') as f:
                baseline_attrs = pickle.load(f)
            with open(f'solver_attrs_level={level}.pkl', 'rb') as f:
                solver_attrs = pickle.load(f)
            with open(f'marginal_attrs_level={level}.pkl', 'rb') as f:
                marginal_attrs = pickle.load(f)

            methods = {
                "GradCAM": gc_attrs,
                "InputXGradients": inxgrad_attrs,
                "DeepLIFT": dl_attrs,
                "LRP": lrp_attrs,
                "ViT Attention": vit_attrs,
                "Baseline Attention": baseline_attrs,
                "Solver Attention": solver_attrs,
                "Marginal Attention": marginal_attrs
            }

            for name, attrs in methods.items():
                print(name)
                if len(attrs) == 0:
                    result[name] = [None, None, None, None]
                else:
                    result[name] = [
                        get_q_iou_stats(attrs, q=0.1),
                        get_q_iou_stats(attrs, q=0.05),
                        get_q_iou_stats(attrs, q=0.025),
                        get_ap_stats(attrs)
                    ]

            with open(f'segmentation_results_level={level}.pkl', 'wb') as f:
                pickle.dump(result, f)

            # dl_stats = []
            # gc_stats = []
            # ggc_stats = []
            # baseline_stats = []
            # kl_stats = []
            # solver_stats = []
            # marginal_stats = []


