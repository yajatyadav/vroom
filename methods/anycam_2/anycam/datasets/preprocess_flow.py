import argparse
import math
import os
import torch
import requests

import sys


sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")


from tqdm import tqdm
from unimatch.unimatch import UniMatch
import torch.nn.functional as F

from anycam.common.io.io import save_flow
from anycam.datasets.waymo.waymo_dataset import WaymoDataset
from anycam.datasets.sintel.sintel_dataset import SintelDataset
from anycam.datasets.realestate10k.re10k_dataset import RealEstate10kDataset
from anycam.datasets.youtubevos.youtubevos_dataset import YouTubeVOSDataset
from anycam.datasets.opendv.opendv_dataset import OpenDVDataset
from anycam.datasets.walkingtours.walkingtours_dataset import WalkingToursDataset
from anycam.datasets.common import get_index_selector, get_sequence_sampler


def get_outpath_sintel(out_folder, img_path, is_fwd):
    parts = img_path.split("/")
    seq = parts[-2]
    name = parts[-1][:-4]
    out_path = os.path.join(out_folder, "unimatch_flows", seq, f"{name}_{'fwd' if is_fwd else 'bwd'}.png")
    return out_path


def get_outpath_waymo(out_folder, img_path, is_fwd):
    parts = img_path.split("/")
    seq = parts[-4]
    cam_name = parts[-2]
    name = parts[-1][:-4]
    out_path = os.path.join(out_folder, seq, "unimatch_flows", cam_name, f"{name}_{'fwd' if is_fwd else 'bwd'}.png")
    return out_path


def get_outpath_realestate10k(out_folder, img_path, is_fwd):
    parts = img_path.split("/")
    split = parts[-3]
    seq = parts[-2]
    name = parts[-1][:-4]
    out_path = os.path.join(out_folder, split, seq, f"{name}_{'fwd' if is_fwd else 'bwd'}.png")
    return out_path


def get_outpath_youtubevos(out_folder, img_path, is_fwd):
    parts = img_path.split("/")
    seq = parts[-2]
    name = parts[-1][:-4]
    out_path = os.path.join(out_folder, "unimatch_flows", seq, f"{name}_{'fwd' if is_fwd else 'bwd'}.png")
    return out_path


def get_outpath_opendv(out_folder, img_path, is_fwd):
    parts = img_path.split("/")
    seq = parts[-2]
    name = parts[-1][:-4]
    out_path = os.path.join(out_folder, seq, f"{name}_{'fwd' if is_fwd else 'bwd'}.png")
    return out_path


def get_outpath_walkingtours(out_folder, img_path, is_fwd):
    parts = img_path.split("/")
    seq = parts[-2]
    name = parts[-1][:-4]
    out_path = os.path.join(out_folder, seq, f"{name}_{'fwd' if is_fwd else 'bwd'}.png")
    return out_path


def get_sintel_dataset(data_path, split, every_nth=1):
    dataset = SintelDataset(
        data_path,
        split,
        image_size=None,
        index_selector=get_index_selector(True),
        sequence_sampler=get_sequence_sampler(True),
        frame_count=2,
        dilation=every_nth,
    )
    return dataset


def get_waymo_dataset(data_path, split, every_nth=1):
    dataset = WaymoDataset(
        data_path,
        split,
        image_size=None,
        index_selector=get_index_selector(True),
        sequence_sampler=get_sequence_sampler(True),
        frame_count=2,
        dilation=every_nth,
    )
    return dataset


def get_realestate10k_dataset(data_path, split, every_nth=1):
    dataset = RealEstate10kDataset(
        data_path,
        split,
        image_size=None,
        index_selector=get_index_selector(True),
        sequence_sampler=get_sequence_sampler(True),
        frame_count=2,
        dilation=every_nth,
    )
    return dataset


def get_youtubevos_dataset(data_path, split, every_nth=1):
    dataset = YouTubeVOSDataset(
        data_path,
        split,
        image_size=None,
        index_selector=get_index_selector(True),
        sequence_sampler=get_sequence_sampler(True),
        frame_count=2,
        dilation=every_nth,
    )
    return dataset


def get_opendv_dataset(data_path, split, every_nth=1):
    dataset = OpenDVDataset(
        data_path,
        split,
        image_size=None,
        index_selector=get_index_selector(True),
        sequence_sampler=get_sequence_sampler(True),
        frame_count=2,
        dilation=every_nth,
    )
    return dataset


def get_walkingtours_dataset(data_path, split, every_nth=1):
    dataset = WalkingToursDataset(
        data_path,
        split,
        image_size=None,
        index_selector=get_index_selector(True),
        sequence_sampler=get_sequence_sampler(True),
        frame_count=2,
        dilation=every_nth,
    )
    return dataset


def get_unimatch_model():
    ckpt_path = os.path.join(os.environ["HOME"], ".cache", "torch", "checkpoints", "gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth")
    if not os.path.exists(ckpt_path):
        r = requests.get('https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
        with open(ckpt_path , 'wb') as f:
            f.write(r.content)
    unimatch = UniMatch(
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task="flow")
    unimatch.load_state_dict(torch.load(ckpt_path)['model'], strict=True)
    unimatch.eval()
    for param in unimatch.parameters():
        param.requires_grad = False

    return unimatch


def unimatch_fwd(model, img0, img1):
    n, c, h, w = img0.shape

    max_size = 320
    smaller = min(h, w)

    if smaller > max_size:
        scale_factor = max_size / smaller
        target_h = h * scale_factor
        target_w = w * scale_factor
    else:
        target_h = h
        target_w = w
    
    target_h = math.ceil(target_h / 32) * 32
    target_w = math.ceil(target_w / 32) * 32

    if target_h != h or target_w != w:
        img0 = F.interpolate(img0, (target_h, target_w), mode='bilinear', align_corners=True)
        img1 = F.interpolate(img1, (target_h, target_w), mode='bilinear', align_corners=True)

    img0 = img0 * 255
    img1 = img1 * 255

    attn_type = 'swin'
    attn_splits_list = [2, 8]
    corr_radius_list = [-1, 4]
    prop_radius_list = [-1, 1]
    num_reg_refine = 1

    results_dict = model(img0, img1,
                        attn_type=attn_type,
                        attn_splits_list=attn_splits_list,
                        corr_radius_list=corr_radius_list,
                        prop_radius_list=prop_radius_list,
                        num_reg_refine=num_reg_refine,
                        task="flow",
                        pred_bidir_flow=True,
                        )
    
    flows = results_dict['flow_preds'][-1]

    if target_h != h or target_w != w:
        flows = F.interpolate(flows, (h, w), mode='nearest')

    flow_fwd = flows[:n]
    flow_bwd = flows[n:]
    return flow_fwd, flow_bwd


@torch.no_grad()
def main(args):
    # Your code logic goes here
    
    dataset = args.dataset
    split = args.split
    data_path = args.data_path
    out_path = args.out_path
    replace_first = args.replace_first
    replace_last = args.replace_last
    every_nth = args.every_nth

    if dataset == "sintel":
        dataset = get_sintel_dataset(data_path, split, every_nth)
        out_path_fn = get_outpath_sintel
    elif dataset == "waymo":
        dataset = get_waymo_dataset(data_path, split, every_nth)
        out_path_fn = get_outpath_waymo
    elif dataset == "realestate10k":
        dataset = get_realestate10k_dataset(data_path, split, every_nth)
        out_path_fn = get_outpath_realestate10k
    elif dataset == "youtubevos":
        dataset = get_youtubevos_dataset(data_path, split, every_nth)
        out_path_fn = get_outpath_youtubevos
    elif dataset == "opendv":
        dataset = get_opendv_dataset(data_path, split, every_nth)
        out_path_fn = get_outpath_opendv
    elif dataset == "walkingtours":
        dataset = get_walkingtours_dataset(data_path, split, every_nth)
        out_path_fn = get_outpath_walkingtours
    else:
        raise ValueError("Unknown dataset")

    model = get_unimatch_model().cuda()

    prev_sequnence = None

    pbar = tqdm(range(len(dataset)))
    for i in pbar:
        seq, _ = dataset._index_to_seq_ids(i)
        paths = dataset.get_img_paths(i)

        if seq != prev_sequnence:
            prev_sequnence = seq
            is_first = True
        else:
            is_first = False

        if i == len(dataset) - 1 or seq != dataset._index_to_seq_ids(i+1)[0]:
            is_last = True
        else:
            is_last = False
        
        is_missing = False
        for j, img_path in enumerate(paths):
            if j != len(paths) - 1:
                out_path_img = out_path_fn(out_path, img_path, is_fwd=True)
                if not os.path.exists(out_path_img):
                    # print("Missing", out_path_img)
                    is_missing = True
                    break
            if j != 0:
                out_path_img = out_path_fn(out_path, img_path, is_fwd=False)
                if not os.path.exists(out_path_img):
                    # print("Missing", out_path_img)
                    is_missing = True
                    break
        if (not (replace_first and is_first)) and (not (replace_last and is_last)) and not is_missing:
            # print("Skipping", i)
            continue

        data = dataset[i]

        for j in range(len(data["imgs"])-1):
            out_path_fwd = out_path_fn(out_path, paths[j], is_fwd=True)
            out_path_bwd = out_path_fn(out_path, paths[j+1], is_fwd=False)

            if not (replace_first and is_first) and not (replace_last and is_last):
                if os.path.exists(out_path_fwd) and os.path.exists(out_path_bwd):
                    continue

            img0 = torch.tensor(data["imgs"][j+0:j+1]).cuda()
            img1 = torch.tensor(data["imgs"][j+1:j+2]).cuda()

            flow_fwd, flow_bwd = unimatch_fwd(model, img0, img1)

            os.makedirs(os.path.dirname(out_path_fwd), exist_ok=True)
            os.makedirs(os.path.dirname(out_path_bwd), exist_ok=True)

            save_flow(out_path_fwd, flow_fwd[0].cpu().numpy())
            save_flow(out_path_bwd, flow_bwd[0].cpu().numpy())
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow Preprocessing")
    
    # Add your command-line arguments here
    parser.add_argument("--dataset")
    parser.add_argument("--split", default=None)
    parser.add_argument("--data_path")
    parser.add_argument("--out_path")
    parser.add_argument("--every_nth", type=int, default=1)
    parser.add_argument("--replace_first", action="store_true")
    parser.add_argument("--replace_last", action="store_true")

    args = parser.parse_args()
    main(args)