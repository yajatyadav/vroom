import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

from unimatch.unimatch import UniMatch
from utils.flow_viz import flow_to_image
from dataloader.stereo import transforms
from utils.visualization import vis_disparity

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def inference(image1, image2, task='flow'):
    """Inference on an image pair for optical flow or stereo disparity prediction"""

    model = UniMatch(feature_channels=128,
                     num_scales=2,
                     upsample_factor=4,
                     ffn_dim_expansion=4,
                     num_transformer_layers=6,
                     reg_refine=True,
                     task=task)

    model.eval()

    if task == 'flow':
        checkpoint_path = 'pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth'
    else:
        checkpoint_path = 'pretrained/gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth'

    checkpoint_flow = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_flow['model'], strict=True)

    padding_factor = 32
    attn_type = 'swin' if task == 'flow' else 'self_swin2d_cross_swin1d'
    attn_splits_list = [2, 8]
    corr_radius_list = [-1, 4]
    prop_radius_list = [-1, 1]
    num_reg_refine = 6 if task == 'flow' else 3

    # smaller inference size for faster speed
    max_inference_size = [384, 768] if task == 'flow' else [640, 960]

    transpose_img = False

    image1 = np.array(image1).astype(np.float32)
    image2 = np.array(image2).astype(np.float32)

    if len(image1.shape) == 2:  # gray image
        image1 = np.tile(image1[..., None], (1, 1, 3))
        image2 = np.tile(image2[..., None], (1, 1, 3))
    else:
        image1 = image1[..., :3]
        image2 = image2[..., :3]

    if task == 'flow':
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0)
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0)
    else:
        val_transform_list = [transforms.ToTensor(),
                              transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                              ]

        val_transform = transforms.Compose(val_transform_list)

        sample = {'left': image1, 'right': image2}
        sample = val_transform(sample)

        image1 = sample['left'].unsqueeze(0)  # [1, 3, H, W]
        image2 = sample['right'].unsqueeze(0)  # [1, 3, H, W]

    # the model is trained with size: width > height
    if task == 'flow' and image1.size(-2) > image1.size(-1):
        image1 = torch.transpose(image1, -2, -1)
        image2 = torch.transpose(image2, -2, -1)
        transpose_img = True

    nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                    int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

    inference_size = [min(max_inference_size[0], nearest_size[0]), min(max_inference_size[1], nearest_size[1])]

    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                               align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                               align_corners=True)

    results_dict = model(image1, image2,
                         attn_type=attn_type,
                         attn_splits_list=attn_splits_list,
                         corr_radius_list=corr_radius_list,
                         prop_radius_list=prop_radius_list,
                         num_reg_refine=num_reg_refine,
                         task=task,
                         )

    flow_pr = results_dict['flow_preds'][-1]  # [1, 2, H, W] or [1, H, W]

    # resize back
    if task == 'flow':
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]
    else:
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            pred_disp = F.interpolate(flow_pr.unsqueeze(1), size=ori_size,
                                      mode='bilinear',
                                      align_corners=True).squeeze(1)  # [1, H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

    if task == 'flow':
        if transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)

        flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

        output = flow_to_image(flow)  # [H, W, 3]
    else:
        disp = pred_disp[0].cpu().numpy()

        output = vis_disparity(disp, return_rgb=True)

    return Image.fromarray(output)


title = "UniMatch"

description = "<p style='text-align: center'>Optical flow and stereo matching demo for <a href='https://haofeixu.github.io/unimatch/' target='_blank'>Unifying Flow, Stereo and Depth Estimation</a> | <a href='https://arxiv.org/abs/2211.05783' target='_blank'>Paper</a> | <a href='https://github.com/autonomousvision/unimatch' target='_blank'>Code</a> | <a href='https://colab.research.google.com/drive/1r5m-xVy3Kw60U-m5VB-aQ98oqqg_6cab?usp=sharing' target='_blank'>Colab</a><br>Task <strong>flow</strong>: Image1: <strong>video frame t</strong>, Image2: <strong>video frame t+1</strong>; Task <strong>stereo</strong>: Image1: <strong>left</strong> image, Image2: <strong>right</strong> image<br>Simply upload your images or click one of the provided examples.<br><strong>Select the task type according to your input images</strong>.</p>"

examples = [
    ['demo/stereo_drivingstereo_test_2018-07-11-14-48-52_2018-07-11-14-58-34-673_left.jpg',
     'demo/stereo_drivingstereo_test_2018-07-11-14-48-52_2018-07-11-14-58-34-673_right.jpg', 'stereo'],
    ['demo/stereo_middlebury_plants_im0.png', 'demo/stereo_middlebury_plants_im1.png', 'stereo'],
    ['demo/stereo_holopix_left.png', 'demo/stereo_holopix_right.png', 'stereo'],
    ['demo/flow_kitti_test_000197_10.png', 'demo/flow_kitti_test_000197_11.png', 'flow'],
    ['demo/flow_sintel_cave_3_frame_0049.png', 'demo/flow_sintel_cave_3_frame_0050.png', 'flow'],
    ['demo/flow_davis_skate-jump_00059.jpg', 'demo/flow_davis_skate-jump_00060.jpg', 'flow']
]

gr.Interface(
    inference,
    [gr.Image(type="pil", label="Image1"), gr.Image(type="pil", label="Image2"), gr.Radio(choices=['flow', 'stereo'], value='flow', label='Task')],
    gr.Image(type="pil", label="flow/disparity"),
    title=title,
    description=description,
    examples=examples,
).launch(debug=True, quiet=True)
