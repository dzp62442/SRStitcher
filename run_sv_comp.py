"""
进行与 ChatStitch SV-UDIS 的对比实验
"""

import torch
from PIL import Image
import os
import cv2
from diffusers import ControlNetModel, AutoencoderKL
import math
from omegaconf import OmegaConf
import argparse
import numpy as np

from util.preprocessing import merged_img, preprocess_image, preprocess_map, make_inpaint_condition
from util.weighted_mask import make_mask

# SRStitcher Pipes
from pipes.diff_pipe_inpaint import DiffusionDiffInpaintingPipeline
from pipes.diff_pipe_SD2 import StableDiffusionDiffImg2ImgPipeline
from pipes.diff_pipe_unclip import StableUnCLIPImg2ImgPipeline
from pipes.diff_pipe_control import StableDiffusionControlNetInpaintPipeline

from sv_comp.udis2_warp import UDIS2Warp
from sv_comp.Warp.Codes.dataset import MultiWarpDataset
import yaml
from loguru import logger

PROJ_ROOT = os.path.abspath(os.path.dirname(__file__)) 
print(PROJ_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(description="SRStitcher.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inpaint_config_sv_comp.yaml",
    )
    args = parser.parse_args()
    return args


def calculate_k(image_width, lamb):
    k = image_width / lamb
    k = math.ceil(k) * 10
    return (k, k)


def main(cfg):
    # 初始化 UDIS2 的 warp 网络
    udis2_warp_cfg = cfg.udis2_warp
    udis2_warp = UDIS2Warp(udis2_warp_cfg)

    # 加载数据集
    with open('sv_comp/intrinsics.yaml', 'r', encoding='utf-8') as file:
        intrinsics = yaml.safe_load(file)
    dataset = MultiWarpDataset(config=udis2_warp_cfg, intrinsics=intrinsics, is_train=False)

    # 初始化 SRStitcher 网络
    device = cfg.device
    if cfg.mode == "SD2-inpaint":
        pipe = DiffusionDiffInpaintingPipeline.from_pretrained(cfg.pretrained_model_name_or_path,
                                                               safety_checker=None,
                                                               torch_dtype=torch.float16).to(device)
    elif cfg.mode == "SD2":
        pipe = StableDiffusionDiffImg2ImgPipeline.from_pretrained(cfg.pretrained_model_name_or_path,
                                                                  safety_checker=None,
                                                                  torch_dtype=torch.float16).to(device)
    elif cfg.mode == "unclipSD2":
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(cfg.pretrained_model_name_or_path,
                                                           safety_checker=None,
                                                           torch_dtype=torch.float16).to(device)
    elif cfg.mode == "controlnet":
        controlnet = ControlNetModel.from_pretrained(cfg.controlnet_path, torch_dtype=torch.float16).to(device)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(cfg.pretrained_model_name_or_path,
                                                                        controlnet=controlnet,
                                                                        torch_dtype=torch.float16,
                                                                        safety_checker=None).to(device)

    path = cfg.datapath
    save_dir = cfg.save_dir
    generator = torch.Generator(device="cuda").manual_seed(cfg.seed)

    # Check if the folder exists
    if not os.path.exists(save_dir):
        # Folder does not exist, so create the folder
        os.makedirs(save_dir)
        print(f"Folder '{save_dir}' created.")

    if not os.path.exists('coarse'+save_dir):
        # Folder does not exist, so create the folder
        os.makedirs('coarse'+save_dir)
        print(f"Folder '{'coarse'+save_dir}' created.")

    R = cfg.R

    failure_num = 0  # 拼接失败的次数
    for idx in range(len(dataset)):
        # if idx > 2:
        #     break
        sample = dataset[idx]
        input_imgs, input_masks = sample[0], sample[1]
        middle_stitch_result = None  # 中间拼接结果

        for k in range(udis2_warp_cfg['input_img_num'] - 1):
            # 创建保存结果文件夹
            batch_path = dataset.get_path(idx)
            path_coarse = os.path.join(batch_path, 'coarse_srstitcher/')
            os.makedirs(path_coarse, exist_ok=True)
            path_result = os.path.join(batch_path, 'srstitcher/')
            os.makedirs(path_result, exist_ok=True)
            path_warp = os.path.join(batch_path, 'warp/')
            os.makedirs(path_warp, exist_ok=True)
            path_mask = os.path.join(batch_path, 'mask/')
            os.makedirs(path_mask, exist_ok=True)
            
            # 使用 UDIS2 的 warp 网络进行拼接
            if k == 0:
                input1 = dataset.to_tensor(input_imgs[k])
                input2 = dataset.to_tensor(input_imgs[k+1])
            else:
                input1 = dataset.to_tensor(middle_stitch_result)
                input2 = dataset.to_tensor(input_imgs[k+1])
            out_dict = udis2_warp.test([input1, input2])
            if (not out_dict['success']):  # 拼接失败情况处理
                logger.warning(f'Failed to stitch {dataset.get_path(idx)} !!!')
                logger.warning(f'batch {idx} stitch fail when k={k} !!!')
                failure_num += 1
                torch.cuda.empty_cache()
                break
            warp1 = out_dict['warp1'].astype(np.uint8)
            warp2 = out_dict['warp2'].astype(np.uint8)
            mask1 = cv2.cvtColor(out_dict['mask1'].astype(np.uint8), cv2.COLOR_BGR2GRAY)*255
            mask2 = cv2.cvtColor(out_dict['mask2'].astype(np.uint8), cv2.COLOR_BGR2GRAY)*255
            
            # 掩码中白色区域可能存在细小黑点，通过形态学操作去除
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 形态学操作去除细小黑点
            mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)  # 闭操作：先膨胀后腐蚀，填充小的黑点（在白色区域中）
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)  # 开操作：先腐蚀后膨胀，去除小的白点（在黑色背景上）
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            
            # 保存 warp 和 mask
            name = f"{udis2_warp_cfg['input_img_num']}_{k+2}.jpg"
            cv2.imwrite(os.path.join(path_warp, name.replace('.jpg', '_warp1.jpg')), warp1)
            cv2.imwrite(os.path.join(path_mask, name.replace('.jpg', '_mask1.jpg')), mask1)
            cv2.imwrite(os.path.join(path_warp, name.replace('.jpg', '_warp2.jpg')), warp2)
            cv2.imwrite(os.path.join(path_mask, name.replace('.jpg', '_mask2.jpg')), mask2)

            _, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
            _, mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)

            coarse_fusion_img = merged_img(warp1, warp2, mask1, mask2)

            K = calculate_k(coarse_fusion_img.shape[1], cfg.lamb)

            h, w, c = coarse_fusion_img.shape

            newsize = (cfg.size, cfg.size)

            map, content_mask, map2, map3= make_mask(mask1, mask2, K, cfg.epsilon1, cfg.epsilon2)
            map = preprocess_map(map, newsize).to(device)
            map3 = preprocess_map(map3, newsize).to(device)

            name = f"{udis2_warp_cfg['input_img_num']}_{k+2}.jpg"
            Image.fromarray(cv2.cvtColor(coarse_fusion_img, cv2.COLOR_BGR2RGB)).save(os.path.join(path_coarse, name))

            coarse_rectangling_img = Image.fromarray(
                cv2.cvtColor(cv2.inpaint(coarse_fusion_img, content_mask, R, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB))

            image = preprocess_image(coarse_rectangling_img, newsize)

            if cfg.mode == "SD2-inpaint":
                map2 = Image.fromarray(map2).resize((512, 512))
                edited_image = pipe(
                    prompt=[""],
                    image=image,
                    guidance_scale=cfg.guidance_scale,
                    num_images_per_prompt=1,
                    mask_image=map2,
                    generator=generator,
                    map=map,
                    num_inference_steps=cfg.num_inference_steps,
                ).images[0]
            elif cfg.mode == "SD2":
                edited_image = pipe(
                    prompt=[""],
                    image=image,
                    guidance_scale=cfg.guidance_scale,
                    num_images_per_prompt=1,
                    generator=generator,
                    map=map3,
                    num_inference_steps=cfg.num_inference_steps,
                ).images[0]
            elif cfg.mode == "unclipSD2":
                edited_image = pipe(prompt=[""],
                                    image=image,
                                    guidance_scale=7.5,
                                    num_images_per_prompt=1,
                                    generator=generator,
                                    map=map3,
                                    num_inference_steps=50).images[0]
            elif cfg.mode == "controlnet":
                map2 = Image.fromarray(map2).resize((512, 512))
                control_image = make_inpaint_condition(image, map2)
                edited_image = pipe(prompt=[""],
                                    image=image,
                                    guidance_scale=7.5,
                                    num_images_per_prompt=1,
                                    generator=generator,
                                    mask_image=Image.fromarray(mask2).resize((512, 512)),
                                    map=map,
                                    control_image=control_image.to(device),
                                    num_inference_steps=50).images[0]

            # 处理中间拼接结果
            middle_stitch_result = np.array(edited_image)
            middle_stitch_result = cv2.resize(middle_stitch_result, (udis2_warp_cfg['net_input_width'], udis2_warp_cfg['net_input_height']))
            middle_stitch_result = cv2.cvtColor(middle_stitch_result, cv2.COLOR_RGB2BGR)

            # 保存结果
            edited_image = edited_image.resize((w, h))
            name = f"{udis2_warp_cfg['input_img_num']}_{k+2}.jpg"
            edited_image.save(os.path.join(path_result, name))

        print(f'processing image {idx}/{len(dataset)} completed')


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    main(config)