import contextlib
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import itertools
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from safetensors.torch import load_model
from peft import LoraConfig
import gradio as gr
import pandas as pd

import transformers
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    CLIPProcessor,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    ColorGuiderPixArtModel,
    ColorGuiderSDModel,
    UNet2DConditionModel,
    PixArtTransformer2DModel,
    ColorFlowPixArtAlphaPipeline,
    ColorFlowSDPipeline,
    UniPCMultistepScheduler,
)
from colorflow_utils.utils import *

sys.path.append('./BidirectionalTranslation')
from options.test_options import TestOptions
from models import create_model
from util import util

from huggingface_hub import snapshot_download

model_global_path = snapshot_download(repo_id="TencentARC/ColorFlow", cache_dir='./colorflow/', repo_type="model")
print(model_global_path)


transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])
weight_dtype = torch.float16

# line model
line_model_path = model_global_path + '/LE/erika.pth'
line_model = res_skip()
line_model.load_state_dict(torch.load(line_model_path))
line_model.eval()
line_model.cuda()

# screen model
global opt

opt = TestOptions().parse(model_global_path)
ScreenModel = create_model(opt, model_global_path)
ScreenModel.setup(opt)
ScreenModel.eval()

image_processor = CLIPImageProcessor()
image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_global_path + '/image_encoder/').to('cuda')


examples = [
    [
        "./assets/example_5/input.png", 
        ["./assets/example_5/ref1.png", "./assets/example_5/ref2.png", "./assets/example_5/ref3.png"], 
        "GrayImage(ScreenStyle)", 
        "800x512",  
        0, 
        10 
    ],
    [
        "./assets/example_4/input.jpg", 
        ["./assets/example_4/ref1.jpg", "./assets/example_4/ref2.jpg", "./assets/example_4/ref3.jpg"], 
        "GrayImage(ScreenStyle)", 
        "640x640",  
        0, 
        10 
    ],
    [
        "./assets/example_3/input.png", 
        ["./assets/example_3/ref1.png", "./assets/example_3/ref2.png", "./assets/example_3/ref3.png"], 
        "GrayImage(ScreenStyle)", 
        "800x512", 
        0, 
        10 
    ],
    [
        "./assets/example_2/input.png",  
        ["./assets/example_2/ref1.png", "./assets/example_2/ref2.png", "./assets/example_2/ref3.png"], 
        "GrayImage(ScreenStyle)",  
        "800x512",  
        0,  
        10  
    ],
    [
        "./assets/example_1/input.jpg", 
        ["./assets/example_1/ref1.jpg", "./assets/example_1/ref2.jpg", "./assets/example_1/ref3.jpg"], 
        "Sketch",  
        "640x640", 
        0, 
        10  
    ],
    [
        "./assets/example_0/input.jpg", 
        ["./assets/example_0/ref1.jpg"], 
        "Sketch", 
        "640x640",  
        0, 
        10 
    ],
]

global pipeline
global MultiResNetModel

def load_ckpt(input_style):
    global pipeline
    global MultiResNetModel
    if input_style == "Sketch":
        ckpt_path = model_global_path + '/sketch/'
        rank = 128
        pretrained_model_name_or_path = 'PixArt-alpha/PixArt-XL-2-1024-MS'
        transformer = PixArtTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer", revision=None, variant=None
        )
        pixart_config = get_pixart_config()

        ColorGuider = ColorGuiderPixArtModel.from_pretrained(ckpt_path)

        transformer_lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2", "proj", "linear", "linear_1", "linear_2"]
        )
        transformer.add_adapter(transformer_lora_config)
        ckpt_key_t = torch.load(ckpt_path + 'transformer_lora.bin', map_location='cpu')
        transformer.load_state_dict(ckpt_key_t, strict=False)

        transformer.to('cuda', dtype=weight_dtype)
        ColorGuider.to('cuda', dtype=weight_dtype)
        
        pipeline = ColorFlowPixArtAlphaPipeline.from_pretrained(
            pretrained_model_name_or_path,
            transformer=transformer,
            colorguider=ColorGuider,
            safety_checker=None,
            revision=None,
            variant=None,
            torch_dtype=weight_dtype,
        )
        pipeline = pipeline.to("cuda")
        block_out_channels = [128, 128, 256, 512, 512]
        
        MultiResNetModel = MultiHiddenResNetModel(block_out_channels, len(block_out_channels))
        MultiResNetModel.load_state_dict(torch.load(ckpt_path + 'MultiResNetModel.bin', map_location='cpu'), strict=False)
        MultiResNetModel.to('cuda', dtype=weight_dtype)

    elif input_style == "GrayImage(ScreenStyle)":
        ckpt_path = model_global_path + '/GraySD/'
        rank = 64
        pretrained_model_name_or_path = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", revision=None, variant=None
        )
        ColorGuider = ColorGuiderSDModel.from_pretrained(ckpt_path)
        ColorGuider.to('cuda', dtype=weight_dtype)
        unet.to('cuda', dtype=weight_dtype)
        
        pipeline = ColorFlowSDPipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=unet,
            colorguider=ColorGuider,
            safety_checker=None,
            revision=None,
            variant=None,
            torch_dtype=weight_dtype,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        unet_lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],#ff.net.0.proj ff.net.2
        )
        pipeline.unet.add_adapter(unet_lora_config)
        pipeline.unet.load_state_dict(torch.load(ckpt_path + 'unet_lora.bin', map_location='cpu'), strict=False)
        pipeline = pipeline.to("cuda")
        block_out_channels = [128, 128, 256, 512, 512]
        
        MultiResNetModel = MultiHiddenResNetModel(block_out_channels, len(block_out_channels))
        MultiResNetModel.load_state_dict(torch.load(ckpt_path + 'MultiResNetModel.bin', map_location='cpu'), strict=False)
        MultiResNetModel.to('cuda', dtype=weight_dtype)

    



global cur_input_style
cur_input_style = None



def fix_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def process_multi_images(files):
    images = [Image.open(file.name) for file in files]
    imgs = []
    for i, img in enumerate(images):
        imgs.append(img)
    return imgs 

def extract_lines(image):
    src = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    rows = int(np.ceil(src.shape[0] / 16)) * 16
    cols = int(np.ceil(src.shape[1] / 16)) * 16

    patch = np.ones((1, 1, rows, cols), dtype="float32")
    patch[0, 0, 0:src.shape[0], 0:src.shape[1]] = src

    tensor = torch.from_numpy(patch).cuda()

    with torch.no_grad():
        y = line_model(tensor)

    yc = y.cpu().numpy()[0, 0, :, :]
    yc[yc > 255] = 255
    yc[yc < 0] = 0

    outimg = yc[0:src.shape[0], 0:src.shape[1]]
    outimg = outimg.astype(np.uint8)
    outimg = Image.fromarray(outimg)
    torch.cuda.empty_cache()
    return outimg

def to_screen_image(input_image):
    global opt
    global ScreenModel
    input_image = input_image.convert('RGB')
    input_image = get_ScreenVAE_input(input_image, opt)
    h = input_image['h']
    w = input_image['w']
    ScreenModel.set_input(input_image)
    fake_B, fake_B2, SCR = ScreenModel.forward(AtoB=True)
    images=fake_B2[:,:,:h,:w]
    im = util.tensor2im(images)
    image_pil = Image.fromarray(im)
    torch.cuda.empty_cache()
    return image_pil

def extract_line_image(query_image_, input_style, resolution):
    if resolution == "640x640":
        tar_width = 640
        tar_height = 640
    elif resolution == "512x800":
        tar_width = 512
        tar_height = 800
    elif resolution == "800x512":
        tar_width = 800
        tar_height = 512
    else:
        gr.Info("Unsupported resolution")

    query_image = process_image(query_image_, int(tar_width*1.5), int(tar_height*1.5))
    if input_style == "GrayImage(ScreenStyle)":
        extracted_line = to_screen_image(query_image)
        extracted_line = Image.blend(extracted_line.convert('L').convert('RGB'), query_image.convert('L').convert('RGB'), 0.5)
        input_context = extracted_line
    elif input_style == "Sketch":
        query_image = query_image.convert('L').convert('RGB')
        extracted_line = extract_lines(query_image)
        extracted_line = extracted_line.convert('L').convert('RGB')
        input_context = extracted_line
    torch.cuda.empty_cache()
    return input_context, extracted_line, input_context  

def colorize_image(VAE_input, input_context, reference_images, resolution, seed, input_style, num_inference_steps):
    if VAE_input is None or input_context is None:
        gr.Info("Please preprocess the image first")
        raise ValueError("Please preprocess the image first")
    global cur_input_style
    global pipeline
    global MultiResNetModel
    if input_style != cur_input_style:
        gr.Info(f"Loading {input_style} model...")
        load_ckpt(input_style)
        cur_input_style = input_style
        gr.Info(f"{input_style} model loaded")
    reference_images = process_multi_images(reference_images)
    fix_random_seeds(seed)
    if resolution == "640x640":
        tar_width = 640
        tar_height = 640
    elif resolution == "512x800":
        tar_width = 512
        tar_height = 800
    elif resolution == "800x512":
        tar_width = 800
        tar_height = 512
    else:
        gr.Info("Unsupported resolution")
    validation_mask = Image.open('./assets/mask.png').convert('RGB').resize((tar_width*2, tar_height*2))
    gr.Info("Image retrieval in progress...")
    query_image_bw = process_image(input_context, int(tar_width), int(tar_height))
    query_image = query_image_bw.convert('RGB')
    query_image_vae = process_image(VAE_input, int(tar_width*1.5), int(tar_height*1.5))
    reference_images = [process_image(ref_image, tar_width, tar_height) for ref_image in reference_images]
    query_patches_pil = process_image_Q_varres(query_image, tar_width, tar_height)
    reference_patches_pil = []
    for reference_image in reference_images:
        reference_patches_pil += process_image_ref_varres(reference_image, tar_width, tar_height)
    combined_image = None
    with torch.no_grad():
        clip_img = image_processor(images=query_patches_pil, return_tensors="pt").pixel_values.to(image_encoder.device, dtype=image_encoder.dtype)
        query_embeddings = image_encoder(clip_img).image_embeds
        reference_patches_pil_gray = [rimg.convert('RGB').convert('RGB') for rimg in reference_patches_pil]
        clip_img = image_processor(images=reference_patches_pil_gray, return_tensors="pt").pixel_values.to(image_encoder.device, dtype=image_encoder.dtype)
        reference_embeddings = image_encoder(clip_img).image_embeds
        cosine_similarities = F.cosine_similarity(query_embeddings.unsqueeze(1), reference_embeddings.unsqueeze(0), dim=-1)
        sorted_indices = torch.argsort(cosine_similarities, descending=True, dim=1).tolist()
        top_k = 3
        top_k_indices = [cur_sortlist[:top_k] for cur_sortlist in sorted_indices]
        combined_image = Image.new('RGB', (tar_width * 2, tar_height * 2), 'white')
        combined_image.paste(query_image_bw.resize((tar_width, tar_height)), (tar_width//2, tar_height//2))
        idx_table = {0:[(1,0), (0,1), (0,0)], 1:[(1,3), (0,2),(0,3)], 2:[(2,0),(3,1), (3,0)], 3:[(2,3), (3,2),(3,3)]}
        for i in range(2):
            for j in range(2):
                idx_list = idx_table[i * 2 + j]
                for k in range(top_k):
                    ref_index = top_k_indices[i * 2 + j][k]
                    idx_y = idx_list[k][0]
                    idx_x = idx_list[k][1]
                    combined_image.paste(reference_patches_pil[ref_index].resize((tar_width//2-2, tar_height//2-2)), (tar_width//2 * idx_x + 1, tar_height//2 * idx_y + 1))
    gr.Info("Model inference in progress...")
    generator = torch.Generator(device='cuda').manual_seed(seed)
    image = pipeline(
        "manga", cond_image=combined_image, cond_mask=validation_mask, num_inference_steps=num_inference_steps, generator=generator
    ).images[0]
    gr.Info("Post-processing image...")
    with torch.no_grad():
        width, height = image.size
        new_width = width // 2
        new_height = height // 2
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        center_crop = image.crop((left, top, right, bottom))
        up_img = center_crop.resize(query_image_vae.size)
        test_low_color = transform(up_img).unsqueeze(0).to('cuda', dtype=weight_dtype)
        query_image_vae = transform(query_image_vae).unsqueeze(0).to('cuda', dtype=weight_dtype)

        h_color, hidden_list_color = pipeline.vae._encode(test_low_color,return_dict = False, hidden_flag = True)
        h_bw, hidden_list_bw = pipeline.vae._encode(query_image_vae, return_dict = False, hidden_flag = True)

        hidden_list_double = [torch.cat((hidden_list_color[hidden_idx], hidden_list_bw[hidden_idx]), dim = 1) for hidden_idx in range(len(hidden_list_color))]


        hidden_list = MultiResNetModel(hidden_list_double)
        output = pipeline.vae._decode(h_color.sample(),return_dict = False, hidden_list = hidden_list)[0]

        output[output > 1] = 1
        output[output < -1] = -1
        high_res_image = Image.fromarray(((output[0] * 0.5 + 0.5).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
    gr.Info("Colorization complete!")
    torch.cuda.empty_cache()
    return high_res_image, up_img, image, query_image_bw

with gr.Blocks() as demo:
    gr.HTML(
    """
<div style="text-align: center;">
    <h1 style="text-align: center; font-size: 3em;">🎨 ColorFlow:</h1>
    <h3 style="text-align: center; font-size: 1.8em;">Retrieval-Augmented Image Sequence Colorization</h3>
    <p style="text-align: center; font-weight: bold;">
        <a href="https://zhuang2002.github.io/ColorFlow/">Project Page</a> | 
        <a href="">ArXiv Preprint</a> | 
        <a href="https://github.com/TencentARC/ColorFlow">GitHub Repository</a>
    </p>
    <p style="text-align: center; font-weight: bold;">
        NOTE: Each time you switch the input style, the corresponding model will be reloaded, which may take some time. Please be patient.
    </p>
    <p style="text-align: left; font-size: 1.1em;">
        Welcome to the demo of <strong>ColorFlow</strong>. Follow the steps below to explore the capabilities of our model:
    </p>
</div>
<div style="text-align: left; margin: 0 auto;">
    <ol style="font-size: 1.1em;">
        <li>Choose input style: GrayImage(ScreenStyle) or Sketch.</li>
        <li>Upload your image: Use the 'Upload' button to select the image you want to colorize.</li>
        <li>Preprocess the image: Click the 'Preprocess' button to decolorize the image.</li>
        <li>Upload reference images: Upload multiple reference images to guide the colorization.</li>
        <li>Set sampling parameters (optional): Adjust the settings and click the <b>Colorize</b> button.</li>
    </ol>
    <p>
        ⏱️ <b>ZeroGPU Time Limit</b>: Hugging Face ZeroGPU has an inference time limit of 180 seconds. You may need to log in with a free account to use this demo. Large sampling steps might lead to timeout (GPU Abort). In that case, please consider logging in with a Pro account or running it on your local machine.
    </p>
</div>
<div style="text-align: center;">
    <p style="text-align: center; font-weight: bold;">
        注意：每次切换输入样式时，相应的模型将被重新加载，可能需要一些时间。请耐心等待。
    </p>
    <p style="text-align: left; font-size: 1.1em;">
        欢迎使用 <strong>ColorFlow</strong> 演示。请按照以下步骤探索我们模型的能力：
    </p>
</div>
<div style="text-align: left; margin: 0 auto;">
    <ol style="font-size: 1.1em;">
        <li>选择输入样式：灰度图(ScreenStyle)、线稿。</li>
        <li>上传您的图像：使用“上传”按钮选择要上色的图像。</li>
        <li>预处理图像：点击“预处理”按钮以去色图像。</li>
        <li>上传参考图像：上传多张参考图像以指导上色。</li>
        <li>设置采样参数（可选）：调整设置并点击 <b>上色</b> 按钮。</li>
    </ol>
    <p>
        ⏱️ <b>ZeroGPU时间限制</b>：Hugging Face ZeroGPU 的推理时间限制为 180 秒。您可能需要使用免费帐户登录以使用此演示。大采样步骤可能会导致超时（GPU 中止）。在这种情况下，请考虑使用专业帐户登录或在本地计算机上运行。
    </p>
</div>
    """
)
    VAE_input = gr.State()
    input_context = gr.State()
    # example_loading = gr.State(value=None)
    
    with gr.Column():
        with gr.Row():
            input_style = gr.Radio(["GrayImage(ScreenStyle)", "Sketch"], label="Input Style", value="GrayImage(ScreenStyle)")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Image to Colorize")
                resolution = gr.Radio(["640x640", "512x800", "800x512"], label="Select Resolution(Width*Height)", value="640x640")
                extract_button = gr.Button("Preprocess (Decolorize)")
            extracted_image = gr.Image(type="pil", label="Decolorized Result")
        with gr.Row():
            reference_images = gr.Files(label="Reference Images (Upload multiple)", file_count="multiple")
            with gr.Column():
                output_gallery = gr.Gallery(label="Colorization Results", type="pil")
                seed = gr.Slider(label="Random Seed", minimum=0, maximum=100000, value=0, step=1)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=4, maximum=100, value=10, step=1)
                colorize_button = gr.Button("Colorize")
    
    # progress_text = gr.Textbox(label="Progress", interactive=False)
    
    
    extract_button.click(
        extract_line_image, 
        inputs=[input_image, input_style, resolution], 
        outputs=[extracted_image, VAE_input, input_context]
    )
    colorize_button.click(
        colorize_image, 
        inputs=[VAE_input, input_context, reference_images, resolution, seed, input_style, num_inference_steps], 
        outputs=output_gallery
    )

    with gr.Column():
        gr.Markdown("### Quick Examples")
        gr.Examples(
            examples=examples,
            inputs=[input_image, reference_images, input_style, resolution, seed, num_inference_steps],
            label="Examples",
            examples_per_page=6,
        )
demo.launch(share=True)