#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange
import math # Added for utility functions
import argparse # Added for command-line argument parsing

from diffsynth import ModelManager, FlashVSRFullPipeline
from utils.utils import Causal_LQ4x_Proj
import diffsynth.models.wan_video_dit

def clean_vram():
    torch.cuda.empty_cache()

def tensor2video(frames: torch.Tensor, return_tensor=False):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames_float = (frames.float() + 1) / 2.0 # Scale to [0, 1]
    if return_tensor:
        return frames_float # Return tensor in [0, 1] range
    
    frames_np = (frames_float * 255).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames_pil = [Image.fromarray(frame) for frame in frames_np]
    return frames_pil

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def is_video(path): 
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0                                              # CHW in [-1,1]
    return t.to(dtype)

def save_video(frames, save_path, fps=30, quality=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
        w.append_data(np.array(f))
    w.close()

def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []
    stride = tile_size - overlap
    num_rows, num_cols = math.ceil((height - overlap) / stride), math.ceil((width - overlap) / stride)
    for r in range(num_rows):
        for c in range(num_cols):
            y1, x1 = r * stride, c * stride
            y2, x2 = min(y1 + tile_size, height), min(x1 + tile_size, width)
            if y2 - y1 < tile_size: y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size: x1 = max(0, x2 - tile_size)
            coords.append((x1, y1, x2, y2))
    return coords

def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    return mask

def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH

def upscale_then_center_crop(img: Image.Image, scale: int, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW, sH = w0 * scale, h0 * scale
    # 先放大
    up = img.resize((sW, sH), Image.BICUBIC)
    # 中心裁剪
    l = max(0, (sW - tW) // 2); t = max(0, (sH - tH) // 2)
    return up.crop((l, t, l + tW, t + tH))

def prepare_tensors_for_dit_tiling(path: str, dtype=torch.bfloat16):
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0: raise FileNotFoundError(f"No images in {path}")
        with Image.open(paths0[0]) as _img0: w0, h0 = _img0.size
        frames = [torch.from_numpy(np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0).to(dtype) for p in tqdm(paths0, desc="Loading images")]
        return torch.stack(frames, 0), 30
    if is_video(path):
        with imageio.get_reader(path) as rdr:
            meta = rdr.get_meta_data()
            fps = meta.get('fps', 30)
            frames = [torch.from_numpy(frame_data.astype(np.float32) / 255.0).to(dtype) for frame_data in tqdm(rdr, desc="Loading video frames")]
        return torch.stack(frames, 0), fps
    raise ValueError(f"Unsupported input: {path}")

def prepare_input_tensor_for_tile(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    multiple = 128
    sW, sH, tW, tH = w0 * scale, h0 * scale, max(multiple, (w0 * scale // multiple) * multiple), max(multiple, (h0 * scale // multiple) * multiple)
    num_frames_padded = largest_8n1_leq(N0 + 4)
    if F == 0: raise RuntimeError(f"Not enough frames. Got {N0 + 4}.")

    frames = []
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_bchw = frame_slice.permute(2, 0, 1).unsqueeze(0)
        upscaled_tensor = F.interpolate(tensor_bchw, size=(h0 * scale, w0 * scale), mode='bicubic', align_corners=False)
        l, t = max(0, (w0 * scale - tW) // 2), max(0, (h0 * scale - tH) // 2)
        cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]
        tensor_out = (cropped_tensor.squeeze(0) * 2.0 - 1.0).to('cpu').to(dtype)
        frames.append(tensor_out)
    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    clean_vram()
    return vid_final, tH, tW, num_frames_padded

def prepare_input_tensor(path: str, scale: int = 4, dtype=torch.bfloat16, device='cuda'):
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")
        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {N0}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled Resolution (x{scale}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        paths = paths0 + [paths0[-1]] * 4
        F = largest_8n1_leq(len(paths))
        if F == 0:
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(paths)}.")
        paths = paths[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F-4}")

        frames = []
        for p in paths:
            with Image.open(p).convert('RGB') as img:
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)   
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))             
        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)             
        fps = 30
        return vid, tH, tW, F, fps

    if is_video(path):
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert('RGB')
        w0, h0 = first.size

        meta = {}
        try:
            meta = rdr.get_meta_data()
        except Exception:
            pass
        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get('nframes', None)
                if isinstance(nf, int) and nf > 0:
                    return nf
            except Exception:
                pass
            try:
                return r.count_frames()
            except Exception:
                n = 0
                try:
                    while True:
                        r.get_data(n); n += 1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {total} | FPS: {fps}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled Resolution (x{scale}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        idx = list(range(total)) + [total - 1] * 4
        F = largest_8n1_leq(len(idx))
        if F == 0:
            rdr.close()
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(idx)}.")
        idx = idx[:F]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {F-4}")

        frames = []
        try:
            for i in idx:
                img = Image.fromarray(rdr.get_data(i)).convert('RGB')
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        finally:
            try:
                rdr.close()
            except Exception:
                pass

        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)   # 1 C F H W
        return vid, tH, tW, F, fps

    raise ValueError(f"Unsupported input: {path}")

def init_pipeline(num_persistent_param_in_dit):
    print(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FlashVSR-v1.1")
    mm.load_models([
        os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors"),
        os.path.join(model_path, "Wan2.1_VAE.pth"),
    ])
    pipe = FlashVSRFullPipeline.from_model_manager(mm, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    LQ_proj_in_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    if os.path.exists(LQ_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu"), strict=True)

    pipe.denoising_model().LQ_proj_in.to('cuda')
    pipe.vae.model.encoder = None
    pipe.vae.model.conv1 = None
    pipe.to('cuda'); pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_param_in_dit)
    pipe.init_cross_kv(); pipe.load_models_to_device(["dit","vae"])
    return pipe

diffsynth.models.wan_video_dit.USE_BLOCK_ATTN = True

def run_inference(
    video_path,
    scale,
    tiled_vae,  # Renamed from 'tiled' to avoid confusion with 'tiled_dit'
    tile_size_h,
    tile_size_w,
    tile_stride_h,
    tile_stride_w,
    num_persistent_param_in_dit,
    tiled_dit,
    dit_tile_size,
    dit_tile_overlap
):
    pipe = init_pipeline(num_persistent_param_in_dit)
    
    dtype, device = torch.bfloat16, 'cuda'
    
    if tiled_dit:
        frames, original_fps = prepare_tensors_for_dit_tiling(video_path, dtype=dtype)
        N, H, W, C = frames.shape
        tile_coords = calculate_tile_coords(H, W, dit_tile_size, dit_tile_overlap)
        
        num_aligned_frames = largest_8n1_leq(N + 4) - 4
        final_output_canvas, weight_sum_canvas = torch.zeros((num_aligned_frames, H*scale, W*scale, C), dtype=torch.float32), torch.zeros((num_aligned_frames, H*scale, W*scale, C), dtype=torch.float32)
        
        for i in tqdm(range(len(tile_coords)), desc="[FlashVSR] Processing tiles"):
            x1, y1, x2, y2 = tile_coords[i]
            input_tile = frames[:, y1:y2, x1:x2, :]
            
            LQ_tile, th, tw, F_frames = prepare_input_tensor_for_tile(input_tile, device, scale=scale, dtype=dtype)
            LQ_tile = LQ_tile.to(device)
            
            output_tile_gpu = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
                tiled=tiled_vae,
                tile_size=(tile_size_h, tile_size_w),
                tile_stride=(tile_stride_h, tile_stride_w),
                LQ_video=LQ_tile, num_frames=F_frames, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=2.0*768*1280/(th*tw), 
                kv_ratio=3.0,
                local_range=11,
                color_fix = True,
            )
            processed_tile_cpu_tensor = tensor2video(output_tile_gpu, return_tensor=True).cpu() # T H W C, [0, 1]
            
            mask = create_feather_mask((processed_tile_cpu_tensor.shape[1], processed_tile_cpu_tensor.shape[2]), dit_tile_overlap * scale).to(device).permute(0, 2, 3, 1) # 1 H W C
            # Expand mask to match number of frames
            mask_expanded = mask.repeat(processed_tile_cpu_tensor.shape[0], 1, 1, 1) # T H W C
            
            x1_s, y1_s = x1 * scale, y1 * scale
            x2_s, y2_s = x1_s + processed_tile_cpu_tensor.shape[2], y1_s + processed_tile_cpu_tensor.shape[1]
            
            final_output_canvas[:, y1_s:y2_s, x1_s:x2_s, :] += processed_tile_cpu_tensor * mask_expanded
            weight_sum_canvas[:, y1_s:y2_s, x1_s:x2_s, :] += mask_expanded
            del LQ_tile, output_tile_gpu, input_tile; clean_vram()
            
        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        video_output_tensor = final_output_canvas / weight_sum_canvas # T H W C, [0, 1]
        
        # Convert the final accumulated tensor to PIL Images
        video = (video_output_tensor * 255).clip(0, 255).numpy().astype(np.uint8)
        video = [Image.fromarray(frame) for frame in video]
        fps = original_fps

            # Output of pipe is (1 C F H W) so (C T H W) after squeeze(0) if B=1
            output_tile_tensor = output_tile_gpu # This is the tensor (C T H W)
            output_tile_tensor = rearrange(output_tile_tensor, "C T H W -> T H W C") # T H W C, range [-1, 1]

            mask = create_feather_mask((output_tile_tensor.shape[1], output_tile_tensor.shape[2]), dit_tile_overlap * scale).to(device).permute(0, 2, 3, 1) # 1 H W C
            # Expand mask to match number of frames
            mask_expanded = mask.repeat(output_tile_tensor.shape[0], 1, 1, 1) # T H W C

            x1_s, y1_s = x1 * scale, y1 * scale
            x2_s, y2_s = x1_s + output_tile_tensor.shape[2], y1_s + output_tile_tensor.shape[1]
            
            # Convert output_tile_tensor to [0, 1] range for accumulation
            output_tile_tensor = (output_tile_tensor + 1.0) / 2.0 
            
            final_output_canvas[:, y1_s:y2_s, x1_s:x2_s, :] += output_tile_tensor.cpu() * mask_expanded.cpu()
            weight_sum_canvas[:, y1_s:y2_s, x1_s:x2_s, :] += mask_expanded.cpu()
            del LQ_tile, output_tile_gpu, input_tile; clean_vram()
            
        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        video_output_tensor = final_output_canvas / weight_sum_canvas
        
        # Convert the final accumulated tensor to PIL Images (0-255)
        video_output_tensor = (video_output_tensor * 255.0).clip(0, 255).numpy().astype(np.uint8)
        video = [Image.fromarray(frame) for frame in video_output_tensor]
        fps = original_fps
    else:
        # Non-tiled DiT logic
        LQ, th, tw, F, fps = prepare_input_tensor(video_path, scale=scale, dtype=dtype, device=device)

        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0, 
            tiled=tiled_vae,
            tile_size=(tile_size_h, tile_size_w),
            tile_stride=(tile_stride_h, tile_stride_w),
            LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
            topk_ratio=2.0*768*1280/(th*tw), 
            kv_ratio=3.0,
            local_range=11,
            color_fix = True,
        )
        video = tensor2video(video)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"FlashVSR_v1.1_Full_{os.path.basename(video_path).split('.')[0]}_seed0.mp4"
    output_path = os.path.join(output_dir, output_filename)
    save_video(video, output_path, fps=fps, quality=6)
    
    del pipe
    clean_vram()
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashVSR v1.1 Full Inference Script")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video or folder of images.")
    parser.add_argument("--scale", type=int, default=4, help="Upscaling factor.")
    parser.add_argument("--tiled_vae", action="store_true", help="Enable VAE tiling.")
    parser.add_argument("--vae_tile_size_h", type=int, default=32, help="VAE Tile Size Height.")
    parser.add_argument("--vae_tile_size_w", type=int, default=64, help="VAE Tile Size Width.")
    parser.add_argument("--vae_tile_stride_h", type=int, default=16, help="VAE Tile Stride Height.")
    parser.add_argument("--vae_tile_stride_w", type=int, default=32, help="VAE Tile Stride Width.")
    parser.add_argument("--num_persistent_param_in_dit", type=int, default=0, help="Number of persistent parameters in DiT for VRAM management.")
    parser.add_argument("--tiled_dit", action="store_true", help="Enable DiT tiling.")
    parser.add_argument("--dit_tile_size", type=int, default=256, help="DiT Tile Size.")
    parser.add_argument("--dit_tile_overlap", type=int, default=24, help="DiT Tile Overlap.")

    args = parser.parse_args()

    # Call run_inference with parsed arguments
    run_inference(
        video_path=args.input_video,
        scale=args.scale,
        tiled_vae=args.tiled_vae,
        tile_size_h=args.vae_tile_size_h,
        tile_size_w=args.vae_tile_size_w,
        tile_stride_h=args.vae_tile_stride_h,
        tile_stride_w=args.vae_tile_stride_w,
        num_persistent_param_in_dit=args.num_persistent_param_in_dit,
        tiled_dit=args.tiled_dit,
        dit_tile_size=args.dit_tile_size,
        dit_tile_overlap=args.dit_tile_overlap
    )
