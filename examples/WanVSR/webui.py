import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import gradio as gr
import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import imageio
from tqdm import tqdm
from einops import rearrange
import re
import math
import glob

from diffsynth import ModelManager, FlashVSRFullPipeline
from utils.utils import Causal_LQ4x_Proj
import diffsynth.models.wan_video_vae

# --- Monkey-patch for build_1d_mask ---
def corrected_build_1d_mask(self, length, is_bound_0, is_bound_1, border_width):
    x = torch.ones((length,))
    if length == 0:
        return x
    if not is_bound_0:
        slice_len = min(length, border_width)
        ramp = (torch.arange(border_width, device=x.device) + 1) / border_width
        x[:slice_len] = ramp[:slice_len]
    if not is_bound_1:
        slice_len = min(length, border_width)
        ramp = torch.flip((torch.arange(border_width, device=x.device) + 1) / border_width, dims=(0,))
        x[-slice_len:] = ramp[-slice_len:]
    return x

diffsynth.models.wan_video_vae.WanVideoVAE.build_1d_mask = corrected_build_1d_mask
# --- End of Monkey-patch ---

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def is_video(path): 
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def tensor2video(frames: torch.Tensor, return_tensor=False):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames_float = (frames.float() + 1) / 2.0 # Scale to [0, 1]
    if return_tensor:
        return frames_float # Return tensor in [0, 1] range
    
    frames_np = (frames_float * 255).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames_pil = [Image.fromarray(frame) for frame in frames_np]
    return frames_pil

def save_video(frames, save_path, fps=30, quality=5):
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
        w.append_data(np.array(f))
    w.close()

def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0                                              # CHW in [-1,1]
    return t.to(dtype)

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
    
def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def prepare_input_tensor(path: str, scale: int = 4, dtype=torch.bfloat16, device='cuda'):
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

def init_pipeline(num_persistent_param_in_dit):
    # print(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device())) # Debug print
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
    if W > 0:
        mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
        mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    if H > 0:
        mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
        mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    return mask

def clean_vram():
    torch.cuda.empty_cache()

def prepare_tensors(path: str, dtype=torch.bfloat16):
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
    if num_frames_padded == 0: raise RuntimeError(f"Not enough frames. Got {N0 + 4}.")

    frames = []
    for i in range(num_frames_padded):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device) / 255.0
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

# --- Core Upscaling Logic ---
def _upscale_core_logic(image_pil: Image.Image, scale: int, tiled: bool, tile_size_h: int, tile_size_w: int, tile_stride_h: int, tile_stride_w: int, num_persistent_param_in_dit: int, tiled_dit: bool, dit_tile_size: int, dit_tile_overlap: int, num_inference_steps_val: int, num_frames_for_sequence_val: int) -> Image.Image:
    
    pipe = init_pipeline(num_persistent_param_in_dit)
    
    dtype, device = torch.bfloat16, 'cuda'
    
    frame_tensor = torch.from_numpy(np.array(image_pil)).to(dtype=dtype).unsqueeze(0) # 1 H W C
    
    if tiled_dit:
        num_frames_for_sequence = num_frames_for_sequence_val
        frames = frame_tensor.repeat(num_frames_for_sequence, 1, 1, 1) # N H W C

        N, H, W, C = frames.shape
        tile_coords = calculate_tile_coords(H, W, dit_tile_size, dit_tile_overlap)
        
        final_output_canvas = torch.zeros((H * scale, W * scale, C), dtype=torch.float32)
        weight_sum_canvas = torch.zeros_like(final_output_canvas, dtype=torch.float32)

        print(f"Processing in {len(tile_coords)} tiles...")
        for i, (x1, y1, x2, y2) in enumerate(tqdm(tile_coords, desc="[FlashVSR] Processing image tiles")):
            input_tile = frames[:, y1:y2, x1:x2, :] # N H_tile W_tile C
            
            LQ_tile, th, tw, F = prepare_input_tensor_for_tile(input_tile, device, scale=scale, dtype=dtype)
            LQ_tile = LQ_tile.to(device)

            with torch.no_grad():
                processed_tile_output_tensor = pipe(
                    prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=num_inference_steps_val, seed=0,
                    tiled=tiled,
                    tile_size=(tile_size_h, tile_size_w),
                    tile_stride=(tile_stride_h, tile_stride_w),
                    LQ_video=LQ_tile, 
                    num_frames=F,
                    height=th,
                    width=tw,
                    is_full_block=False, 
                    if_buffer=True,
                    topk_ratio=2.0*768*1280/(th*tw), 
                    kv_ratio=3.0,
                    local_range=11,
                    color_fix = True,
                )
            
            single_frame_output_tensor = processed_tile_output_tensor[:, 0, :, :] # C H_tile_out W_tile_out
            
            processed_tile_tensor_0_1 = ((single_frame_output_tensor + 1.0) / 2.0).permute(1, 2, 0).cpu().float() # H_tile_out W_tile_out C
            
            mask = create_feather_mask((processed_tile_tensor_0_1.shape[0], processed_tile_tensor_0_1.shape[1]), dit_tile_overlap * scale).cpu() # H_tile_out W_tile_out
            
            mask_expanded = mask.squeeze(0).squeeze(0).unsqueeze(2).repeat(1, 1, 3)

            x1_s, y1_s = x1 * scale, y1 * scale
            x2_s, y2_s = x1_s + processed_tile_tensor_0_1.shape[1], y1_s + processed_tile_tensor_0_1.shape[0]

            final_output_canvas[y1_s:y2_s, x1_s:x2_s, :] += processed_tile_tensor_0_1 * mask_expanded
            weight_sum_canvas[y1_s:y2_s, x1_s:x2_s, :] += mask_expanded
            
            del LQ_tile, processed_tile_output_tensor, single_frame_output_tensor, processed_tile_tensor_0_1, mask, mask_expanded; clean_vram()

        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        output_tensor_final = final_output_canvas / weight_sum_canvas
        
        output_image_pil = Image.fromarray(
            (output_tensor_final.numpy() * 255).clip(0, 255).astype(np.uint8)
        )

    else:
        raise gr.Error("Non-tiled DiT for images is not recommended. Please enable 'Tiled DiT'.")

    del pipe
    clean_vram()
    
    return output_image_pil


def unified_image_inference(input_path_textbox_val: str, input_upload_val: str | None, output_folder_path: str, process_mode: str, scale: int, tiled: bool, tile_size_h: int, tile_size_w: int, tile_stride_h: int, tile_stride_w: int, num_persistent_param_in_dit: int, tiled_dit: bool, dit_tile_size: int, dit_tile_overlap: int, num_inference_steps_val: int, num_frames_for_sequence_val: int):
    
    output_image_display = gr.Image(visible=False) # Default to invisible
    batch_status_text = ""

    if not output_folder_path:
        output_folder_path = "outputs"
    os.makedirs(output_folder_path, exist_ok=True)

    if process_mode == "Single File":
        input_file_path = input_upload_val
        if not input_file_path:
            raise gr.Error("No image file uploaded for Single File mode.")
        if not os.path.isfile(input_file_path):
            raise gr.Error(f"Input Path '{input_file_path}' is not a valid file.")
        
        try:
            image_pil = Image.open(input_file_path).convert('RGB')
        except Exception as e:
            raise gr.Error(f"Could not open image file: {e}")

        upscaled_image_pil = _upscale_core_logic(
            image_pil, scale, tiled, tile_size_h, tile_size_w, tile_stride_h, tile_stride_w,
            num_persistent_param_in_dit, tiled_dit, dit_tile_size, dit_tile_overlap,
            num_inference_steps_val, num_frames_for_sequence_val
        )

        input_filename = os.path.basename(input_file_path)
        name, ext = os.path.splitext(input_filename)
        output_filename = f"{name}_{scale}x_upscaled.png"
        final_output_path = os.path.join(output_folder_path, output_filename)
        upscaled_image_pil.save(final_output_path)
        
        return gr.Image(value=final_output_path, visible=True), "Single image processed and saved."

    elif process_mode == "Batch Folder":
        input_folder_path = input_path_textbox_val
        if not input_folder_path:
            raise gr.Error("Input Folder Path cannot be empty for Batch Folder mode.")
        if not os.path.isdir(input_folder_path):
            raise gr.Error(f"Input Path '{input_folder_path}' is not a valid directory for Batch Folder mode.")
        
        image_files = list_images_natural(input_folder_path)
        if not image_files:
            raise gr.Error(f"No supported image files found in '{input_folder_path}'.")

        processed_count = 0
        for img_path in tqdm(image_files, desc="[FlashVSR] Batch Processing"):
            try:
                image_pil = Image.open(img_path).convert('RGB')
                upscaled_image_pil = _upscale_core_logic(
                    image_pil, scale, tiled, tile_size_h, tile_size_w, tile_stride_h, tile_stride_w,
                    num_persistent_param_in_dit, tiled_dit, dit_tile_size, dit_tile_overlap,
                    num_inference_steps_val, num_frames_for_sequence_val
                )

                input_filename = os.path.basename(img_path)
                name, ext = os.path.splitext(input_filename)
                output_filename = f"{name}_{scale}x_upscaled.png"
                final_output_path = os.path.join(output_folder_path, output_filename)
                upscaled_image_pil.save(final_output_path)
                processed_count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Continue with other images in batch
        
        if processed_count == 0:
            batch_status_text = f"No images were successfully processed in batch from {input_folder_path}."
        else:
            batch_status_text = f"Batch processing complete. {processed_count} images saved to {output_folder_path}."
        
        return gr.Image(visible=False), batch_status_text # For batch mode, hide image output and show status message

    return gr.Image(visible=False), "Invalid processing mode selected."


def run_inference(video_path, scale, tiled, tile_size_h, tile_size_w, tile_stride_h, tile_stride_w, num_persistent_param_in_dit, tiled_dit, dit_tile_size, dit_tile_overlap, num_inference_steps_val):
    
    pipe = init_pipeline(num_persistent_param_in_dit)
    
    dtype, device = torch.bfloat16, 'cuda'
    
    if tiled_dit:
        frames, original_fps = prepare_tensors(video_path, dtype=dtype)
        N, H, W, C = frames.shape
        tile_coords = calculate_tile_coords(H, W, dit_tile_size, dit_tile_overlap)
        
        num_aligned_frames = largest_8n1_leq(N + 4) - 4
        final_output_canvas, weight_sum_canvas = torch.zeros((num_aligned_frames, H*scale, W*scale, C), dtype=torch.float32), torch.zeros((num_aligned_frames, H*scale, W*scale, C), dtype=torch.float32)
        
        for i in tqdm(range(len(tile_coords)), desc="[FlashVSR] Processing tiles"):
            x1, y1, x2, y2 = tile_coords[i]
            input_tile = frames[:, y1:y2, x1:x2, :]
            
            LQ_tile, th, tw, F = prepare_input_tensor_for_tile(input_tile, device, scale=scale, dtype=dtype)
            LQ_tile = LQ_tile.to(device)
            
            output_tile_gpu = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=num_inference_steps_val, seed=0,
                tiled=tiled,
                tile_size=(tile_size_h, tile_size_w),
                tile_stride=(tile_stride_h, tile_stride_w),
                LQ_video=LQ_tile, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=2.0*768*1280/(th*tw), 
                kv_ratio=3.0,
                local_range=11,
                color_fix = True,
            )
            
            processed_tile_cpu_tensor = tensor2video(output_tile_gpu, return_tensor=True).cpu() # T H W C, [0, 1]
            
            mask = create_feather_mask((processed_tile_cpu_tensor.shape[1], processed_tile_cpu_tensor.shape[2]), dit_tile_overlap * scale).cpu().permute(0, 2, 3, 1) # 1 H W C
            mask_expanded = mask.repeat(processed_tile_cpu_tensor.shape[0], 1, 1, 1) # T H W C
            
            x1_s, y1_s = x1 * scale, y1 * scale
            x2_s, y2_s = x1_s + processed_tile_cpu_tensor.shape[2], y1_s + processed_tile_cpu_tensor.shape[1]
            
            final_output_canvas[:, y1_s:y2_s, x1_s:x2_s, :] += processed_tile_cpu_tensor * mask_expanded
            weight_sum_canvas[:, y1_s:y2_s, x1_s:x2_s, :] += mask_expanded
            del LQ_tile, output_tile_gpu, input_tile; clean_vram()
            
        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        video_output_tensor = final_output_canvas / weight_sum_canvas # T H W C, [0, 1]
        
        video = (video_output_tensor * 255).clip(0, 255).numpy().astype(np.uint8)
        video = [Image.fromarray(frame) for frame in video]
        fps = original_fps
    else:
        LQ, th, tw, F, fps = prepare_input_tensor(video_path, scale=scale, dtype=dtype, device=device)

        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=num_inference_steps_val, seed=0, 
            tiled=tiled,
            tile_size=(tile_size_h, tile_size_w),
            tile_stride=(tile_stride_h, tile_stride_w),
            LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
            topk_ratio=2.0*768*1280/(th*tw), 
            kv_ratio=3.0,
            local_range=11,
            color_fix = True,
        )
        video = tensor2video(video)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_{scale}x.mp4")
    save_video(video, output_path, fps=fps, quality=6)
    
    del pipe
    clean_vram()
    
    return output_path

with gr.Blocks() as demo:
    gr.Markdown("Note: If you are running into memory issues, try reducing the tile size and stride. You can also try setting the environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before running the script.")
    
    with gr.Tabs():
        with gr.TabItem("Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Input Video")
                    video_scale_slider = gr.Slider(minimum=1, maximum=4, step=1, label="Scale Factor", value=4)
                    video_num_inference_steps_slider = gr.Slider(minimum=1, maximum=100, step=1, label="Inference Steps", value=20)
                    video_tiled_checkbox = gr.Checkbox(label="Tiled VAE", value=True)
                    video_tiled_dit_checkbox = gr.Checkbox(label="Tiled DiT", value=False)
                    with gr.Row():
                        video_tile_size_h_slider = gr.Slider(minimum=32, maximum=256, step=16, label="VAE Tile Size Height", value=32)
                        video_tile_size_w_slider = gr.Slider(minimum=32, maximum=256, step=16, label="VAE Tile Size Width", value=64)
                    with gr.Row():
                        video_tile_stride_h_slider = gr.Slider(minimum=16, maximum=128, step=16, label="VAE Tile Stride Height", value=16)
                        video_tile_stride_w_slider = gr.Slider(minimum=16, maximum=128, step=16, label="VAE Tile Stride Width", value=32)
                    with gr.Row():
                        video_dit_tile_size_slider = gr.Slider(minimum=64, maximum=512, step=16, label="DiT Tile Size", value=256)
                        video_dit_tile_overlap_slider = gr.Slider(minimum=8, maximum=128, step=8, label="DiT Tile Overlap", value=24)
                    video_run_button = gr.Button("Run Video Inference")
                with gr.Column():
                    video_output = gr.Video(label="Output Video")

        with gr.TabItem("Image"):
            with gr.Row():
                with gr.Column():
                    process_mode_radio = gr.Radio(
                        ["Single File", "Batch Folder"], label="Processing Mode", value="Single File"
                    )
                    image_input_path_textbox = gr.Textbox(label="Input Path (Folder for Batch)", placeholder="Enter folder path for batch processing", visible=False)
                    image_input_upload_component = gr.Image(type="filepath", label="Upload Image (Single File)", visible=True)
                    
                    image_output_folder_path_textbox = gr.Textbox(label="Output Folder Path", placeholder="Enter folder path to save outputs", value="outputs")
                    
                    image_scale_slider = gr.Slider(minimum=1, maximum=4, step=1, label="Scale Factor", value=4)
                    image_num_inference_steps_slider = gr.Slider(minimum=1, maximum=100, step=1, label="Inference Steps", value=20)
                    image_num_frames_for_sequence_slider = gr.Slider(minimum=1, maximum=25, step=1, label="Frames per Tile (Image)", value=25)
                    image_tiled_checkbox = gr.Checkbox(label="Tiled VAE", value=True)
                    image_tiled_dit_checkbox = gr.Checkbox(label="Tiled DiT", value=True)
                    with gr.Row():
                        image_tile_size_h_slider = gr.Slider(minimum=32, maximum=256, step=16, label="VAE Tile Size Height", value=32)
                        image_tile_size_w_slider = gr.Slider(minimum=32, maximum=256, step=16, label="VAE Tile Size Width", value=64)
                    with gr.Row():
                        image_tile_stride_h_slider = gr.Slider(minimum=16, maximum=128, step=16, label="VAE Tile Stride Height", value=16)
                        image_tile_stride_w_slider = gr.Slider(minimum=16, maximum=128, step=16, label="VAE Tile Stride Width", value=32)
                    with gr.Row():
                        image_dit_tile_size_slider = gr.Slider(minimum=64, maximum=512, step=16, label="DiT Tile Size", value=256)
                        image_dit_tile_overlap_slider = gr.Slider(minimum=8, maximum=128, step=8, label="DiT Tile Overlap", value=24)
                    image_run_inference_button = gr.Button("Run Inference")
                with gr.Column():
                    image_output = gr.Image(type="pil", label="Output Image (Single File Mode)", visible=True)
                    image_batch_status_text = gr.Textbox(label="Batch Status", interactive=False, visible=False)


    num_persistent_param_in_dit_slider = gr.Slider(minimum=0, maximum=10000000, step=100000, label="Persistent Parameters in DiT", value=0, interactive=True)

    # Dynamic visibility for image input components
    process_mode_radio.change(
        lambda mode: gr.Image(visible=mode == "Single File"),
        inputs=[process_mode_radio],
        outputs=[image_input_upload_component]
    ).then(
        lambda mode: gr.Textbox(visible=mode == "Batch Folder"),
        inputs=[process_mode_radio],
        outputs=[image_input_path_textbox]
    ).then(
        lambda mode: gr.Image(visible=mode == "Single File"),
        inputs=[process_mode_radio],
        outputs=[image_output]
    ).then(
        lambda mode: gr.Textbox(visible=mode == "Batch Folder"),
        inputs=[process_mode_radio],
        outputs=[image_batch_status_text]
    )

    video_run_button.click(
        fn=run_inference,
        inputs=[video_input, video_scale_slider, video_tiled_checkbox, video_tile_size_h_slider, video_tile_size_w_slider, video_tile_stride_h_slider, video_tile_stride_w_slider, num_persistent_param_in_dit_slider, video_tiled_dit_checkbox, video_dit_tile_size_slider, video_dit_tile_overlap_slider, video_num_inference_steps_slider],
        outputs=video_output,
    )
    
    image_run_inference_button.click(
        fn=unified_image_inference,
        inputs=[
            image_input_path_textbox,
            image_input_upload_component, # Pass the upload component value
            image_output_folder_path_textbox,
            process_mode_radio,
            image_scale_slider,
            image_tiled_checkbox,
            image_tile_size_h_slider,
            image_tile_size_w_slider,
            image_tile_stride_h_slider,
            image_tile_stride_w_slider,
            num_persistent_param_in_dit_slider,
            image_tiled_dit_checkbox,
            image_dit_tile_size_slider,
            image_dit_tile_overlap_slider,
            image_num_inference_steps_slider,
            image_num_frames_for_sequence_slider
        ],
        outputs=[image_output, image_batch_status_text],
    )

demo.launch()