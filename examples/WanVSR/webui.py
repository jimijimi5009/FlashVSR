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

from diffsynth import ModelManager, FlashVSRFullPipeline
from utils.utils import Causal_LQ4x_Proj

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
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

import math
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

def stitch_video_tiles(
    tile_paths, 
    tile_coords, 
    final_dims, 
    scale, 
    overlap, 
    output_path, 
    fps, 
    quality, 
    cleanup=True,
    chunk_size=40
):
    if not tile_paths:
        return
    
    final_W, final_H = final_dims
    
    readers = [imageio.get_reader(p) for p in tile_paths]
    
    try:
        num_frames = readers[0].count_frames()
        if num_frames is None or num_frames <= 0:
            num_frames = len([_ for _ in readers[0]])
            for r in readers: r.close()
            readers = [imageio.get_reader(p) for p in tile_paths]
            
        with imageio.get_writer(output_path, fps=fps, quality=quality) as writer:
            for start_frame in tqdm(range(0, num_frames, chunk_size), desc="[FlashVSR] Stitching Chunks"):
                end_frame = min(start_frame + chunk_size, num_frames)
                current_chunk_size = end_frame - start_frame
                chunk_canvas = np.zeros((current_chunk_size, final_H, final_W, 3), dtype=np.float32)
                weight_canvas = np.zeros_like(chunk_canvas, dtype=np.float32)
                
                for i, reader in enumerate(readers):
                    try:
                        tile_chunk_frames = [
                            frame.astype(np.float32) / 255.0 
                            for idx, frame in enumerate(reader.iter_data()) 
                            if start_frame <= idx < end_frame
                        ]
                        tile_chunk_np = np.stack(tile_chunk_frames, axis=0)
                    except Exception as e:
                        continue
                    
                    if tile_chunk_np.shape[0] != current_chunk_size:
                        continue
                    
                    tile_H, tile_W, _ = tile_chunk_np.shape[1:]
                    ramp = np.linspace(0, 1, overlap * scale, dtype=np.float32)
                    mask = np.ones((tile_H, tile_W, 1), dtype=np.float32)
                    mask[:, :overlap*scale, :] *= ramp[np.newaxis, :, np.newaxis]
                    mask[:, -overlap*scale:, :] *= np.flip(ramp)[np.newaxis, :, np.newaxis]
                    mask[:overlap*scale, :, :] *= ramp[:, np.newaxis, np.newaxis]
                    mask[-overlap*scale:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]
                    mask_4d = mask[np.newaxis, :, :, :]
                    
                    x1_orig, y1_orig, _, _ = tile_coords[i]
                    out_y1, out_x1 = y1_orig * scale, x1_orig * scale
                    out_y2, out_x2 = out_y1 + tile_H, out_x1 + tile_W
                    
                    chunk_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += tile_chunk_np * mask_4d
                    weight_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_4d
                    
                weight_canvas[weight_canvas == 0] = 1.0
                stitched_chunk = chunk_canvas / weight_canvas
                
                for frame_idx_in_chunk in range(current_chunk_size):
                    frame_uint8 = (np.clip(stitched_chunk[frame_idx_in_chunk], 0, 1) * 255).astype(np.uint8)
                    writer.append_data(frame_uint8)
                    
    finally:
        for reader in readers:
            reader.close()
            
    if cleanup:
        for path in tile_paths:
            try:
                os.remove(path)
            except OSError as e:
                pass

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

def run_inference(video_path, scale, tiled, tile_size_h, tile_size_w, tile_stride_h, tile_stride_w, num_persistent_param_in_dit, tiled_dit, dit_tile_size, dit_tile_overlap):
    
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
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
                tiled=tiled,
                tile_size=(tile_size_h, tile_size_w),
                tile_stride=(tile_stride_h, tile_stride_w),
                LQ_video=LQ_tile, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=2.0*768*1280/(th*tw), 
                kv_ratio=3.0,
                local_range=11,
                color_fix = True,
            )
            
            # Get tensor output in [0, 1] range
            processed_tile_cpu_tensor = tensor2video(output_tile_gpu, return_tensor=True).cpu() # T H W C, [0, 1]
            
            mask = create_feather_mask((processed_tile_cpu_tensor.shape[1], processed_tile_cpu_tensor.shape[2]), dit_tile_overlap * scale).cpu().permute(0, 2, 3, 1) # 1 H W C
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
    else:
        # Non-tiled DiT logic
        LQ, th, tw, F, fps = prepare_input_tensor(video_path, scale=scale, dtype=dtype, device=device)

        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0, 
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
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input Video")
            scale_slider = gr.Slider(minimum=1, maximum=4, step=1, label="Scale Factor", value=4)
            tiled_checkbox = gr.Checkbox(label="Tiled VAE", value=True)
            tiled_dit_checkbox = gr.Checkbox(label="Tiled DiT", value=False)
            with gr.Row():
                tile_size_h_slider = gr.Slider(minimum=32, maximum=256, step=16, label="VAE Tile Size Height", value=32)
                tile_size_w_slider = gr.Slider(minimum=32, maximum=256, step=16, label="VAE Tile Size Width", value=64)
            with gr.Row():
                tile_stride_h_slider = gr.Slider(minimum=16, maximum=128, step=16, label="VAE Tile Stride Height", value=16)
                tile_stride_w_slider = gr.Slider(minimum=16, maximum=128, step=16, label="VAE Tile Stride Width", value=32)
            with gr.Row():
                dit_tile_size_slider = gr.Slider(minimum=64, maximum=512, step=16, label="DiT Tile Size", value=256)
                dit_tile_overlap_slider = gr.Slider(minimum=8, maximum=128, step=8, label="DiT Tile Overlap", value=24)
            run_button = gr.Button("Run Inference")
        with gr.Column():
            video_output = gr.Video(label="Output Video")

    num_persistent_param_in_dit_slider = gr.Slider(minimum=0, maximum=10000000, step=100000, label="Persistent Parameters in DiT", value=0)

    run_button.click(
        fn=run_inference,
        inputs=[video_input, scale_slider, tiled_checkbox, tile_size_h_slider, tile_size_w_slider, tile_stride_h_slider, tile_stride_w_slider, num_persistent_param_in_dit_slider, tiled_dit_checkbox, dit_tile_size_slider, dit_tile_overlap_slider],
        outputs=video_output,
    )

demo.launch()
