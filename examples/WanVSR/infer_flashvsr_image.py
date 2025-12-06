import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import math
import argparse

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


def largest_8n1_leq(n):
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def init_pipeline(num_persistent_param_in_dit):
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
    torch.cuda.empty_cache()
    return vid_final, tH, tW, num_frames_padded

def run_inference(image_path, output_path, scale, tiled_dit, dit_tile_size, dit_tile_overlap, vae_tiled, vae_tile_size_h, vae_tile_size_w, vae_tile_stride_h, vae_tile_stride_w, num_persistent_param_in_dit):
    pipe = init_pipeline(num_persistent_param_in_dit)
    
    dtype, device = torch.bfloat16, 'cuda'
    
    image = Image.open(image_path).convert('RGB')
    frame_tensor = torch.from_numpy(np.array(image)).to(dtype=dtype).unsqueeze(0)

    if tiled_dit:
        num_frames_for_sequence = 25
        frames = frame_tensor.repeat(num_frames_for_sequence, 1, 1, 1)

        N, H, W, C = frames.shape
        tile_coords = calculate_tile_coords(H, W, dit_tile_size, dit_tile_overlap)
        
        final_output_canvas = torch.zeros((H * scale, W * scale, C), dtype=torch.float32)
        weight_sum_canvas = torch.zeros_like(final_output_canvas, dtype=torch.float32)

        print(f"Processing in {len(tile_coords)} tiles...")
        for i, (x1, y1, x2, y2) in enumerate(tqdm(tile_coords, desc="[FlashVSR] Processing tiles")):
            input_tile = frames[:, y1:y2, x1:x2, :]
            
            LQ_tile, th, tw, F = prepare_input_tensor_for_tile(input_tile, device, scale=scale, dtype=dtype)
            LQ_tile = LQ_tile.to(device)

            with torch.no_grad():
                processed_tile_output_tensor = pipe(
                    prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
                    tiled=vae_tiled,
                    tile_size=(vae_tile_size_h, vae_tile_size_w),
                    tile_stride=(vae_tile_stride_h, vae_tile_stride_w),
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
            
            single_frame_output_tensor = processed_tile_output_tensor[:, 0, :, :]
            
            processed_tile_tensor_0_1 = ((single_frame_output_tensor + 1.0) / 2.0).permute(1, 2, 0).cpu().float()
            
            mask = create_feather_mask((processed_tile_tensor_0_1.shape[0], processed_tile_tensor_0_1.shape[1]), dit_tile_overlap * scale).cpu()
            
            mask_expanded = mask.squeeze(0).squeeze(0).unsqueeze(2).repeat(1, 1, 3)

            x1_s, y1_s = x1 * scale, y1 * scale
            x2_s, y2_s = x1_s + processed_tile_tensor_0_1.shape[1], y1_s + processed_tile_tensor_0_1.shape[0]

            final_output_canvas[y1_s:y2_s, x1_s:x2_s, :] += processed_tile_tensor_0_1 * mask_expanded
            weight_sum_canvas[y1_s:y2_s, x1_s:x2_s, :] += mask_expanded
            
            del LQ_tile, processed_tile_output_tensor, single_frame_output_tensor, processed_tile_tensor_0_1, mask, mask_expanded; torch.cuda.empty_cache()

        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        output_tensor_final = final_output_canvas / weight_sum_canvas
        
        output_image_pil = Image.fromarray((output_tensor_final.numpy() * 255).astype(np.uint8))

    else:
        raise NotImplementedError("Non-tiled image upscaling is not implemented. Please use --tiled_dit.")

    output_image_pil.save(output_path)
    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiled Image Upscaling with FlashVSR")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("output_path", type=str, help="Path to save the upscaled image.")
    parser.add_argument("--scale", type=int, default=4, help="Upscaling factor.")
    parser.add_argument("--tiled_dit", action="store_true", default=True, help="Enable DiT tiling.")
    parser.add_argument("--dit_tile_size", type=int, default=128, help="DiT Tile Size.")
    parser.add_argument("--dit_tile_overlap", type=int, default=24, help="DiT Tile Overlap.")
    parser.add_argument("--vae_tiled", action="store_true", default=True, help="Enable VAE tiling.")
    parser.add_argument("--vae_tile_size_h", type=int, default=32, help="VAE Tile Size Height.")
    parser.add_argument("--vae_tile_size_w", type=int, default=32, help="VAE Tile Size Width.")
    parser.add_argument("--vae_tile_stride_h", type=int, default=32, help="VAE Tile Stride Height.")
    parser.add_argument("--vae_tile_stride_w", type=int, default=32, help="VAE Tile Stride Width.")
    parser.add_argument("--num_persistent_param_in_dit", type=int, default=0, help="Persistent Parameters in DiT.")

    args = parser.parse_args()

    if not args.tiled_dit:
        print("Warning: Running without --tiled_dit may cause out-of-memory errors for large images. It is recommended to use --tiled_dit.")

    run_inference(
        args.image_path, args.output_path, args.scale, args.tiled_dit, args.dit_tile_size, 
        args.dit_tile_overlap, args.vae_tiled, args.vae_tile_size_h, args.vae_tile_size_w, 
        args.vae_tile_stride_h, args.vae_tile_stride_w, args.num_persistent_param_in_dit
    )
