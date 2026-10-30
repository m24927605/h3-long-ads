"""Plan B: one MiniMax-H3 Ref2VA windowed pass over a whole ad via the ComfyUI HTTP API.

Usage: python3 run_vm.py PREFIX PROMPT_FILE CONTROL(0|1) [STRENGTH]
Inputs on the VM: /content/source_ad.mp4 (control source, 464 frames 1280x704),
/content/person.png (<Picture 1>), /content/product.png (<Picture 2>).
Writes /content/<prefix>_workflow_api.json, /content/<prefix>_events.jsonl, /content/<prefix>.mp4.
"""
import asyncio, json, shutil, subprocess, sys, time
from pathlib import Path
import aiohttp, requests

PREFIX, PROMPT_FILE, CONTROL = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
STRENGTH = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
COMFY = Path("/content/ComfyUI")
BASE = "http://127.0.0.1:8188"
SEED = 768914106098754
FRAMES, W, H = 379, 1280, 704
DIFFUSION = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
PATCH = "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"
assert (FRAMES - 5) % 17 == 0

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

for _ in range(300):
    try:
        requests.get(BASE + "/system_stats", timeout=2).raise_for_status(); break
    except requests.RequestException:
        time.sleep(2)
log("comfy up")

for f in ("person.png", "product.png", "source_ad.mp4"):
    shutil.copy(f"/content/{f}", COMFY / "input" / f)
prompt = Path(PROMPT_FILE).read_text(encoding="utf-8").strip()
graph = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": DIFFUSION, "weight_dtype": "default"}},
    "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": LORA, "strength_model": 1.0}},
    "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
    "4": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
    "5": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
    "6": {"class_type": "LoadImage", "inputs": {"image": "person.png"}},
    "7": {"class_type": "LoadImage", "inputs": {"image": "product.png"}},
    "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0],
          "prompt": prompt, "width": W, "height": H, "length": FRAMES, "ref_image_size": "match",
          "ref_images.ref_image_0": ["6", 0], "ref_images.ref_image_1": ["7", 0]}},
}
model_out = ["2", 0]
if CONTROL:
    graph.update({
        "20": {"class_type": "VHS_LoadVideoPath", "inputs": {"video": "/content/source_ad.mp4", "force_rate": 24,
               "custom_width": W, "custom_height": H, "frame_load_cap": FRAMES, "skip_first_frames": 0, "select_every_nth": 1}},
        "21": {"class_type": "Canny", "inputs": {"image": ["20", 0], "low_threshold": 0.2, "high_threshold": 0.5}},
        "22": {"class_type": "ModelPatchLoader", "inputs": {"name": PATCH}},
        "23": {"class_type": "MiniMaxH3FunControlNetApply", "inputs": {"model": model_out, "model_patch": ["22", 0], "vae": ["4", 0],
               "strength": STRENGTH, "start_percent": 0.0, "end_percent": 1.0, "control_video": ["21", 0]}},
        "24": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["21", 0], "frame_rate": 24, "loop_count": 0,
               "filename_prefix": PREFIX + "_canny", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 23,
               "save_metadata": False, "trim_to_audio": False, "pingpong": False, "save_output": True}},
    })
    model_out = ["23", 0]
graph.update({
    "17": {"class_type": "MiniMaxH3ContextWindows", "inputs": {"model": model_out, "context_length": 124,
           "context_overlap": 39, "fuse_method": "pyramid", "freenoise": True, "split_conds_to_windows": False}},
    "9": {"class_type": "BasicGuider", "inputs": {"model": ["17", 0], "conditioning": ["8", 0]}},
    "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
    "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "12": {"class_type": "BasicScheduler", "inputs": {"model": ["17", 0], "scheduler": "beta", "steps": 4, "denoise": 1.0}},
    "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["9", 0], "sampler": ["11", 0],
           "sigmas": ["12", 0], "latent_image": ["8", 1]}},
    "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["4", 0]}},
    "15": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["13", 0], "vae": ["5", 0]}},
    "16": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["14", 0], "audio": ["15", 0], "frame_rate": 24,
           "loop_count": 0, "filename_prefix": PREFIX, "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 18,
           "save_metadata": True, "trim_to_audio": False, "pingpong": False, "save_output": True}},
})
Path(f"/content/{PREFIX}_workflow_api.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False))

async def main():
    client_id = "h3-" + PREFIX
    events = open(f"/content/{PREFIX}_events.jsonl", "a")
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(f"ws://127.0.0.1:8188/ws?clientId={client_id}", max_msg_size=0) as ws:
            r = await s.post(BASE + "/prompt", json={"prompt": graph, "client_id": client_id})
            job = await r.json()
            log("submitted", json.dumps(job)[:3000])
            pid = job["prompt_id"]
            t0 = time.time()
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                m = json.loads(msg.data)
                if m["type"] in ("executing", "execution_start", "execution_success", "execution_error", "execution_cached"):
                    m["t"] = round(time.time() - t0, 2)
                    events.write(json.dumps(m) + "\n"); events.flush()
                    log(m["type"], json.dumps(m.get("data", {}))[:3000])
                    if m["type"] in ("execution_success", "execution_error"):
                        break
                    if m["type"] == "executing" and m["data"].get("node") is None and m["data"].get("prompt_id") == pid:
                        break
                elif m["type"] == "progress":
                    d = m["data"]
                    log("progress", d.get("node"), d.get("value"), "/", d.get("max"))
    hist = requests.get(BASE + f"/history/{pid}").json()[pid]
    log("status", json.dumps(hist.get("status"))[:3000])
    outs = [g for o in hist["outputs"].values() for g in o.get("gifs", []) if g["filename"].endswith(".mp4") and "_canny" not in g["filename"]]
    out = outs[0]
    src = COMFY / "output" / out.get("subfolder", "") / out["filename"]
    shutil.copy(src, f"/content/{PREFIX}.mp4")
    log("OUTPUT_DONE", f"/content/{PREFIX}.mp4", src.stat().st_size, "wall", round(time.time() - t0, 1))

asyncio.run(main())
