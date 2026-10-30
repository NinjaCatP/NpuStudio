$ErrorActionPreference = 'Continue'
$app = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $app
$log = Join-Path $app 'setup.log'
function Log($m) { $m | Tee-Object -FilePath $log -Append }
Log "Setup started $(Get-Date)"
[Net.ServicePointManager]::SecurityProtocol = 'Tls12'
if (!(Test-Path "$app\uv\uv.exe")) {
  Log 'Downloading uv...'
  Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile "$app\uv.zip" -UseBasicParsing
  Expand-Archive "$app\uv.zip" "$app\uv" -Force
  Remove-Item "$app\uv.zip"
}
$uv = "$app\uv\uv.exe"
$env:UV_PYTHON_INSTALL_DIR = "$app\python"
$env:UV_CACHE_DIR = "$app\uvcache"
Log 'Installing Python 3.11...'
& $uv venv "$app\venv" --python 3.11 2>&1 | Tee-Object -FilePath $log -Append
Log 'Installing packages (this takes a few minutes)...'
& $uv pip install --python "$app\venv\Scripts\python.exe" openvino==2026.4.1 openvino-genai==2026.4.1.0 openvino-tokenizers==2026.4.1.0 huggingface_hub pycaw comtypes pillow numpy pywin32 tkinterdnd2 2>&1 | Tee-Object -FilePath $log -Append
# NVIDIA GPU support (InstructPix2Pix, SD inpaint, Image Gen, Sharp upscale on CUDA): torch CUDA 12.8 (~2.6 GB download,
# ~4.5 GB installed) + diffusers. Skipped when no NVIDIA GPU is present - the app then uses NPU / Intel Arc / CPU only.
if (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' }) {
  Log 'NVIDIA GPU found - installing torch (CUDA) + diffusers, ~2.6 GB...'
  & $uv pip install --python "$app\venv\Scripts\python.exe" torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | Tee-Object -FilePath $log -Append
  & $uv pip install --python "$app\venv\Scripts\python.exe" diffusers transformers accelerate safetensors 2>&1 | Tee-Object -FilePath $log -Append
  & $uv pip install --python "$app\venv\Scripts\python.exe" nvidia-ml-py 2>&1 | Tee-Object -FilePath $log -Append   # NVIDIA power/temp in Chip Monitor
  # Pose changes: OpenPose detector (controlnet_aux). Its deps would swap CUDA torch for a CPU build, so torch is re-pinned after.
  & $uv pip install --python "$app\venv\Scripts\python.exe" controlnet_aux 2>&1 | Tee-Object -FilePath $log -Append
  & $uv pip install --python "$app\venv\Scripts\python.exe" "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | Tee-Object -FilePath $log -Append
  # AI change redraw engine: ControlNet anime-lineart (fp16, ~0.7 GB) -> models\hf
  Log 'Downloading ControlNets (lineart + openpose, ~1.4 GB), DreamShaper 8 inpainting (~2 GB) and OpenPose detector (~0.5 GB) for AI change...'
  & "$app\venv\Scripts\python.exe" -c "from huggingface_hub import snapshot_download as d; d('lllyasviel/control_v11p_sd15s2_lineart_anime', cache_dir=r'$app\models\hf', allow_patterns=['config.json','diffusion_pytorch_model.fp16.safetensors']); d('lllyasviel/control_v11p_sd15_openpose', cache_dir=r'$app\models\hf', allow_patterns=['config.json','diffusion_pytorch_model.fp16.safetensors']); d('lllyasviel/Annotators', cache_dir=r'$app\models\hf', allow_patterns=['body_pose_model.pth','hand_pose_model.pth','facenet.pth']); d('Lykon/dreamshaper-8-inpainting', cache_dir=r'$app\models\hf', allow_patterns=['*.json','*.txt','unet/*.fp16.safetensors','vae/*.fp16.safetensors','text_encoder/*.fp16.safetensors'])" 2>&1 | Tee-Object -FilePath $log -Append
}
New-Item -ItemType Directory -Force "$app\models" | Out-Null
$base = 'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/single-image-super-resolution-1032/FP16/single-image-super-resolution-1032'
foreach ($ext in 'xml','bin') {
  $f = "$app\models\single-image-super-resolution-1032.$ext"
  if (!(Test-Path $f)) { Log "Downloading model .$ext..."; Invoke-WebRequest "$base.$ext" -OutFile $f -UseBasicParsing }
}
# Sharp model (Real-ESRGAN x4plus, converted to OpenVINO IR FP16 1x3x256x256) is shipped in models\ (made by convert_esrgan.py from
# https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth). Unzip esrmodels.zip if present.
if (!(Test-Path "$app\models\realesrgan-x4plus-256.xml") -and (Test-Path "$app\esrmodels.zip")) { Log 'Unpacking Sharp model...'; Expand-Archive "$app\esrmodels.zip" "$app\models" -Force }
if (!(Test-Path "$app\models\realesrgan-x4plus-256.xml")) { Log 'NOTE: Sharp model missing - only Fast will work.' }
# Image Gen model (OpenVINO/LCM_Dreamshaper_v7-int8-ov, ~1 GB) downloads on first use of the Image Gen tab.
# Edit tab: IS-Net (background removal) and LaMa (erase) ONNX models download on first use.
# Prompt inpaint model (stable-diffusion-inpainting, OpenVINO int8, ~1 GB) ships in sdinpaint.zip.
if (!(Test-Path "$app\models\sd-inpaint-int8\unet") -and (Test-Path "$app\sdinpaint.zip")) { Log 'Unpacking inpaint model...'; Expand-Archive "$app\sdinpaint.zip" "$app\models\sd-inpaint-int8" -Force }
# AI change window: the vision model (Phi-3.5-vision int4, ~2.3 GB) downloads on first use; CLIPSeg (auto-mask, ~270 MB)
# ships in models\clipseg. On NVIDIA, InstructPix2Pix / SD-inpainting (fp16) download to models\hf on first use (~2 GB each).
# Body-part / clothing parsers for AI change (SegFormer human + face parsing, converted to OpenVINO; needs torch):
if (!(Test-Path "$app\models\human-parse\model.xml")) { Log 'Converting body-part parsers (~110 MB download)...'; & "$app\venv\Scripts\python.exe" "$app\bpconvert.py" 2>&1 | Tee-Object -FilePath $log -Append }
# Device benchmark (fills models\devbench.json so Device=Auto picks the fastest):
if (!(Test-Path "$app\models\devbench.json")) { Log 'Benchmarking devices...'; & "$app\venv\Scripts\python.exe" "$app\devbench.py" 2>&1 | Tee-Object -FilePath $log -Append }
if (!(Test-Path "$app\venv\Scripts\pythonw.exe")) { Log 'SETUP FAILED: python missing'; exit 1 }
'ok' | Set-Content "$app\setup.done"
Log "Setup finished $(Get-Date)"
