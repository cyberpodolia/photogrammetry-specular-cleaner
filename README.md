# DHAN-SHR Inference (Pretrained Model)

This repository contains a pretrained specular highlight removal model checkpoint and an inference script.

The goal is not "deblurring" in the classic motion-blur sense. The model is used to suppress specular highlights
(glare/shine) in photos, which can help photogrammetry reconstruction and texture baking on shiny surfaces.

## Structure

- `weights/model.pth`: pretrained checkpoint
- `src/test_model.py`: inference runner for single image or folder
- `requirements.txt`: Python dependencies

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### CUDA Environment (NVIDIA)

If you want GPU inference, install a CUDA-enabled PyTorch build in the same venv:

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Run Inference

### 1. Folder of images

```powershell
python -m src.test_model --input "C:\path\to\images"
```

### 2. Single image

```powershell
python -m src.test_model --input "C:\path\to\image.png"
```

Outputs are saved to `outputs/` as `*_shr.png` by default.
If multiple source files share the same stem, the script appends a short hash to avoid overwriting.

## Photogrammetry notes (why this helps)

Specular highlights are view-dependent and violate the "consistent appearance" assumption used by many
feature matchers and multi-view reconstruction pipelines. Reducing highlights can:

- improve keypoint matching consistency
- reduce reconstruction noise on glossy areas
- produce textures that look closer to diffuse albedo

This is not a replacement for good capture, but it can be a helpful preprocessing step.

## Cross-polarization workflow (recommended capture)

If you can control your capture setup, cross-polarization is the most reliable way to remove glare.
It separates diffuse vs specular components optically instead of guessing them from a single image.

Minimal setup:

- Linear polarizing film/filter on the light source (or softbox/ring light)
- Polarizing filter on the camera lens

Capture two images per viewpoint:

1) **Cross** (filters ~90 degrees): specular is strongly suppressed (diffuse-dominant)
2) **Parallel** (filters aligned): specular passes (diffuse + specular)

Practical pipeline:

- Align/register the pair (small shifts happen even on tripods)
- Match exposure/white balance between the pair
- Use the **Cross** image directly for photogrammetry when possible
- (Optional) derive a specular map:
  - `spec ~= max(0, parallel - k * cross)`

When cross-polarization is not possible (outdoor scenes, uncontrolled lighting), this model can be used as a
best-effort single-image approximation.

## Useful Options

- `--checkpoint weights/model.pth`
- `--output outputs`
- `--device cuda` (or `cpu`)
- `--resize 512 512`
- `--tile-size 512`
- `--tile-overlap 64`
- `--in-range zero_one`
- `--out-range zero_one`

Example:

```powershell
python -m src.test_model --input "C:\path\to\images" --device cuda --resize 512 512
```

Memory-safe (tiled) example:

```powershell
python -m src.test_model `
  --input "C:\work_sahsa\2\IMG_0877.DNG" `
  --output "C:\work_sahsa\2\out" `
  --checkpoint "weights\model.pth" `
  --model-module "src.dhan_shr" `
  --model-class "Model" `
  --state-dict-key "state_dict" `
  --device cpu `
  --tile-size 512 `
  --tile-overlap 64
```

Final command (CUDA + tiled):

```powershell
python -m src.test_model `
  --input "C:\work_sahsa\2\IMG_0877.DNG" `
  --output "C:\work_sahsa\2\out" `
  --checkpoint "weights\model.pth" `
  --model-module "src.dhan_shr" `
  --model-class "Model" `
  --state-dict-key "state_dict" `
  --device cuda `
  --tile-size 512 `
  --tile-overlap 64
```

## If Checkpoint Is a `state_dict`

If loading fails because checkpoint is not a full module/TorchScript, pass model class info:

```powershell
python -m src.test_model `
  --input "C:\path\to\images" `
  --checkpoint "weights/model.pth" `
  --model-module "src.your_model_file" `
  --model-class "YourModelClass"
```

Optional:

- `--state-dict-key model` (or `state_dict`, `net`, etc.)
- `--model-kwargs "key=value,key2=value2"`

## Notes

- The script supports common image types: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`.
- `.dng` is included in extension scanning, but decoding depends on available image backend support.
- If PowerShell blocks activation scripts, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Model provenance / credits

This project uses a pretrained checkpoint that was originally published on GitHub by its authors.

Upstream repo:
- https://github.com/cyberpodolia/-photogrammetry-specular-cleaner

Paper / method:
- Unknown / not linked yet.

License for weights:
- Unknown / check upstream repository.
