from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required. Install it first, e.g. `pip install torch torchvision pillow`."
    ) from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".dng"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run specular highlight removal model on one image or a folder of images."
    )
    p.add_argument("--checkpoint", default="weights/model.pth", type=Path)
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        nargs="+",
        help="One or more input image files and/or folders",
    )
    p.add_argument("--output", default=Path("outputs"), type=Path, help="Output folder")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Tile size for memory-safe inference (0 disables tiling)",
    )
    p.add_argument(
        "--tile-overlap",
        type=int,
        default=32,
        help="Tile overlap in pixels (used when --tile-size > 0)",
    )

    p.add_argument(
        "--model-module",
        type=str,
        default=None,
        help="Python module path that defines your model class (needed for state_dict checkpoints)",
    )
    p.add_argument(
        "--model-class",
        type=str,
        default=None,
        help="Model class name inside --model-module",
    )
    p.add_argument(
        "--model-kwargs",
        type=str,
        default="",
        help="Optional constructor kwargs as k=v,k=v (ints/floats/bools/strings)",
    )
    p.add_argument(
        "--state-dict-key",
        type=str,
        default=None,
        help="Force a key for state_dict in checkpoint dict (example: model, state_dict, net)",
    )

    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("W", "H"))
    p.add_argument(
        "--in-range",
        choices=["zero_one", "minus_one_one", "zero_255"],
        default="zero_one",
        help="Input tensor range expected by model",
    )
    p.add_argument(
        "--out-range",
        choices=["zero_one", "minus_one_one", "zero_255"],
        default="zero_one",
        help="Output tensor range produced by model",
    )
    return p.parse_args()


def parse_kv(text: str) -> dict:
    if not text:
        return {}
    out = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid key=value pair: {item}")
        k, v = item.split("=", 1)
        v = v.strip()
        if v.lower() in {"true", "false"}:
            out[k.strip()] = v.lower() == "true"
            continue
        try:
            if "." in v:
                out[k.strip()] = float(v)
            else:
                out[k.strip()] = int(v)
            continue
        except ValueError:
            out[k.strip()] = v
    return out


def collect_images(paths: Sequence[Path]) -> list[Path]:
    collected: list[Path] = []
    for path in paths:
        if path.is_file():
            collected.append(path)
            continue
        if path.is_dir():
            collected.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS))
            continue
        raise FileNotFoundError(f"Input path not found: {path}")
    return list(dict.fromkeys(collected))


def to_tensor(img: Image.Image, in_range: str) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    if in_range == "zero_one":
        arr = arr / 255.0
    elif in_range == "minus_one_one":
        arr = (arr / 127.5) - 1.0
    elif in_range == "zero_255":
        pass

    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def to_image(t: torch.Tensor, out_range: str) -> Image.Image:
    t = t.detach().cpu().float()
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"Expected output shape [C,H,W] or [B,C,H,W], got {tuple(t.shape)}")

    if out_range == "zero_one":
        t = t.clamp(0, 1) * 255.0
    elif out_range == "minus_one_one":
        t = ((t.clamp(-1, 1) + 1) * 0.5) * 255.0
    elif out_range == "zero_255":
        t = t.clamp(0, 255)

    arr = t.permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(arr)


def unwrap_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for v in output:
            if isinstance(v, torch.Tensor):
                return v
    if isinstance(output, dict):
        for k in ["pred", "output", "out", "restored", "image", "result"]:
            v = output.get(k)
            if isinstance(v, torch.Tensor):
                return v
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v
    raise TypeError("Could not find tensor in model output")


def build_model(module_path: str, cls_name: str, kwargs_text: str) -> nn.Module:
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    kwargs = parse_kv(kwargs_text)
    return cls(**kwargs)


def find_state_dict(payload: object, forced_key: str | None) -> dict:
    if isinstance(payload, dict):
        if forced_key:
            sd = payload.get(forced_key)
            if isinstance(sd, dict):
                return sd
            raise KeyError(f"Key '{forced_key}' not found or not a state_dict")

        for k in ["state_dict", "model", "model_state_dict", "net", "params"]:
            sd = payload.get(k)
            if isinstance(sd, dict):
                return sd

        if payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload

    raise ValueError("Checkpoint does not look like a state_dict container")


def strip_dataparallel_prefix(sd: dict) -> dict:
    if not sd:
        return sd
    keys = list(sd.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def load_model(args: argparse.Namespace) -> nn.Module:
    ckpt = args.checkpoint

    try:
        model = torch.jit.load(str(ckpt), map_location=args.device)
        model.eval()
        return model
    except Exception:
        pass

    payload = torch.load(str(ckpt), map_location="cpu")

    if isinstance(payload, nn.Module):
        model = payload.to(args.device)
        model.eval()
        return model

    if not args.model_module or not args.model_class:
        raise SystemExit(
            "Checkpoint appears to be a state_dict. Please pass --model-module and --model-class."
        )

    model = build_model(args.model_module, args.model_class, args.model_kwargs)
    sd = find_state_dict(payload, args.state_dict_key)
    sd = strip_dataparallel_prefix(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys: {len(unexpected)}")

    model = model.to(args.device)
    model.eval()
    return model


def run_model(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    y = model(x)
    y = unwrap_output(y)
    if y.shape[-2:] != x.shape[-2:]:
        y = torch.nn.functional.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return y


def run_model_tiled(model: nn.Module, x: torch.Tensor, tile_size: int, tile_overlap: int) -> torch.Tensor:
    _, c, h, w = x.shape
    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        return run_model(model, x)

    stride = tile_size - tile_overlap
    if stride <= 0:
        raise ValueError("--tile-overlap must be smaller than --tile-size")

    output = torch.zeros((1, c, h, w), device=x.device, dtype=x.dtype)
    weight = torch.zeros((1, 1, h, w), device=x.device, dtype=x.dtype)

    y_starts = list(range(0, h, stride))
    x_starts = list(range(0, w, stride))
    if y_starts[-1] != max(0, h - tile_size):
        y_starts.append(max(0, h - tile_size))
    if x_starts[-1] != max(0, w - tile_size):
        x_starts.append(max(0, w - tile_size))

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tile = x[:, :, y0:y1, x0:x1]
            pred = run_model(model, tile)
            output[:, :, y0:y1, x0:x1] += pred
            weight[:, :, y0:y1, x0:x1] += 1.0

    return output / weight.clamp_min(1e-6)


def run_inference(model: nn.Module, image_paths: Sequence[Path], args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(image_paths, 1):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"[warn] Skipping unreadable image: {path} ({exc})")
            continue
        if args.resize:
            w, h = args.resize
            img = img.resize((w, h), Image.BICUBIC)

        x = to_tensor(img, args.in_range).to(args.device)

        with torch.no_grad():
            y = run_model_tiled(model, x, args.tile_size, args.tile_overlap)

        out_img = to_image(y, args.out_range)
        out_path = args.output / f"{path.stem}_shr.png"
        out_img.save(out_path)
        print(f"[{i}/{len(image_paths)}] {path.name} -> {out_path}")


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but current PyTorch build has no CUDA. Falling back to CPU.")
        args.device = "cpu"
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    image_paths = collect_images(args.input)
    if not image_paths:
        raise SystemExit("No images found in --input")

    model = load_model(args)
    run_inference(model, image_paths, args)


if __name__ == "__main__":
    main()
