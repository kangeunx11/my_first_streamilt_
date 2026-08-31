from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    architecture = checkpoint.get("architecture")
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint["class_names"]))
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, len(checkpoint["class_names"])
        )
    else:
        raise ValueError(f"지원하지 않는 모델 구조입니다: {architecture}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(checkpoint["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
        ]
    )
    return model, transform, checkpoint["class_names"]


def main() -> None:
    parser = argparse.ArgumentParser(description="음식 이미지 카테고리 예측")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/food_classifier.pt"))
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, class_names = load_model(args.model, device)
    results = []
    for path in args.images:
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = model(tensor).softmax(dim=1)[0]
        k = min(args.top_k, len(class_names))
        values, indices = probabilities.topk(k)
        ranked = [
            {"category": class_names[index], "probability": round(float(value), 6)}
            for value, index in zip(values.cpu(), indices.cpu())
        ]
        results.append({"image": str(path), "prediction": ranked[0]["category"], "scores": ranked})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

