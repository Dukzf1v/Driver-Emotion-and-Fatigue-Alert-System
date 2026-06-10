import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torchvision import models

def build_model(num_classes: int = 2, dropout: float = 0.3) -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )
    return model

def main():
    ckpt_path = "best_mobilenetv3_5class.pth"
    output_dir = "static/models"
    os.makedirs(output_dir, exist_ok=True)
    output_ptl_path = os.path.join(output_dir, "emotion_model.ptl")

    print(f"Loading checkpoint: {ckpt_path}")
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    class_names = ckpt.get(
        "class_names",
        ckpt.get("cfg", {}).get("class_names", ["Neutral", "Anger", "Fear", "Happiness", "Sadness"]))
    img_size = ckpt.get("img_size", ckpt.get("cfg", {}).get("img_size", 224))
    num_classes = len(class_names)
    print(f"Model properties: classes={num_classes} ({class_names}), input_size={img_size}")

    model = build_model(num_classes=num_classes)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dummy_input = torch.rand(1, 3, img_size, img_size)
    print("Tracing model using JIT...")
    traced = torch.jit.trace(model, dummy_input)

    # Directly save for lite interpreter without mobile optimizer
    print(f"Saving raw PTL model to {output_ptl_path}")
    traced._save_for_lite_interpreter(output_ptl_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
