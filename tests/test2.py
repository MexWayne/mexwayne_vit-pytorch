import torch
from vit_pytorch import ViT
import cv2
import numpy as np
import random
from transformers import ViTForImageClassification, ViTImageProcessor


def test_new():

    import torch
    from transformers import ViTForImageClassification, ViTImageProcessor
    from PIL import Image

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    img = Image.open("../datasets/000000000110.jpg").convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits           # shape: [1, 1000]
        pred_class = logits.argmax(dim=-1)  # tensor([idx])
    return pred_class


if __name__ == "__main__":
    import urllib
    pred_class = test_new()

    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    urllib.request.urlretrieve(url, "imagenet_classes.txt")

    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]

    print("Predicted class index:", pred_class.item())
    print("Class name:", labels[pred_class.item()])