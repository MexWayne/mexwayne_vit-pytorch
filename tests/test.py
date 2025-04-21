import torch
from vit_pytorch import ViT
import cv2
import numpy as np
import random
from transformers import ViTForImageClassification, ViTImageProcessor

# google/vit-base-patch16-224 from hugging face
def load_pretrain_model(vit, hf_model_name='google/vit-base-patch16-384'):

    from vit_pytorch import ViT
    import cv2
    import numpy as np
    import os
    from collections import OrderedDict
    from transformers import ViTModel

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')
    cfg = model.config
    print("hugging face model info: ###############################")
    print(f"hidden_size (dim): {cfg.hidden_size}")
    print(f"num_hidden_layers (depth): {cfg.num_hidden_layers}")
    print(f"num_attention_heads (heads): {cfg.num_attention_heads}")
    print(f"intermediate_size (mlp_dim): {cfg.intermediate_size}")
    print(f"image size: {cfg.image_size}")
    print(f"patch size: {cfg.patch_size}")
    print("#########################################################")


    print(">> Loading HuggingFace ViT weights...")
    hf_model = ViTModel.from_pretrained(hf_model_name)
    hf_state_dict = hf_model.state_dict()
    custom_state_dict = vit.state_dict()
    new_state_dict = OrderedDict()
    
    # cls token and pos embedding
    new_state_dict['cls_token'] = hf_state_dict['embeddings.cls_token']
    new_state_dict['pos_embedding'] = hf_state_dict['embeddings.position_embeddings']
    
    # patch embedding
    w_patch = hf_state_dict['embeddings.patch_embeddings.projection.weight']  # [768, 3, 16, 16]
    b_patch = hf_state_dict['embeddings.patch_embeddings.projection.bias']    # [768]
    new_state_dict['to_patch_embedding.2.weight'] = w_patch.flatten(1).T      # [768, 768]
    new_state_dict['to_patch_embedding.2.bias'] = b_patch
    
    # transformer blocks
    for i in range(12):  # ViT-base has 12 layers
        prefix_hf = f'encoder.layer.{i}'
        prefix_my = f'transformer.layers.{i}'
    
        # qkv weight (concat)
        q = hf_state_dict[f'{prefix_hf}.attention.attention.query.weight']
        k = hf_state_dict[f'{prefix_hf}.attention.attention.key.weight']
        v = hf_state_dict[f'{prefix_hf}.attention.attention.value.weight']
        qkv = torch.cat([q, k, v], dim=0)
        new_state_dict[f'{prefix_my}.0.to_qkv.weight'] = qkv
    
        # attention out
        new_state_dict[f'{prefix_my}.0.to_out.0.weight'] = hf_state_dict[f'{prefix_hf}.attention.output.dense.weight']
        new_state_dict[f'{prefix_my}.0.to_out.0.bias']   = hf_state_dict[f'{prefix_hf}.attention.output.dense.bias']
        new_state_dict[f'{prefix_my}.0.norm.weight']     = hf_state_dict[f'{prefix_hf}.layernorm_before.weight']
        new_state_dict[f'{prefix_my}.0.norm.bias']       = hf_state_dict[f'{prefix_hf}.layernorm_before.bias']
    
        # FFN
        new_state_dict[f'{prefix_my}.1.net.0.weight'] = hf_state_dict[f'{prefix_hf}.layernorm_after.weight']
        new_state_dict[f'{prefix_my}.1.net.0.bias']   = hf_state_dict[f'{prefix_hf}.layernorm_after.bias']
        new_state_dict[f'{prefix_my}.1.net.1.weight'] = hf_state_dict[f'{prefix_hf}.intermediate.dense.weight']
        new_state_dict[f'{prefix_my}.1.net.1.bias']   = hf_state_dict[f'{prefix_hf}.intermediate.dense.bias']
        new_state_dict[f'{prefix_my}.1.net.4.weight'] = hf_state_dict[f'{prefix_hf}.output.dense.weight']
        new_state_dict[f'{prefix_my}.1.net.4.bias']   = hf_state_dict[f'{prefix_hf}.output.dense.bias']
    
    # final norm
    new_state_dict['transformer.norm.weight'] = hf_state_dict['layernorm.weight']
    new_state_dict['transformer.norm.bias']   = hf_state_dict['layernorm.bias']
    
    custom_state_dict.update(new_state_dict)
    # if the strict = false the classify results will be random results
    vit.load_state_dict(custom_state_dict, strict=True) 
    print(">> Done loading weights!")


def test():
    img = cv2.imread("../datasets/000000000110.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))


    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-384')
    inputs = processor(images=img, return_tensors="pt")


    # (384 / 16) * (384 / 16) = 24 * 24 = 576

    # patch_height = 16 = patch_width 
    # 3 * 16 * 16 = 768 = patch_dim = channels * patch_height * patch_width
    v = ViT(
        image_size=384,     
        patch_size=16,        # patch_height and patch_width
        num_classes=1000,     # hugging face classess amount
        dim=768,              # patch_dim
        depth=12,             # 
        heads=12,
        mlp_dim=3072,
        dropout=0.0,
        emb_dropout=0.0
    )

    # model
    if False:
        # init from a model saved in a specific directory
        print("hehe")
    else: # from hugging face 
        # init from a given GPT-2 model
        load_pretrain_model(v)

    img = img.astype(np.float32) / 255.0
    torch_tensor = torch.from_numpy(img).permute(2, 0, 1)  # shape: [3, H, W]
    img = torch_tensor.unsqueeze(0)

    with torch.no_grad():
        v.eval() #with this code, the results will run as random
        preds = v(img)
        assert preds.shape == (1, 1000), 'correct logits outputted'

    return preds

if __name__ == "__main__":

    preds = test()

    predicted_class = torch.argmax(preds, dim=-1)
    print("Predicted class index:", predicted_class.item())

    import json
    import urllib
    
    # down load the classes 
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    filename = "imagenet_classes.txt"
    urllib.request.urlretrieve(url, filename)
    
    # read the class list 
    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]


    
    print("Class name:", labels[predicted_class.item()])
    print("\n>>> finished!")





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
