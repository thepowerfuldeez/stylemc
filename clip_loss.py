import clip
import torch
import torch.nn.functional as F
import torch.nn as nn


class CLIPLoss(nn.Module):
    def __init__(self, device, text_prompt: str, negative_text_prompt: str, clip_type='small'):
        super().__init__()
        if clip_type == "small":
            self.model, _ = clip.load("ViT-B/32", device=device)
        else:
            self.model, _ = clip.load("ViT-B/16", device=device)

        self.tgt_text_features = self.model.encode_text(clip.tokenize([text_prompt]).to(device))
        self.neg_text_features = self.model.encode_text(clip.tokenize([negative_text_prompt]).to(device))
        self.text_features = self.tgt_text_features - self.neg_text_features
        self.text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)

    def compute_image_embedding(self, image):
        image_features = self.model.encode_image(image)
        return image_features / image_features.norm(dim=1, keepdim=True)

    def forward(self, src_image, tgt_image):
        src_image_features = self.model.encode_image(src_image)
        tgt_image_features = self.model.encode_image(tgt_image)
        image_features = tgt_image_features - src_image_features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        cos_sim = F.cosine_similarity(
            image_features,
            self.text_features
        )

        return (len(src_image) - cos_sim.sum()) / len(src_image)
