import clip
import torch
import torch.nn.functional as F
import torch.nn as nn

class CLIPLoss(nn.Module):
    def __init__(self, text_prompt: str, device, negative_text_prompt=None):
        super(CLIPLoss, self).__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        text = clip.tokenize([text_prompt]).to(device)
        self.text_features = self.model.encode_text(text)

        self.use_negative_text = False
        if negative_text_prompt is not None:
            negative_text = clip.tokenize([negative_text_prompt]).to(device)
            self.negative_text_features = self.model.encode_text(negative_text)
            self.use_negative_text = True

    def forward(self, image):
        image_features = self.model.encode_image(image)
        if self.use_negative_text:
            cos_sim = -1 * F.cosine_similarity(
                image_features,
                (self.text_features[0]).unsqueeze(0) - (self.self.negative_text_features[0]).unsqueeze(0))
        else:
            cos_sim = -1 * F.cosine_similarity(image_features, (self.text_features[0]).unsqueeze(0))
        return cos_sim.sum()
