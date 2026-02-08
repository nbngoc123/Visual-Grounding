"""TransVG Model Architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import BertModel

from base import BaseModel


class MLP(nn.Module):
    """MLP for bbox prediction."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.num_layers = num_layers

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class VisualLinguisticTransformer(nn.Module):
    """VL Transformer with [REG] token."""
    def __init__(self, d_model, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, mask=None):
        return self.norm(self.transformer(tokens, src_key_padding_mask=mask))


class TransVG(BaseModel):
    """TransVG: End-to-End Visual Grounding with Transformers."""
    
    BACKBONES = {
        'vit': ('vit_base_patch16_224', 768, 196),
        'convnext': ('convnextv2_base.fcmae_ft_in22k_in1k', 1024, 49),
    }
    
    def __init__(self, backbone_type, config):
        super().__init__()
        self.backbone_type = backbone_type
        
        # Vision backbone
        model_name, vision_dim, self.num_visu_token = self.BACKBONES[backbone_type]
        self.vision_encoder = timm.create_model(model_name, pretrained=True, num_classes=0)

        # Text encoder
        self.text_encoder = BertModel.from_pretrained(config.text_encoder)
        self.num_text_token = config.max_text_len

        # Projections
        self.vision_proj = nn.Linear(vision_dim, config.hidden_dim)
        self.text_proj = nn.Linear(768, config.hidden_dim)

        # [REG] token and positional embeddings
        self.reg_token = nn.Embedding(1, config.hidden_dim)
        num_total = 1 + self.num_visu_token + self.num_text_token
        self.vl_pos_embed = nn.Embedding(num_total, config.hidden_dim)

        # VL Transformer & Bbox head
        self.vl_transformer = VisualLinguisticTransformer(d_model=config.hidden_dim)
        self.bbox_head = MLP(config.hidden_dim, config.hidden_dim, 4, 3)

    def forward(self, image, input_ids, attention_mask):
        B = image.shape[0]
        device = image.device

        # Visual features
        v = self.vision_encoder.forward_features(image)
        if isinstance(v, dict): v = v["x"]
        visual_feat = v[:, 1:, :] if self.backbone_type == 'vit' else v.flatten(2).transpose(1, 2)
        visual_tokens = self.vision_proj(visual_feat)

        # Text features
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = self.text_proj(text_out.last_hidden_state)

        # [REG] + visual + text
        reg_tokens = self.reg_token.weight.unsqueeze(0).expand(B, -1, -1)
        all_tokens = torch.cat([reg_tokens, visual_tokens, text_tokens], dim=1)
        all_tokens = all_tokens + self.vl_pos_embed.weight.unsqueeze(0)

        # Attention mask
        mask = torch.cat([
            torch.zeros(B, 1 + self.num_visu_token, dtype=torch.bool, device=device),
            ~attention_mask.bool()
        ], dim=1)

        # Predict bbox from [REG] token
        fused = self.vl_transformer(all_tokens, mask=mask)
        return self.bbox_head(fused[:, 0, :]).sigmoid()
