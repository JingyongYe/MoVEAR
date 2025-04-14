import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict
from functools import partial
import math

from .var import VAR
from .basic_var import AdaLNSelfAttn, FFN
from .basic_var import AdaLNSelfAttn, FFN, SelfAttention
from .hlrouter import ScaleAdaptiveMoEFFN, compute_scale_aware_balance_loss
from .vqvae import VQVAE
from movear.models.helpers import DropPath, drop_path
import movear.models.dist as dist


class HLMoEAdaLNSelfAttn(nn.Module):
    """Self-attention block with scale-aware MoE FFN"""
    
    def __init__(
        self, cond_dim, shared_aln, block_idx, embed_dim, norm_layer, num_heads,
        mlp_ratio, drop, attn_drop, drop_path, last_drop_p, attn_l2_norm,
        flash_if_available, fused_if_available, num_experts, k, noise_std,
        num_scales=10, scale_embed_dim=64,
        shared_ada_lin=None
    ):
        super().__init__()
        self.block_idx = block_idx
        self.last_drop_p = last_drop_p
        self.C = self.D = embed_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SelfAttention(
            block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, 
            attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, 
            flash_if_available=flash_if_available
        )
        
        # Use Scale-Adaptive MoE FFN
        self.ffn = ScaleAdaptiveMoEFFN(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            num_experts=num_experts,
            k=k,
            num_scales=num_scales,
            scale_embed_dim=scale_embed_dim,
            noise_std=noise_std,
            drop_rate=drop
        )
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.zeros(1, 1, 6, embed_dim))
            self.shared_ada_lin_ref = shared_ada_lin
        else:
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(cond_dim, 6*embed_dim, bias=True))
    
    def forward(self, x, cond_BD, attn_bias, scale_idx=None):
        B, L, C = x.shape
        
        # Process attention same as original
        if self.shared_aln:
            cond_reshaped = self.shared_ada_lin_ref(cond_BD).view(-1, 1, 6, self.C)
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_reshaped).unbind(2)
        else:
            adaln_output = self.ada_lin(cond_BD).view(-1, 1, 6, self.C)
            gamma1, gamma2, scale1, scale2, shift1, shift2 = adaln_output.unbind(2)
        
        norm_x = self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1)
        attn_out = self.attn(norm_x, attn_bias=attn_bias)
        x = x + self.drop_path(attn_out.mul_(gamma1))
        
        # Use scale_idx for scale-aware MoE FFN
        norm_x = self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)
        moe_out, aux_data = self.ffn(norm_x, scale_idx=scale_idx)
        x = x + self.drop_path(moe_out.mul(gamma2))
        
        return x, aux_data
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'


class HLMoEVAR(VAR):
    """Enhanced VAR model with scale-adaptive mixture of experts"""
    
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4.,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        flash_if_available=True, fused_if_available=True,
        num_experts=8, k=2, noise_std=0.1, aux_loss_weight=0.01
    ):
        # Initialize parent class
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, 
            depth=depth, 
            embed_dim=embed_dim, 
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_eps=norm_eps,
            shared_aln=shared_aln,
            cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available,
            fused_if_available=fused_if_available
        )
        
        # Store MoE specific parameters
        self.num_experts = num_experts
        self.k = k
        self.noise_std = noise_std
        self.aux_loss_weight = aux_loss_weight
        
        # Replace the transformer blocks with scale-aware MoE versions
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Re-create blocks with scale-aware MoE FFN
        self.blocks = nn.ModuleList([
            HLMoEAdaLNSelfAttn(
                cond_dim=self.D, 
                shared_aln=shared_aln,
                block_idx=block_idx, 
                embed_dim=self.C, 
                norm_layer=norm_layer, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[block_idx], 
                last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, 
                fused_if_available=fused_if_available,
                num_experts=num_experts, 
                k=k, 
                noise_std=noise_std,
                num_scales=len(patch_nums),
                shared_ada_lin=self.shared_ada_lin if shared_aln else None
            )
            for block_idx in range(depth)
        ])
        
        print(f"[HLMoEVAR] Initialized with {num_experts} experts, top-{k}, noise={noise_std}, aux_weight={aux_loss_weight}")

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor, prog_si: int = -1) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with scale-aware MoE routing"""
        B = label_B.shape[0]
        cond_BD = self.class_emb(label_B)
        
        if self.cond_drop_rate > 0 and self.training:
            cond_mask = torch.rand(B, 1, device=cond_BD.device) > self.cond_drop_rate
            cond_BD = cond_BD * cond_mask.float()
        
        # Token preparation (same as original)
        BL = x_BLCv_wo_first_l.shape[0] * x_BLCv_wo_first_l.shape[1]
        x_BLCv_wo_first_l = x_BLCv_wo_first_l.reshape(BL, -1)
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        token_emb = self.word_embed(x_BLCv_wo_first_l).reshape(B, -1, self.C)
        first_pos = self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        x = torch.cat([first_pos, token_emb], dim=1)
        
        # Get attention mask if needed (same as original)
        attn_bias = self.attn_bias_for_masking
        
        # Use prog_si for scale-aware routing
        current_scale = prog_si if prog_si >= 0 else len(self.patch_nums)-1
        
        # Forward through blocks and collect auxiliary losses with scale awareness
        aux_losses = []
        for block in self.blocks:
            x, aux_data = block(x, cond_BD=cond_BD, attn_bias=attn_bias, scale_idx=current_scale)
            if aux_data is not None:
                # Compute scale-aware balance loss for this block
                balance_loss = compute_scale_aware_balance_loss(aux_data, self.num_experts)
                aux_losses.append(balance_loss)
        
        # Get output logits (same as original)
        x = self.head_nm(x, cond_BD)
        logits = self.head(x)
        
        # Compute final auxiliary loss
        aux_loss = None
        if aux_losses and len(aux_losses) > 0:
            aux_loss = torch.stack(aux_losses).mean() * self.aux_loss_weight
        
        return logits, aux_loss

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1):
        """Override weight initialization for HLMoEVAR variant"""
        print(f"[init_weights] HLMoEVAR with init_std={init_std}")
        
        if init_std < 0:
            std = math.sqrt(1.0 / self.C)
            init_std = std
        
        # Initialize embeddings and common layers
        nn.init.normal_(self.word_embed.weight, std=init_std)
        nn.init.normal_(self.class_emb.weight, std=init_std)
        nn.init.normal_(self.pos_1LC, std=init_std)
        nn.init.normal_(self.pos_start, std=init_std)
        
        if hasattr(self, 'lvl_embed'):
            nn.init.normal_(self.lvl_embed.weight, std=init_std)
        
        # Initialize transformer blocks
        depth = len(self.blocks)
        for i, sab in enumerate(self.blocks):
            # Initialize attention weights
            if hasattr(sab.attn, 'qkv'):
                nn.init.normal_(sab.attn.qkv.weight, std=init_std)
            elif hasattr(sab.attn, 'mat_qkv'):
                nn.init.normal_(sab.attn.mat_qkv.weight, std=init_std)
            
            # Initialize projection layer
            nn.init.normal_(sab.attn.proj.weight, std=init_std)
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            
            # Initialize MoE experts
            if hasattr(sab.ffn, 'local_experts'):
                for expert in sab.ffn.local_experts:
                    # First linear layer
                    nn.init.normal_(expert[0].weight, std=init_std)
                    # Second linear layer
                    nn.init.normal_(expert[3].weight, std=init_std)
                    expert[3].weight.data.div_(math.sqrt(2 * depth))
            
            # Initialize router
            if hasattr(sab.ffn, 'router'):
                nn.init.normal_(sab.ffn.router.router.weight, std=init_std)
                # Initialize scale embeddings if they exist
                if hasattr(sab.ffn.router, 'scale_embeddings'):
                    nn.init.normal_(sab.ffn.router.scale_embeddings, std=0.02)
            
            # Initialize adaptive layer norm parameters
            if sab.shared_aln:
                if hasattr(sab, 'ada_gss'):
                    nn.init.normal_(sab.ada_gss, std=init_adaln_gamma)
            else:
                if hasattr(sab, 'ada_lin'):
                    for module in sab.ada_lin:
                        if isinstance(module, nn.Linear):
                            nn.init.constant_(module.weight, init_adaln)
                            nn.init.normal_(module.bias, std=init_adaln_gamma)
            
            # Initialize gamma parameters (if they exist)
            if hasattr(sab, 'gamma1'):
                nn.init.ones_(sab.gamma1)
                nn.init.ones_(sab.gamma2)
        
        # Initialize head
        nn.init.normal_(self.head.weight, std=init_head)
        nn.init.zeros_(self.head.bias)
        
        return self