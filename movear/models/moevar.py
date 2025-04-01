import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict
from functools import partial
import math

from .var import VAR
from .basic_var import AdaLNSelfAttn, FFN
from .moe import MoEFFN, compute_balance_loss
from .vqvae import VQVAE
from .basic_var import AdaLNSelfAttn, FFN, SelfAttention
from movear.models.helpers import DropPath, drop_path
import movear.models.dist as dist


class MoEAdaLNSelfAttn(nn.Module):
    """Self-attention block with MoE FFN instead of standard FFN"""
    
    def __init__(
        self, cond_dim, shared_aln, block_idx, embed_dim, norm_layer, num_heads,
        mlp_ratio, drop, attn_drop, drop_path, last_drop_p, attn_l2_norm,
        flash_if_available, fused_if_available, num_experts, k, noise_std,
        shared_ada_lin=None  # 新增参数接收父模型的shared_ada_lin
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
        
        # Use MoE FFN instead of standard FFN
        self.ffn = MoEFFN(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            num_experts=num_experts,
            k=k,
            noise_std=noise_std,
            drop_rate=drop
        )
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        
        if self.shared_aln:
            # Properly initialize adaptive parameters for shared attention
            self.ada_gss = nn.Parameter(torch.zeros(1, 1, 6, embed_dim))
            # 存储对父模型shared_ada_lin的引用
            self.shared_ada_lin_ref = shared_ada_lin  
        else:
            # For non-shared attention, create separate adaptive layers
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(cond_dim, 6*embed_dim, bias=True))
    
    def forward(self, x, cond_BD, attn_bias):
        B, L, C = x.shape
        
        # 分片处理不同部分的序列
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
        chunk_size = L // world_size
        local_start = local_rank * chunk_size
        local_end = local_start + chunk_size if local_rank < world_size - 1 else L
        
        # 只处理属于当前GPU的序列部分
        local_x = x[:, local_start:local_end, :]
        
        # 注意力计算（只对本地序列）
        if self.shared_aln:
            cond_reshaped = self.shared_ada_lin_ref(cond_BD).view(-1, 1, 6, self.C)
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_reshaped).unbind(2)
        else:
            adaln_output = self.ada_lin(cond_BD).view(-1, 1, 6, self.C)
            gamma1, gamma2, scale1, scale2, shift1, shift2 = adaln_output.unbind(2)
        
        # 计算仅限于本地序列
        local_norm_x = self.ln_wo_grad(local_x).mul(scale1.add(1)).add_(shift1)
        local_attn_out = self.attn(local_norm_x, attn_bias=attn_bias)
        local_x = local_x + self.drop_path(local_attn_out.mul_(gamma1))
        
        # MoE FFN计算
        local_norm_x = self.ln_wo_grad(local_x).mul(scale2.add(1)).add_(shift2)
        local_moe_out, aux_data = self.ffn(local_norm_x)
        local_x = local_x + self.drop_path(local_moe_out.mul(gamma2))
        
        # 将各GPU的处理结果合并
        all_x = [torch.zeros_like(local_x) for _ in range(world_size)]
        dist.all_gather(all_x, local_x)
        
        # 重构完整序列
        output = torch.cat(all_x, dim=1)
        
        return output, aux_data
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'


class MoEVAR(VAR):
    """VAR model with Mixture of Experts structure"""
    
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
        
        # Replace the transformer blocks with MoE versions
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Re-create blocks with MoE FFN
        self.blocks = nn.ModuleList([
            MoEAdaLNSelfAttn(
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
                shared_ada_lin=self.shared_ada_lin if shared_aln else None
            )
            for block_idx in range(depth)
        ])
        
        print(f"[MoEVAR] Initialized with {num_experts} experts, top-{k}, noise={noise_std}, aux_weight={aux_loss_weight}")

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with MoE auxiliary loss collection"""
        B = label_B.shape[0]
        cond_BD = self.class_emb(label_B)
        
        if self.cond_drop_rate > 0 and self.training:
            cond_mask = torch.rand(B, 1, device=cond_BD.device) > self.cond_drop_rate
            cond_BD = cond_BD * cond_mask.float()
        
        # Token preparation
        BL = x_BLCv_wo_first_l.shape[0] * x_BLCv_wo_first_l.shape[1]
        x_BLCv_wo_first_l = x_BLCv_wo_first_l.reshape(BL, -1)
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        token_emb = self.word_embed(x_BLCv_wo_first_l).reshape(B, -1, self.C)
        first_pos = self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        x = torch.cat([first_pos, token_emb], dim=1)
        
        # Get attention mask if needed
        attn_bias = self.attn_bias_for_masking
        if self.training and hasattr(self, 'prog_si') and self.prog_si >= 0:
            # Progressive training logic
            pass
        
        # Forward through blocks and collect auxiliary losses
        aux_losses = []
        for block in self.blocks:
            x, aux_data = block(x, cond_BD=cond_BD, attn_bias=attn_bias)
            if aux_data is not None:
                # Compute balance loss for this block
                balance_loss = compute_balance_loss(aux_data, self.num_experts)
                aux_losses.append(balance_loss)
        
        # Get output logits
        x = self.head_nm(x, cond_BD)
        logits = self.head(x)
        
        # Compute final auxiliary loss
        aux_loss = None
        if aux_losses and len(aux_losses) > 0:
            aux_loss = torch.stack(aux_losses).mean() * self.aux_loss_weight
        
        return logits, aux_loss

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1):
        """Override weight initialization for MoE variant"""
        print(f"[init_weights] MoEVAR with init_std={init_std}")
        
        if init_std < 0:
            std = math.sqrt(1.0 / self.C)
            init_std = std
        
        # 初始化embeddings和公共层
        nn.init.normal_(self.word_embed.weight, std=init_std)
        nn.init.normal_(self.class_emb.weight, std=init_std)
        nn.init.normal_(self.pos_1LC, std=init_std)
        nn.init.normal_(self.pos_start, std=init_std)
        
        if hasattr(self, 'lvl_embed'):
            nn.init.normal_(self.lvl_embed.weight, std=init_std)
        
        # 初始化transformer blocks
        depth = len(self.blocks)
        for i, sab in enumerate(self.blocks):
            # 初始化注意力权重
            if hasattr(sab.attn, 'qkv'):
                nn.init.normal_(sab.attn.qkv.weight, std=init_std)
            elif hasattr(sab.attn, 'mat_qkv'):
                nn.init.normal_(sab.attn.mat_qkv.weight, std=init_std)
            
            # 投影层初始化
            nn.init.normal_(sab.attn.proj.weight, std=init_std)
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            
            # 初始化MoE专家 - 使用local_experts而不是experts
            if hasattr(sab.ffn, 'local_experts'):
                for expert in sab.ffn.local_experts:
                    # 第一个线性层
                    nn.init.normal_(expert[0].weight, std=init_std)
                    # 第二个线性层
                    nn.init.normal_(expert[3].weight, std=init_std)
                    expert[3].weight.data.div_(math.sqrt(2 * depth))
            
            # 初始化路由器
            if hasattr(sab.ffn, 'router'):
                nn.init.normal_(sab.ffn.router.router.weight, std=init_std)
            
            # 初始化自适应层范数参数
            if sab.shared_aln:
                if hasattr(sab, 'ada_gss'):
                    nn.init.normal_(sab.ada_gss, std=init_adaln_gamma)
            else:
                if hasattr(sab, 'ada_lin'):
                    for module in sab.ada_lin:
                        if isinstance(module, nn.Linear):
                            nn.init.constant_(module.weight, init_adaln)
                            nn.init.normal_(module.bias, std=init_adaln_gamma)
            
            # 初始化gamma参数（如果存在）
            if hasattr(sab, 'gamma1'):
                nn.init.ones_(sab.gamma1)
                nn.init.ones_(sab.gamma2)
        
        # 初始化head
        nn.init.normal_(self.head.weight, std=init_head)
        nn.init.zeros_(self.head.bias)
        
        return self