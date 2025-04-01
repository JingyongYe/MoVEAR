import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict
from functools import partial

from .var import VAR, AdaLNSelfAttn
from .basic_var import AdaLNBeforeHead
from .moe import MoEFFN, compute_balance_loss
from .vqvae import VQVAE


class MoEAdaLNSelfAttn(AdaLNSelfAttn):
    """Self-attention block with MoE FFN instead of standard FFN"""
    
    def __init__(
        self, cond_dim, shared_aln, block_idx, 
        embed_dim, norm_layer, num_heads, mlp_ratio, 
        drop, attn_drop, drop_path, last_drop_p, 
        attn_l2_norm=False, 
        flash_if_available=True, fused_if_available=True,
        num_experts=8, k=2, noise_std=0.1
    ):
        # Initialize parent class properly
        super().__init__(
            block_idx=block_idx, last_drop_p=last_drop_p, 
            embed_dim=embed_dim, cond_dim=cond_dim, shared_aln=shared_aln, 
            norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path, 
            attn_l2_norm=attn_l2_norm,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available
        )
        
        # Replace FFN with MoE FFN
        self.ffn = MoEFFN(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            num_experts=num_experts,
            k=k,
            noise_std=noise_std,
            drop_rate=drop
        )
        
    def forward(self, x, cond_BD, attn_bias=None):
        x = x + self.drop_path(self.attn(self.norm1(x.float()), cond_BD, attn_bias))
        ffn_output, self.aux_data = self.ffn(self.norm2(x.float()), training=self.training)
        x = x + self.drop_path(ffn_output)
        return x


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
        # Initialize with parent class but without creating the blocks
        nn.Module.__init__(self)
        
        # Store MoE specific parameters
        self.num_experts = num_experts
        self.k = k
        self.noise_std = noise_std
        self.aux_loss_weight = aux_loss_weight
        
        # Initialize VAR parameters
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1
        
        self.patch_nums = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize all the embeddings and components from original VAR
        self._init_embeddings(vae_local, embed_dim, init_std=0.02)
        
        # Create MoE transformer blocks
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.ModuleList([
            MoEAdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
                num_experts=num_experts, k=k, noise_std=noise_std
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        
        # Initialize attention mask
        self._init_attention_mask()
        
        # Initialize classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
        
        # Initialize uniform probability for class sampling
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, 
                                        dtype=torch.float32, 
                                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def _init_embeddings(self, vae_local, embed_dim, init_std=0.02):
        """Initialize all the embeddings used in VAR"""
        # Class embedding
        num_classes_plus_1 = self.num_classes + 1  # +1 for unconditional generation
        self.class_emb = nn.Embedding(num_classes_plus_1, self.D)
        
        # Position embeddings
        self.pos_start = nn.Parameter(torch.zeros(1, 1, self.C))
        self.pos_1LC = nn.Parameter(torch.zeros(self.L, self.C))
        self.lvl_1L = torch.arange(self.L).long().view(-1)
        self.lvl_embed = nn.Embedding(self.L, self.C)
        
        # Token embedding
        self.word_embed = nn.Linear(self.Cvae, self.C, bias=False)
        
        # Quantization proxy
        self.vae_quant_proxy = nn.ModuleList([vae_local.quantize])
        self.vae_proxy = nn.ModuleList([vae_local])
        
        # Apply initialization
        std = init_std if init_std > 0 else (1.0 / embed_dim ** 0.5)
        nn.init.normal_(self.pos_start, std=std)
        nn.init.normal_(self.pos_1LC, std=std)
        nn.init.normal_(self.class_emb.weight, std=std)
        nn.init.normal_(self.lvl_embed.weight, std=std)
        nn.init.normal_(self.word_embed.weight, std=std)
    
    def _init_attention_mask(self):
        """Initialize attention mask for autoregressive generation"""
        # Create causal attention mask for autoregressive generation
        self.register_buffer("attn_bias_for_masking", torch.zeros(1, 1, self.L, self.L))
        mask = torch.full((self.L, self.L), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.attn_bias_for_masking[0, 0] = mask
    
    def get_logits(self, x, cond_BD):
        """Get logits from the hidden states"""
        x = self.head_nm(x, cond_BD)
        return self.head(x)
        
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with MoE auxiliary loss"""
        # Initialize embedding, position, and conditional inputs
        B = label_B.shape[0]
        cond_BD = self.class_emb(label_B)
        
        if self.cond_drop_rate > 0 and self.training:
            cond_BD = F.dropout(cond_BD, p=self.cond_drop_rate, training=self.training)
        
        # Process input tokens
        BL = x_BLCv_wo_first_l.shape[0] * x_BLCv_wo_first_l.shape[1]
        x_BLCv_wo_first_l = x_BLCv_wo_first_l.reshape(BL, -1)
        
        # Create embeddings for first level tokens
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        token_emb = self.word_embed(x_BLCv_wo_first_l).reshape(B, -1, self.C)
        
        # Position embeddings for all tokens
        first_pos = self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        # Combine first level tokens and subsequent tokens
        x = torch.cat([first_pos, token_emb], dim=1)
        
        # Get attention mask
        attn_bias = self.attn_bias_for_masking
        if self.training and self.prog_si >= 0:
            attn_bias = attn_bias.clone()
            si = self.prog_si
            bg, ed = self.begin_ends[si]
            ratio = si / self.num_stages_minus_1
            
            # Apply progressive training mask
            attn_bias[:, :, bg:ed, :bg] = float("-inf")
            
        # Forward through blocks with MoE and collect auxiliary losses
        aux_losses = []
        for block in self.blocks:
            x = block(x, cond_BD=cond_BD, attn_bias=attn_bias)
            if hasattr(block, 'aux_data'):
                aux_losses.append(compute_balance_loss(block.aux_data, self.num_experts))
        
        # Get output logits
        logits = self.get_logits(x, cond_BD)
        
        # Compute total auxiliary loss
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=logits.device)
        
        return logits, self.aux_loss_weight * aux_loss
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02):
        """Initialize model weights with custom parameters"""
        # Initialize AdaLN weights
        for b in self.blocks:
            if hasattr(b, 'shared_ada_lin') and b.shared_ada_lin is not None:
                nn.init.constant_(b.shared_ada_lin.weight, init_adaln)
                nn.init.constant_(b.shared_ada_lin.bias, 0.0)
                if init_adaln_gamma > 0:
                    nn.init.normal_(b.shared_ada_lin.weight, mean=init_adaln, std=init_adaln_gamma)
        
        # Initialize head weights
        if hasattr(self, 'head') and hasattr(self.head, 'weight'):
            nn.init.normal_(self.head.weight, std=init_head)
            if hasattr(self.head, 'bias') and self.head.bias is not None:
                nn.init.zeros_(self.head.bias)
        
        # Apply custom initialization to embeddings
        if init_std > 0:
            std = init_std
            nn.init.normal_(self.pos_start, std=std)
            nn.init.normal_(self.pos_1LC, std=std)
            nn.init.normal_(self.class_emb.weight, std=std)
            nn.init.normal_(self.lvl_embed.weight, std=std)
            nn.init.normal_(self.word_embed.weight, std=std)
            
        print(f"[init_weights] MoEVAR with init_std={init_std:.7f}")
        return self
        
    def load_from_var(self, var_state_dict, strict=False):
        """Load weights from a regular VAR model"""
        # This is useful for transferring from a pre-trained VAR model
        missing_keys, unexpected_keys = [], []
        
        # Load all keys that match between the models
        for name, param in self.named_parameters():
            if name in var_state_dict:
                param.data.copy_(var_state_dict[name])
            else:
                missing_keys.append(name)
                
        # For expert parameters, initialize from the original FFN weights
        for block_idx, block in enumerate(self.blocks):
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'experts'):
                # Original VAR MLP weights format
                orig_mlp1_name = f"blocks.{block_idx}.mlp.fc1.weight"
                orig_mlp2_name = f"blocks.{block_idx}.mlp.fc2.weight"
                
                if orig_mlp1_name in var_state_dict and orig_mlp2_name in var_state_dict:
                    # Copy weights to each expert
                    for expert in block.ffn.experts:
                        expert[0].weight.data.copy_(var_state_dict[orig_mlp1_name])
                        expert[3].weight.data.copy_(var_state_dict[orig_mlp2_name])
                
        if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
            raise RuntimeError(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
        
        return self