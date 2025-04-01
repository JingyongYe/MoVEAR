import torch
import torch.distributed as dist
from movear.models.moevar import MoEVAR
from movear.models.vqvae import VQVAE

def build_vae_moe_var(
    # Keep the same parameter order and defaults as build_vae_var
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    num_classes=1000, depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
    num_experts=4, k=2, noise_std=0.1, aux_loss_weight=0.01
):
    """Build VAE and MoE-VAR models with expert parallelism"""
    world_size = dist.get_world_size()
    
    # Validate expert configuration for expert parallelism
    if num_experts % world_size != 0:
        print(f"Warning: {num_experts} experts is not divisible by {world_size} GPUs. Adjusting to {num_experts + (world_size - num_experts % world_size)} experts")
        num_experts = num_experts + (world_size - num_experts % world_size)
    
    print(f"[MoE] Building VAR with {num_experts} experts (distributed across {world_size} GPUs), top-{k}, noise={noise_std}, aux_weight={aux_loss_weight}")
    
    # Build VQVAE (same as original build_vae_var function)
    vae_local = VQVAE(
        vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, 
        share_quant_resi=share_quant_resi, v_patch_nums=patch_nums
    )
    
    # Build MoE-VAR
    var_wo_ddp = MoEVAR(
        vae_local=vae_local,
        num_classes=num_classes, depth=depth, 
        embed_dim=1024, num_heads=16, mlp_ratio=4.0,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        num_experts=num_experts, k=k, noise_std=noise_std, aux_loss_weight=aux_loss_weight
    )
    
    # Initialize weights
    var_wo_ddp.init_weights(
        init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma,
        init_head=init_head, init_std=init_std
    )
    
    # Move to device
    if device is not None:
        vae_local = vae_local.to(device)
        var_wo_ddp = var_wo_ddp.to(device)
    
    return vae_local, var_wo_ddp


def load_pretrained_for_moe(var_wo_ddp, pretrained_path, device='cpu', strict=False):
    """Load a pretrained VAR model into the MoEVAR model
    
    Args:
        var_wo_ddp (MoEVAR): The MoEVAR model to load weights into
        pretrained_path (str): Path to the pretrained VAR model
        device (str): Device to load the weights on
        strict (bool): Whether to strictly enforce parameter presence
        
    Returns:
        MoEVAR: The loaded MoEVAR model
    """
    # Load pretrained VAR model
    pretrained_state_dict = torch.load(pretrained_path, map_location=device)
    
    # Transfer weights from VAR to MoEVAR
    var_wo_ddp.load_from_var(pretrained_state_dict, strict=strict)
    
    print(f"[MoE] Loaded pretrained VAR model from {pretrained_path} into MoEVAR")
    return var_wo_ddp