import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import movear.models.dist as dist


class ScaleAdaptiveRouter(nn.Module):
    """Enhanced router with scale awareness for cross-scale continuity and stability."""
    
    def __init__(self, input_dim: int, num_experts: int, k: int, 
                 num_scales: int = 10, scale_embed_dim: int = 64,
                 noise_std: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        self.noise_std = noise_std
        
        # Scale embeddings - one for each scale in the hierarchy
        self.scale_embeddings = nn.Parameter(torch.randn(num_scales, scale_embed_dim))
        
        # Router with concatenated input (feature + scale embedding)
        self.router = nn.Linear(input_dim + scale_embed_dim, num_experts, bias=False)
        
        # For expert parallelism
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.experts_per_rank = num_experts // self.world_size
        
        # Handle case where num_experts is not divisible by world_size
        if num_experts % self.world_size != 0:
            raise ValueError(f"Number of experts ({num_experts}) must be divisible by world_size ({self.world_size})")
    
    def forward(self, x: torch.Tensor, scale_idx: int = None, training: bool = True) -> Tuple[torch.Tensor, Dict]:
        """Route input to top-k experts with expert parallelism and scale awareness"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Get scale embedding if provided
        if scale_idx is not None:
            scale_emb = self.scale_embeddings[scale_idx].unsqueeze(0).unsqueeze(0)
            scale_emb = scale_emb.expand(batch_size, seq_len, -1)
            # Concatenate with input
            x_with_scale = torch.cat([x, scale_emb], dim=-1)
        else:
            # Use average of scale embeddings if no specific scale is provided
            scale_emb = self.scale_embeddings.mean(dim=0).unsqueeze(0).unsqueeze(0)
            scale_emb = scale_emb.expand(batch_size, seq_len, -1)
            x_with_scale = torch.cat([x, scale_emb], dim=-1)
        
        # Get router logits for all experts
        router_logits = self.router(x_with_scale)  # [batch_size, seq_len, num_experts]
        
        # Apply noise during training for better expert utilization
        if training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        
        # Get routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(routing_probs, self.k, dim=-1)
        
        # Normalize the routing weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Create dispatch tensor (one-hot encoding for selected experts with weights)
        # We still create the full dispatch tensor for auxiliary loss computation
        dispatch_tensor = torch.zeros_like(routing_probs)
        dispatch_tensor.scatter_(-1, selected_experts, routing_weights)
        
        # Store data for auxiliary loss calculation
        aux_data = {
            "routing_probs": routing_probs,
            "selected_experts": selected_experts,
            "routing_weights": routing_weights,
            "scale_idx": scale_idx
        }
        
        return dispatch_tensor, aux_data


class ScaleAdaptiveMoEFFN(nn.Module):
    """MoE Feed Forward Network layer with scale-adaptive routing."""
    
    def __init__(self, 
                 embed_dim: int, 
                 mlp_ratio: float,
                 num_experts: int, 
                 k: int,
                 num_scales: int = 10,
                 scale_embed_dim: int = 64,
                 noise_std: float = 0.1,
                 drop_rate: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.num_experts = num_experts
        self.k = k
        self.hidden_dim = int(embed_dim * mlp_ratio)
        
        # Setup parallel expert configuration
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.experts_per_rank = num_experts // self.world_size
        
        # Handle case where num_experts is not divisible by world_size
        if num_experts % self.world_size != 0:
            raise ValueError(f"Number of experts ({num_experts}) must be divisible by world_size ({self.world_size})")
        
        # Create scale-adaptive router
        self.router = ScaleAdaptiveRouter(
            input_dim=embed_dim,
            num_experts=num_experts,
            k=k,
            num_scales=num_scales,
            scale_embed_dim=scale_embed_dim,
            noise_std=noise_std
        )
        
        # Create only the local experts for this rank
        start_idx = self.rank * self.experts_per_rank
        end_idx = start_idx + self.experts_per_rank
        
        self.local_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(self.hidden_dim, embed_dim),
                nn.Dropout(drop_rate)
            ) for _ in range(start_idx, end_idx)
        ])
    
    def forward(self, x: torch.Tensor, scale_idx: int = None) -> Tuple[torch.Tensor, Dict]:
        """Forward pass through distributed MoE layer with scale awareness"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Get global routing probabilities with scale awareness
        dispatch_tensor, aux_data = self.router(x, scale_idx=scale_idx, training=self.training)
        
        # Extract the dispatch weights only for local experts
        start_idx = self.rank * self.experts_per_rank
        end_idx = start_idx + self.experts_per_rank
        local_dispatch = dispatch_tensor[:, :, start_idx:end_idx]
        
        # Process tokens with local experts
        expert_outputs = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device)
        
        # For each local expert, process its tokens
        for local_idx, expert in enumerate(self.local_experts):
            # Global expert index
            expert_idx = start_idx + local_idx
            
            # Get expert weights for this expert
            expert_weights = local_dispatch[:, :, local_idx].unsqueeze(-1)
            
            # Skip computation if no tokens are routed to this expert
            if expert_weights.max() == 0:
                continue
                
            # Apply expert to input and weight the output
            expert_output = expert(x)
            expert_outputs += expert_output * expert_weights
        
        # All-reduce to combine outputs from all experts across devices
        dist.allreduce(expert_outputs)
        
        return expert_outputs, aux_data


def compute_scale_aware_balance_loss(aux_data: Dict, num_experts: int) -> torch.Tensor:
    """Enhanced load balancing loss with scale awareness"""
    # Extract routing probabilities - these are already global across all devices
    routing_probs = aux_data["routing_probs"]  # [batch_size, seq_len, num_experts]
    
    # Compute load (fraction of tokens routed to each expert)
    load = routing_probs.mean(dim=[0, 1])  # [num_experts]
    
    # Compute importance (sum of router probabilities for each expert)
    importance = routing_probs.sum(dim=[0, 1])  # [num_experts]
    importance = importance / importance.sum()
    
    # Compute CV loss (coefficient of variation)
    cv_loss = (load.std() / (load.mean() + 1e-6))
    
    # Compute balance loss
    balance_loss = (load * importance).sum() * num_experts
    
    return cv_loss + balance_loss