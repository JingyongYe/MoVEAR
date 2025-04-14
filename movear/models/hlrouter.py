import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import movear.models.dist as dist


class OptimizedScaleAdaptiveRouter(nn.Module):
    """Simplified router with minimal scale awareness for speed and efficiency."""
    
    def __init__(self, input_dim: int, num_experts: int, k: int, 
                 num_scales: int = 10, noise_std: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        self.noise_std = noise_std
        
        # Single scale conditioning matrix instead of embeddings
        self.scale_condition = nn.Parameter(torch.randn(num_scales, 1, 1))
        
        # Simpler router - direct mapping without concatenation
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        
        # For expert parallelism
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.experts_per_rank = num_experts // self.world_size
        
        # Handle case where num_experts is not divisible by world_size
        if num_experts % self.world_size != 0:
            raise ValueError(f"Number of experts ({num_experts}) must be divisible by world_size ({self.world_size})")
    
    def forward(self, x: torch.Tensor, scale_idx: int = None, training: bool = True) -> Tuple[torch.Tensor, Dict]:
        """Optimized routing with minimal overhead"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Apply scale conditioning as a simple multiplicative factor
        if scale_idx is not None:
            # Simple scale conditioning by multiplication - much faster than concatenation
            scale_factor = self.scale_condition[scale_idx]
            router_input = x * (1.0 + scale_factor * 0.1)
        else:
            router_input = x
        
        # Get router logits directly
        router_logits = self.router(router_input)
        
        # Apply noise during training for better expert utilization
        if training and self.noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std
        
        # Get routing probabilities and top-k selection (same as original)
        routing_probs = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_probs, self.k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Create dispatch tensor more efficiently
        dispatch_tensor = torch.zeros_like(routing_probs)
        dispatch_tensor.scatter_(-1, selected_experts, routing_weights)
        
        # Store minimal data for loss calculation
        aux_data = {
            "routing_probs": routing_probs,
            "selected_experts": selected_experts
        }
        
        return dispatch_tensor, aux_data


class OptimizedMoEFFN(nn.Module):
    """Streamlined MoE FFN layer with minimal scale awareness"""
    
    def __init__(self, 
                 embed_dim: int, 
                 mlp_ratio: float,
                 num_experts: int, 
                 k: int,
                 num_scales: int = 10,
                 noise_std: float = 0.1,
                 drop_rate: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.num_experts = num_experts
        self.k = k
        self.hidden_dim = int(embed_dim * mlp_ratio)
        
        # Expert parallelism setup
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.experts_per_rank = num_experts // self.world_size
        
        # Create optimized router
        self.router = OptimizedScaleAdaptiveRouter(
            input_dim=embed_dim,
            num_experts=num_experts,
            k=k,
            num_scales=num_scales,
            noise_std=noise_std
        )
        
        # Create local experts
        start_idx = self.rank * self.experts_per_rank
        end_idx = start_idx + self.experts_per_rank
        
        # Optimize expert implementation for speed
        self.local_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(self.hidden_dim, embed_dim),
                nn.Dropout(drop_rate)
            ) for _ in range(self.experts_per_rank)
        ])
    
    def forward(self, x: torch.Tensor, scale_idx: int = None) -> Tuple[torch.Tensor, Dict]:
        """Optimized forward pass for memory efficiency"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Get routing information
        dispatch_tensor, aux_data = self.router(x, scale_idx=scale_idx, training=self.training)
        
        # Process only local experts
        start_idx = self.rank * self.experts_per_rank
        local_dispatch = dispatch_tensor[:, :, start_idx:start_idx + self.experts_per_rank]
        
        # Pre-allocate output tensor
        expert_outputs = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device)
        
        # Process tokens with local experts
        for local_idx, expert in enumerate(self.local_experts):
            # Skip computation if no tokens are routed to this expert
            expert_weights = local_dispatch[:, :, local_idx].unsqueeze(-1)
            if expert_weights.max() == 0:
                continue
                
            # Apply expert and accumulate weighted output
            expert_outputs += expert(x) * expert_weights
        
        # All-reduce to combine outputs
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