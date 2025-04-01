import time
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import movear.models.dist as dist
from movear.models.trainer import VARTrainer
from movear.models.moevar import MoEVAR
from movear.models.vqvae import VQVAE, VectorQuantizer2
from movear.utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor


class MoEVARTrainer(VARTrainer):
    """Trainer for MoE-based VAR model with auxiliary loss handling"""
    
    def __init__(
        self, device, patch_nums, resos,
        vae_local: VQVAE, var_wo_ddp: MoEVAR, var: DDP,
        var_opt, label_smooth: float,
    ):
        super().__init__(device, patch_nums, resos, vae_local, var_wo_ddp, var, var_opt, label_smooth)
        
        # Store MoE specific parameters from the model
        self.aux_loss_weight = var_wo_ddp.aux_loss_weight
        self.num_experts = var_wo_ddp.num_experts
        self.k = var_wo_ddp.k
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        """Training step with MoE auxiliary loss"""
        # Progressive training setup - same as parent class
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
        
        # Forward pass
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        
        with self.var_opt.amp_ctx:
            # Forward pass with MoE auxiliary loss
            logits_BLV, aux_loss = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
            
            # Main loss calculation (CE loss)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            
            # Apply loss weighting
            if prog_si >= 0:    # in progressive training
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:               # not in progressive training
                lw = self.loss_weight
                
            # Final loss with CE and auxiliary loss
            loss = loss.mul(lw).sum(dim=-1).mean()
            # Note: aux_loss is already weighted inside the model
            total_loss = loss + aux_loss
        
        # Backward pass
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=total_loss, stepping=stepping)
        
        # Log metrics
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            
            if prog_si >= 0:    # in progressive training
                Ltail = acc_tail = -1
            else:               # not in progressive training
                Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            
            # Also log MoE-specific metrics
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm, 
                             MoELoss=aux_loss.item())
        
        # Log to tensorboard
        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage, moe_loss=aux_loss.item())
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                
                # Log MoE balance metrics
                balance_metrics = self.get_moe_balance_metrics()
                if balance_metrics:
                    tb_lg.update(head='AR_moe', **balance_metrics, step=g_it)
                
                if prog_si >= 0:
                    tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)
        
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2
    
    def get_moe_balance_metrics(self) -> Dict[str, float]:
        """Get metrics about expert load balancing"""
        metrics = {}
        
        # MoE specific metrics
        try:
            # Collect statistics from each block
            load_balance_metrics = []
            max_loads = []
            min_loads = []
            
            # Extract expert usage metrics from each MoE block
            for i, block in enumerate(self.var_wo_ddp.blocks):
                if hasattr(block, 'ffn') and hasattr(block.ffn, 'aux_data'):
                    routing_probs = block.ffn.aux_data.get("routing_probs", None)
                    if routing_probs is not None:
                        # Calculate expert utilization
                        expert_load = routing_probs.mean(dim=[0, 1])  # [num_experts]
                        max_load = expert_load.max().item()
                        min_load = expert_load.min().item()
                        max_loads.append(max_load)
                        min_loads.append(min_load)
                        load_balance = max_load / (min_load + 1e-8)
                        load_balance_metrics.append(load_balance)
                        
                        metrics[f"balance_{i}"] = load_balance
            
            # Calculate aggregated metrics
            if load_balance_metrics:
                metrics["avg_balance"] = sum(load_balance_metrics) / len(load_balance_metrics)
                metrics["max_balance"] = max(load_balance_metrics)
                metrics["avg_max_load"] = sum(max_loads) / len(max_loads)
                metrics["avg_min_load"] = sum(min_loads) / len(min_loads)
        except Exception as e:
            # Just in case something goes wrong, don't crash the training
            print(f"Error in get_moe_balance_metrics: {e}")
        
        return metrics
    
    def get_config(self):
        """Get trainer configuration"""
        config = super().get_config()
        # Add MoE parameters
        config.update({
            'num_experts': self.num_experts,
            'k': self.k,
            'aux_loss_weight': self.aux_loss_weight,
        })
        return config
    
    @torch.no_grad()
    def evaluate(self, ld_val: DataLoader, epoch=-1, prefix="", limit_batches=-1):
        """Evaluate model on validation data"""
        # Use standard evaluation from parent class
        metrics = super().eval_ep(ld_val)
        
        # Also log MoE-specific metrics
        balance_metrics = self.get_moe_balance_metrics()
        if balance_metrics and dist.is_master():
            print(f"{prefix} MoE balance metrics: " + 
                  ", ".join([f"{k}={v:.4f}" for k, v in balance_metrics.items()]))
        
        return metrics