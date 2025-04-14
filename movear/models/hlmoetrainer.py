import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import movear.models.dist as dist
from movear.models.moevar import MoEVAR
from movear.models.vqvae import VQVAE, VectorQuantizer2
from movear.utils.amp_sc import AmpOptimizer
from movear.utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class HLMoEVARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp: MoEVAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
        theory_weight: float = 0.01,  # Consolidated parameter
    ):
        super().__init__()
        
        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: MoEVAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt
        
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        
        self.aux_loss_weight = var_wo_ddp.aux_loss_weight
        
        # Store the consolidated weight for theoretical losses
        self.theory_weight = theory_weight
    
    @torch.no_grad()
    def evaluate(self, ld_val: DataLoader):
        """Evaluate the model on the validation dataset."""
        return self.eval_ep(ld_val)
        
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        """Evaluate the model on the validation dataset."""
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        
        for i, (inp_B3HW, label_B) in enumerate(ld_val):
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            # 处理当前批次
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            
            # 使用前向方法直接获取输出和损失
            logits_BLV, _ = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
            
            # 计算指标
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            tot += B
            
            # 显式删除不再需要的张量并清理缓存
            del gt_idx_Bl, gt_BL, x_BLCv_wo_first_l, logits_BLV
            if (i + 1) % 200 == 0:  # 每10个批次清理一次缓存
                torch.cuda.empty_cache()
        
        self.var_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        """Optimized training step with unified theoretical constraints"""
        # Handle progressive training (same as original)
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1
        if prog_si == len(self.patch_nums) - 1: prog_si = -1
        
        # Forward pass
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        
        with self.var_opt.amp_ctx:
            # Get MoE outputs and auxiliary loss
            logits_BLV, aux_loss = self.var(label_B, x_BLCv_wo_first_l, prog_si=prog_si)
            
            # Main classification loss
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            
            # Handle progressive training loss weighting (same as original)
            if prog_si >= 0:
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:
                lw = self.loss_weight
                
            # Combine main loss with MoE auxiliary loss
            main_loss = loss.mul(lw).sum(dim=-1).mean()
            total_loss = main_loss
            
            # Add auxiliary loss if provided
            if aux_loss is not None:
                total_loss = main_loss + self.aux_loss_weight * aux_loss
            
            # Only compute theoretical losses when:
            # 1. Not in progressive training (prog_si < 0)
            # 2. Every 3 steps to reduce computation overhead
            # 3. Or on first and periodic logging iterations 
            compute_theory = (prog_si < 0) and (it % 3 == 0 or it == 0 or it in metric_lg.log_iters)
            
            if compute_theory:
                # Extract scale representations once for all theoretical constraints
                scale_reps = []
                for si, (bg, ed) in enumerate(self.begin_ends):
                    # Only use every other scale to reduce computation
                    if si % 2 == 0 or si == len(self.begin_ends) - 1:  # Include all scales at half resolution + final scale
                        # Get average representation of tokens at this scale
                        scale_rep = logits_BLV[:, bg:ed, :].mean(dim=1)  # [B, V]
                        scale_reps.append(scale_rep)
                
                # Compute unified theoretical loss
                theory_loss = compute_unified_theoretical_loss(scale_reps)
                
                # Use a single weighting factor for theoretical losses
                theory_weight = self.theory_weight  # Use theory_weight as the consolidated weight
                
                # Add to total loss
                if theory_loss > 0:
                    total_loss = total_loss + theory_weight * theory_loss
            else:
                theory_loss = torch.tensor(0.0, device=logits_BLV.device)
        
        # Backward and optimization
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=total_loss, stepping=stepping)
        
        # Handle None gradients
        if grad_norm is None:
            if stepping and dist.get_rank() == 0:
                print(f"[Warning] Gradient is None at step {g_it}.")
            grad_norm = torch.tensor(0.0, device=inp_B3HW.device)
        
        # Logging (with simplified metrics)
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            
            if prog_si >= 0:
                Ltail = acc_tail = -1
            else:
                Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            
            grad_norm_value = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
            
            # Log all losses with simplified metrics
            moe_loss_value = aux_loss.item() if aux_loss is not None else 0.0
            theory_loss_value = theory_loss.item()
            
            metric_lg.update(
                Lm=Lmean, Lt=Ltail, 
                Accm=acc_mean, Acct=acc_tail, 
                tnm=grad_norm_value, 
                MoELoss=moe_loss_value,
                TheoryLoss=theory_loss_value  # Unified theoretical loss
            )
        
        # Log to tensorboard with less frequency
        if g_it == 0 or (g_it + 1) % 500 == 0:
            # Standard logging code...
            # ...
            
            # Use a single metric for theoretical constraints
            if compute_theory:
                tb_lg.update(head='Theory', unified_loss=theory_loss.item(), step=g_it)
        
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2
    
    def get_config(self):
        """Get configuration for checkpointing."""
        config = {
            'patch_nums':   self.patch_nums, 
            'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 
            'last_prog_si': self.last_prog_si, 
            'first_prog':   self.first_prog,
            'aux_loss_weight': self.aux_loss_weight,  # MoE specific
            'theory_weight': self.theory_weight,      # Consolidated parameter
        }
        return config
    
    def state_dict(self):
        """Get state dict for checkpointing."""
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        """Load state dict from checkpoint."""
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[MoEVARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[MoEVARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: Dict = state.pop('config', None)
        if config is not None:
            self.prog_it = config.get('prog_it', 0)
            self.last_prog_si = config.get('last_prog_si', -1)
            self.first_prog = config.get('first_prog', True)
            
            # Load MoE specific config values if available
            if 'aux_loss_weight' in config:
                self.aux_loss_weight = config.get('aux_loss_weight')
            
            # Load consolidated weight for theoretical losses
            if 'theory_weight' in config:
                self.theory_weight = config.get('theory_weight')
            
            # Check for config mismatches
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[HLMoEVAR.load_state_dict] config mismatch: this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)


def compute_unified_theoretical_loss(scale_reps: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute a unified theoretical loss that combines Lyapunov stability, 
    Hölder continuity, and Jacobi field attractivity in a single efficient computation.
    
    Args:
        scale_reps: List of scale representations [B, V] for each scale
        
    Returns:
        Unified theoretical loss
    """
    # Skip if we don't have at least two scales
    if len(scale_reps) < 2:
        return torch.tensor(0.0, device=scale_reps[0].device)
    
    theory_loss = torch.tensor(0.0, device=scale_reps[0].device)
    
    # Hyperparameters (can be moved to class initialization)
    alpha = 0.1  # Lyapunov parameter
    beta = 0.01  # Lyapunov offset
    holder_const = 2.0  # Hölder constant
    gamma = 0.5  # Hölder exponent
    eta = 1.0  # Maximum allowable expansion
    
    # Only compute for a subset of scale transitions to reduce computation
    step = max(1, len(scale_reps) // 4)  # Sample ~25% of transitions
    scale_indices = list(range(0, len(scale_reps) - 1, step))
    if len(scale_reps) - 2 not in scale_indices:
        scale_indices.append(len(scale_reps) - 2)  # Always include last transition
    
    for i in scale_indices:
        current_rep = scale_reps[i]
        next_rep = scale_reps[i + 1]
        batch_size = current_rep.shape[0]
        
        # 1. Compute representation energies (squared norms)
        current_energy = torch.sum(current_rep ** 2, dim=1)  # [B]
        next_energy = torch.sum(next_rep ** 2, dim=1)  # [B]
        
        # 2. Compute representation distance
        rep_distance = torch.norm(next_rep - current_rep, dim=1)  # [B]
        
        # 3. Unified theoretical constraint
        # Combines Lyapunov stability and Hölder continuity
        unified_term = torch.clamp(
            # Lyapunov component: next energy shouldn't exceed current energy too much
            (next_energy - current_energy + alpha * current_energy - beta) / (current_energy.mean() + 1e-6) +
            # Hölder component: distance between representations shouldn't be too large
            (rep_distance - holder_const * (1.0 ** gamma)) / (torch.sqrt(current_energy.mean()) + 1e-6),
            min=0
        )
        theory_loss += unified_term.mean()
        
        # 4. Simplified Jacobi field constraint (only if batch size > 1)
        if batch_size > 1:
            # Sample at most 4 points to reduce computation
            sample_size = min(4, batch_size)
            if batch_size > sample_size:
                indices = torch.randperm(batch_size, device=current_rep.device)[:sample_size]
                z_i = current_rep[indices]
                z_i1 = next_rep[indices]
            else:
                z_i = current_rep
                z_i1 = next_rep
            
            # Compute pairwise distances directly (maximum 6 pairs)
            max_pairs = min(6, sample_size * (sample_size - 1) // 2)
            jacobi_terms = []
            
            # Generate pairs without using nested loops
            pair_count = 0
            for m in range(sample_size - 1):
                for n in range(m + 1, sample_size):
                    if pair_count >= max_pairs:
                        break
                    
                    # Compute distances
                    dist_i = torch.norm(z_i[m] - z_i[n]) + 1e-6
                    dist_i1 = torch.norm(z_i1[m] - z_i1[n])
                    
                    # Calculate contraction/expansion ratio
                    expansion = dist_i1 / dist_i
                    
                    # Penalize expansion beyond threshold
                    jacobi_terms.append(torch.clamp(expansion - eta, min=0))
                    pair_count += 1
                
                if pair_count >= max_pairs:
                    break
            
            # Add Jacobi loss component if we have terms
            if jacobi_terms:
                jacobi_loss = torch.stack(jacobi_terms).mean()
                theory_loss += jacobi_loss
    
    return theory_loss