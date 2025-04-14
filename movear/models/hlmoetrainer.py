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
        lyapunov_weight: float = 0.01, holder_weight: float = 0.01,
    ):
        super(HLMoEVARTrainer, self).__init__()
        
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
        self.lyapunov_weight = lyapunov_weight
        self.holder_weight = holder_weight
    
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
        """
        Train the model for one step with additional theoretical guarantees.
        
        Added MoE-specific functionality to handle auxiliary loss from expert routing
        and theoretical losses for Lyapunov stability and Hölder continuity.
        """
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
            
            # Calculate Lyapunov stability loss
            lyapunov_loss = torch.tensor(0.0, device=logits_BLV.device)
            
            # Extract scale representations for Lyapunov and Hölder loss calculation
            scale_reps = []
            for si, (bg, ed) in enumerate(self.begin_ends):
                if prog_si >= 0 and si > prog_si:
                    break  # Only use scales up to current progress in progressive training
                    
                # Get average representation of tokens at this scale
                scale_rep = logits_BLV[:, bg:ed, :].mean(dim=1)  # [B, V]
                scale_reps.append(scale_rep)
            
            # Calculate Lyapunov stability loss if we have multiple scales
            if len(scale_reps) > 1:
                alpha = 0.1  # Hyperparameter
                beta = 0.01  # Hyperparameter
                
                for i in range(len(scale_reps) - 1):
                    energy_i = torch.norm(scale_reps[i], dim=1) ** 2
                    energy_i_plus_1 = torch.norm(scale_reps[i+1], dim=1) ** 2
                    
                    stability_term = torch.clamp(
                        energy_i_plus_1 - energy_i + alpha * energy_i - beta, 
                        min=0
                    )
                    lyapunov_loss += stability_term.mean()
            
            # Calculate Hölder continuity loss
            holder_loss = torch.tensor(0.0, device=logits_BLV.device)
            
            if len(scale_reps) > 1:
                holder_constant = 2.0  # Hyperparameter
                holder_exponent = 0.5  # Gamma parameter (0 < gamma <= 1)
                
                for i in range(len(scale_reps) - 1):
                    # Calculate distance between consecutive scale representations
                    dist = torch.norm(scale_reps[i+1] - scale_reps[i], dim=1)
                    
                    # Apply Hölder constraint: |f(s_i+1) - f(s_i)| <= H * |i+1 - i|^gamma
                    scale_diff = 1.0 ** holder_exponent  # |i+1 - i|^gamma = 1^gamma = 1
                    holder_term = torch.clamp(dist - holder_constant * scale_diff, min=0)
                    holder_loss += holder_term.mean()
            
            # Add theoretical losses to total loss
            if len(scale_reps) > 1:
                total_loss = total_loss + self.lyapunov_weight * lyapunov_loss + self.holder_weight * holder_loss
            
            # Add Jacobi loss
            jacobi_loss = torch.tensor(0.0, device=logits_BLV.device)

            if len(scale_reps) > 1:
                eta = 1.0  # Maximum allowed expansion factor (<=1 enforces contraction)
                epsilon = 1e-6  # Small constant for numerical stability
                
                for i in range(len(scale_reps) - 1):
                    batch_size = scale_reps[i].shape[0]
                    
                    # Only calculate if batch_size > 1
                    if batch_size > 1:
                        # Compute pairwise distances at scale i
                        z_i_expand = scale_reps[i].unsqueeze(1)  # [B, 1, D]
                        z_i_transpose = scale_reps[i].unsqueeze(0)  # [1, B, D]
                        dist_i = torch.norm(z_i_expand - z_i_transpose, dim=2)  # [B, B]
                        
                        # Compute pairwise distances at scale i+1
                        z_i1_expand = scale_reps[i+1].unsqueeze(1)
                        z_i1_transpose = scale_reps[i+1].unsqueeze(0)
                        dist_i1 = torch.norm(z_i1_expand - z_i1_transpose, dim=2)
                        
                        # Compute expansion ratio (masked to exclude diagonal)
                        mask = ~torch.eye(batch_size, dtype=torch.bool, device=dist_i.device)
                        expansion_ratio = dist_i1[mask] / (dist_i[mask] + epsilon)
                        
                        # Penalize expansion beyond eta
                        jacobi_term = torch.clamp(expansion_ratio - eta, min=0).mean()
                        jacobi_loss += jacobi_term

            # Add to total loss
            if len(scale_reps) > 1:
                total_loss = total_loss + self.lyapunov_weight * lyapunov_loss + \
                             self.holder_weight * holder_loss + \
                             self.lyapunov_weight * jacobi_loss  # Use lyapunov_weight for jacobi as well
        
        # Backward and optimization
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=total_loss, stepping=stepping)
        
        # Handle None gradients (same as original)
        if grad_norm is None:
            if stepping and dist.get_rank() == 0:
                print(f"[MoE Warning] Gradient is None at step {g_it}. This may occur in first iterations or if no experts were selected on this GPU.")
            grad_norm = torch.tensor(0.0, device=inp_B3HW.device)
        
        # Logging (similar to original but with additional theoretical losses)
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
            
            # Log all losses
            moe_loss_value = aux_loss.item() if aux_loss is not None else 0.0
            lyapunov_loss_value = lyapunov_loss.item()
            holder_loss_value = holder_loss.item()
            jacobi_loss_value = jacobi_loss.item()
            
            metric_lg.update(
                Lm=Lmean, Lt=Ltail, 
                Accm=acc_mean, Acct=acc_tail, 
                tnm=grad_norm_value, 
                MoELoss=moe_loss_value,
                LyapunovLoss=lyapunov_loss_value,
                HolderLoss=holder_loss_value,
                JacobiLoss=jacobi_loss_value
            )
        
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
                
                kw = dict(z_voc_usage=cluster_usage)
                if aux_loss is not None:
                    kw['moe_aux_loss'] = aux_loss.item()
                
                # Add theoretical losses to tensorboard
                kw['lyapunov_loss'] = lyapunov_loss.item()
                kw['holder_loss'] = holder_loss.item()
                kw['jacobi_loss'] = jacobi_loss.item()
                
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)
        
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
            'lyapunov_weight': self.lyapunov_weight,  # Hyperparameter
            'holder_weight': self.holder_weight,      # Hyperparameter
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
            
            # Load hyperparameters for theoretical losses
            if 'lyapunov_weight' in config:
                self.lyapunov_weight = config.get('lyapunov_weight')
            if 'holder_weight' in config:
                self.holder_weight = config.get('holder_weight')
            
            # Check for config mismatches
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[HLMoEVAR.load_state_dict] config mismatch: this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)