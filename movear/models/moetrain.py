import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import movear.models.dist as dist
from movear.utils import arg_util, misc
from movear.utils.data import build_dataset
from movear.utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from movear.utils.misc import auto_resume
from movear.utils.amp_sc import AmpOptimizer
from movear.utils.lr_control import filter_params, lr_wd_annealing
from movear.models.vqvae import VQVAE
from movear.models.moevar import MoEVAR
from movear.models.moebuild import build_vae_moe_var, load_pretrained_for_moe
import movear.models.dist as dist
from movear.models.moetrainer import MoEVARTrainer
from tqdm import tqdm
import signal



def build_everything(args: arg_util.Args):
    # Resume from checkpoint if available
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    
    # Create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, 
                                                      filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'), 
                                verbose=True)
        tb_lg.flush()
    else:
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()
    
    # Log arguments
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    # Build data loaders
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        num_classes, dataset_train, dataset_val = build_dataset(
            args.data_path, final_reso=args.data_load_reso, hflip=args.hflip, mid_reso=args.mid_reso,
        )
        types = str((type(dataset_train).__name__, type(dataset_val).__name__))
        val_batch_size = max(4, round(args.batch_size // 4))
        ld_val = DataLoader(
            dataset_val, num_workers=0, pin_memory=True,
            batch_size=val_batch_size, 
            sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
            shuffle=False, drop_last=False,
        )
        del dataset_val
        
        ld_train = DataLoader(
            dataset=dataset_train, 
            num_workers=8,  # 增加至少8个工作线程
            pin_memory=True,
            persistent_workers=True,  # 保持worker进程活跃
            prefetch_factor=2,  # 预加载批次数
            generator=args.get_different_generator_for_each_rank(),
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, 
                same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                shuffle=True, fill_last=True, rank=dist.get_rank(), 
                world_size=dist.get_world_size(), start_ep=start_ep, start_it=start_it,
            ),
        )
        del dataset_train
        
        [print(line) for line in auto_resume_info]
        print(f'[dataloader multi processing] ...', end='', flush=True)
        stt = time.time()
        iters_train = len(ld_train)
        ld_train = iter(ld_train)
        print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True)
        print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')
    else:
        num_classes = 1000
        ld_val = ld_train = None
        iters_train = 10
    
    # Build MoE models
    vae_local, var_wo_ddp = build_vae_moe_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,  # Hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums,
        num_classes=num_classes, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
        num_experts=args.num_experts, k=args.k, noise_std=args.noise_std, 
        aux_loss_weight=args.aux_weight
    )
    
    # Load VAE checkpoint
    vae_ckpt = 'vae_ch160v4096z32.pth'
    if dist.is_local_master():
        if not os.path.exists(vae_ckpt):
            os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
    dist.barrier()
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    
    # Load pretrained VAR weights if specified
    if args.pretrained_var:
        print(f"[MoE] Loading pretrained VAR weights from {args.pretrained_var}")
        var_wo_ddp = load_pretrained_for_moe(var_wo_ddp, args.pretrained_var, device=dist.get_device())
    
    # Compile models if needed
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    var_wo_ddp: MoEVAR = args.compile_model(var_wo_ddp, args.tfast)
    
    # Create DDP model for distributed training
    class NullDDP(torch.nn.Module):
        def __init__(self, module, *args, **kwargs):
            super(NullDDP, self).__init__()
            self.module = module
            self.require_backward_grad_sync = False
        
        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)
    
    var: DDP = (DDP if dist.initialized() else NullDDP)(
        var_wo_ddp, device_ids=[dist.get_local_rank()], 
        find_unused_parameters=True,  # 将False改为True
        broadcast_buffers=False
    )
    
    # Print model information
    print(f'[INIT] MoEVAR model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([
        f'{k}={count_p(m)}' for k, m in (
            ('VAE', vae_local), ('VAE.enc', vae_local.encoder), 
            ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.quantize)
        )
    ]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('MoEVAR', var_wo_ddp),)]) + '\n\n')
    
    # Build optimizer
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })
    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
    
    var_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), 
        names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    # Build MoE trainer
    trainer = MoEVARTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var,
        var_opt=var_optim, label_smooth=args.ls,
    )
    
    # Load trainer state if available
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True)  # Don't load vae again
    
    del vae_local, var_wo_ddp, var, var_optim
    
    # Debug mode for local testing
    if args.local_debug:
        rng = torch.Generator('cpu')
        rng.manual_seed(0)
        B = 4
        inp = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        label = torch.ones(B, dtype=torch.long)
        
        me = misc.MetricLogger(delimiter='  ')
        trainer.train_step(
            it=0, g_it=0, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=args.pg0, prog_wp_it=20,
        )
        
        trainer.load_state_dict(trainer.state_dict())
        trainer.train_step(
            it=99, g_it=599, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=-1, prog_wp_it=20,
        )
        print({k: meter.global_avg for k, meter in me.meters.items()})
        
        args.dump_log(); tb_lg.flush(); tb_lg.close()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
        exit(0)
    
    dist.barrier()
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val
    )


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, 
                 tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer):
    trainer: MoEVARTrainer
    
    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # Add MoE-specific meter for auxiliary loss
    me.add_meter('MoELoss', misc.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    # 创建主进程的进度条
    pbar = None
    if dist.is_master():
        pbar = tqdm(
            total=iters_train-start_it, 
            desc=f"Epoch {ep}/{args.ep}", 
            position=0, 
            leave=True,
            ncols=100
        )
    
    # 修改训练循环以更新进度条
    for it, (inp, label) in me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        inp = inp.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)
        
        args.cur_it = f'{it+1}/{iters_train}'
        
        # Learning rate and weight decay scheduling
        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
            args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, 
            g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe
        )
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        
        # Progressive training setup
        if args.pg:  # Default: args.pg == 0.0, means no progressive training
            if g_it <= wp_it: 
                prog_si = args.pg0
            elif g_it >= max_it*args.pg: 
                prog_si = len(args.patch_nums) - 1
            else:
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(max((g_it - wp_it) / (max_it*args.pg - wp_it), 0), 1)  # From 0 to 1
                prog_si = args.pg0 + round(progress * delta)  # From args.pg0 to len(args.patch_nums)-1
        else:
            prog_si = -1
        
        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)
        
        # Train one step using MoEVARTrainer
        grad_norm, scale_log2 = trainer.train_step(
            it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
        )
        
        # Update learning rate and other metrics in tensorboard
        me.update(tlr=max_tlr)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
        tb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        tb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)
        
        # Log gradient norm if clipping is enabled
        if args.tclip > 0:
            tb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
            tb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip)
        
        # MoE-specific logging
        if 'MoELoss' in me.meters:
            moe_loss = me.meters['MoELoss'].global_avg
            tb_lg.update(head='MoE/aux_loss', aux_loss=moe_loss, step=g_it)
            tb_lg.update(head='MoE/aux_weight', aux_weight=args.aux_weight, step=g_it)
        
        # 更新进度条
        if dist.is_master() and pbar is not None:
            pbar.update(1)
            # 显示关键指标
            moe_loss = me.meters.get('MoELoss', misc.SmoothedValue()).median if 'MoELoss' in me.meters else 0.0
            pbar.set_postfix({
                'loss': f"{me.meters['Lm'].median:.4f}",
                'acc': f"{me.meters['Accm'].median:.1f}%",
                'lr': f"{max_tlr:.6f}",
                'MoE': f"{moe_loss:.5f}"
            })
    
    # 关闭进度条
    if dist.is_master() and pbar is not None:
        pbar.close()
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)


def main_training():
    
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
    
    os.makedirs(args.local_out_dir_path, exist_ok=True)
    print(f"[DEBUG] Output directory: {args.local_out_dir_path}")
    
    # Build everything needed for training
    (
        tb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val
    ) = build_everything(args)
    
  
    
    # Training loop
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1
    
    L_mean, L_tail = -1, -1
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True)
        tb_lg.set_step(ep * iters_train)
        
        # Train one epoch
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, 
            args, tb_lg, ld_train, iters_train, trainer
        )
        
        # Extract and update metrics
        L_mean, L_tail = stats['Lm'], stats['Lt']
        acc_mean, acc_tail = stats['Accm'], stats['Acct']
        grad_norm = stats['tnm']
        moe_loss = stats.get('MoELoss', 0.0)
        
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1: 
            best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        
        # Log epoch metrics
        AR_ep_loss = dict(
            L_mean=L_mean, L_tail=L_tail, 
            acc_mean=acc_mean, acc_tail=acc_tail,
            moe_loss=moe_loss
        )
        
        # Validation and checkpoint saving
        is_val_and_also_saving = (ep + 1) % 5 == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.evaluate(ld_val)
            
            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean = min(best_val_loss_mean, val_loss_mean)
            best_val_loss_tail = min(best_val_loss_tail, val_loss_tail)
            best_val_acc_mean = max(best_val_acc_mean, val_acc_mean) 
            best_val_acc_tail = max(best_val_acc_tail, val_acc_tail)
            
            AR_ep_loss.update(
                vL_mean=val_loss_mean, vL_tail=val_loss_tail, 
                vacc_mean=val_acc_mean, vacc_tail=val_acc_tail
            )
            
            args.vL_mean = val_loss_mean
            args.vL_tail = val_loss_tail
            args.vacc_mean = val_acc_mean
            args.vacc_tail = val_acc_tail
            
            print(f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, '
                  f'Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  '
                  f'MoE loss: {moe_loss:.5f},  Val cost: {cost:.2f}s')
            
            # Save checkpoint if local master
            if dist.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
                local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                
                print(f'[saving ckpt] ...', end='', flush=True)
                try:
                    tmp_ckpt = local_out_ckpt + '.tmp'
                    torch.save({
                        'epoch': ep+1,
                        'iter': 0,
                        'trainer': trainer.state_dict(),
                        'args': args.state_dict(),
                    }, tmp_ckpt)
                    if os.path.exists(local_out_ckpt):
                        os.rename(local_out_ckpt, local_out_ckpt + '.bak')
                    os.rename(tmp_ckpt, local_out_ckpt)
                    print(f"     [saving ckpt](*) finished! @ {local_out_ckpt}", flush=True)
                except Exception as e:
                    print(f"     [saving ckpt] ERROR: {str(e)}", flush=True)
                
                if best_updated:
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
                
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True)
            
            dist.barrier()
        
        # Print epoch summary
        print(f'     [ep{ep}]  (training)  Lm: {best_L_mean:.3f} ({L_mean:.3f}), '
              f'Lt: {best_L_tail:.3f} ({L_tail:.3f}),  '
              f'Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  '
              f'MoE loss: {moe_loss:.5f},  '
              f'Remain: {remain_time},  Finish: {finish_time}', flush=True)
        
        # Update tensorboard logs
        tb_lg.update(head='AR_ep_loss', step=ep+1, **AR_ep_loss)
        tb_lg.update(head='AR_z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))
        tb_lg.update(head='MoE/ep_aux_loss', aux_loss=moe_loss, step=ep+1)
        
        args.dump_log(); tb_lg.flush()
    
    # Training finished
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [MoE PT finished]  Total cost: {total_time},   '
          f'Lm: {best_L_mean:.3f} ({L_mean}),   '
          f'Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    # Clean up
    del stats
    del iters_train, ld_train
    time.sleep(3)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log()
    tb_lg.flush()
    tb_lg.close()
    dist.barrier()


if __name__ == '__main__':
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close()
            sys.stderr.close()