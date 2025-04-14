export PYTHONPATH=$PYTHONPATH:/private/yjy/project/MoVEAR

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29500 /private/yjy/project/MoVEAR/movear/models/hlmoetrain.py \
  --depth=36 --saln=1 --pn=512 --bs=100 --ac=1 --ep=100 --tblr=4e-5 --fp16=1 --alng=5e-6 --wpe=0.01 --twde=0.08 \
  --num_experts=4 --k=2 --noise_std=0.1 --aux_weight=0.05 \
  --lyapunov_weight=0.01 --holder_weight=0.01 \
  --data_path="/private/yjy/project/MoVEAR/datasets/imagenet_organized" \
  --local_out_dir_path="/private/yjy/project/MoVEAR/movear/hl_local_output"