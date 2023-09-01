set -xe
python -m src.train_head --name $1 --epochs 50 \
--dataset nsynth_instrument --accelerator gpu --devices 1 --num_workers 32 --strategy auto \
--precision 32 --lr 0.3 --weight_decay 0.0001
python -m src.train_head --name $1 --epochs 50 \
--dataset nsynth_pitch --accelerator gpu --devices 1 --num_workers 32 --strategy auto \
--precision 32 --lr 0.03 --weight_decay 0.0001
python -m src.train_head --name $1 --epochs 100 \
--dataset mtat --accelerator gpu --devices 1 --num_workers 32 --strategy auto \
--precision 32 --lr 0.8 --weight_decay 0.0001
