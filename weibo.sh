CUDA_VISIBLE_DEVICES=1 python weibo_train.py \
--output_file Data/weibo/output_image_text/ \
--hidden_dim 32 \
--dataset weibo \
--num_epochs 100 