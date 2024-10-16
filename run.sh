python train_tempobert_ancient.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --train_path datasets/ancient_3 \
    --time_embedding_type temporal_attention \
    --mlm_probability 0.20 \
    --num_time_layers 9 \
    --per_device_train_batch_size 256 \
    --num_train_epochs 5 \
    --warmup_steps 5000 \
    --logging_steps 100 \
    --save_steps 5000 \
    --max_seq_length 128 \
    --do_train \
    --output_dir trained_models \
    --line_by_line \
    --pad_to_max_length    