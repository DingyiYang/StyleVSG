SUMM_DIR=data/vist/processed
SAVE_DIR=$1
STYLE=$2
DAE_DIR=data/$STYLE/processed
#STYLE_PAIR="{}_src-{}_tgt"
bz=64


CUDA_VISIBLE_DEVICES=1,2 python train.py \
    $SUMM_DIR:$DAE_DIR \
    --pic_token_num 5 \
    --whether_clip_encode True \
    --mode_type train \
    --encoder_type transformer \
    --decoder_type transformer \
    --whether_concat True \
    --ddp-backend=no_c10d \
    --device-id 0 \
    --distributed-world-size 2 \
    --num-workers 0 \
    --whether_use_img_fea True \
    --whether_add_memory True \
    --encoder-embed-dim 512 --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 8 --decoder-attention-heads 8 \
    --user-dir mass --task translation_mix --arch transformer_mix_base \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --min-lr 1e-09 \
    --seed 1000 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --weight-decay 0.0 \
    --criterion model_loss  --label-smoothing 0.1 \
    --update-freq 4 --max-sentences $bz  \
    --whether_read_text_clip_fea True \
    --whether_use_text_clip_fea False \
    --ddp-backend=no_c10d --max-epoch 120 \
    --max-source-positions 200 --max-target-positions 200 \
    --skip-invalid-size-inputs-valid-test \
    --dropout 0.2 \
    --model_lang_pairs src-tgt $STYLE\_src-$STYLE\_tgt --lang-pairs src-tgt --dae-styles $STYLE \
    --lambda-parallel-config 0.5 --lambda-denoising-config 0.5 \
    --max-word-shuffle-distance 5 \
    --word-dropout-prob 0.3 \
    --word-blanking-prob 0.2 \
    --divide-decoder-self-attn-norm True \
    --divide-decoder-final-norm True \
    --divide-decoder-encoder-attn-query True \
    --save-dir $SAVE_DIR 
    2>&1 | tee $SAVE_DIR/train_log.txt
