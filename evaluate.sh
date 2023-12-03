MODEL_DIR=$1
DATA_DIR=data/vist/processed
MODEL=$2
STYLE=$5
cp $DATA_DIR/test.src-tgt.src.bin $DATA_DIR/test.src-$STYLE\_tgt.src.bin
cp $DATA_DIR/test.src-tgt.src.idx $DATA_DIR/test.src-$STYLE\_tgt.src.idx
cp $DATA_DIR/test.src-tgt.tgt.bin $DATA_DIR/test.src-$STYLE\_tgt.$STYLE\_tgt.bin
cp $DATA_DIR/test.src-tgt.tgt.idx $DATA_DIR/test.src-$STYLE\_tgt.$STYLE\_tgt.idx
cp $DATA_DIR/dict.src.txt $DATA_DIR/dict.$STYLE\_tgt.txt
cp $DATA_DIR/dict.tgt.txt $DATA_DIR/dict.src.txt
CUDA_VISIBLE_DEVICES=$3 fairseq-generate $DATA_DIR \
	--encoder_type transformer \
	--decoder_type transformer \
	--mode_type test \
        --path $MODEL \
	--whether_use_text_clip_fea False \
	--whether_read_text_clip_fea False \
	--whether_clip_encode True \
        --whether_add_memory True \
        --whether_concat True \
	--whether_use_img_fea True \
        --user-dir mass \
        --task translation_mix \
        --model_lang_pairs src-tgt $STYLE\_src-$STYLE\_tgt \
        --lang-pairs src-$STYLE\_tgt \
        --dae-styles  $STYLE \
        --batch-size 256 \
        --skip-invalid-size-inputs-valid-test \
        --beam 5 \
        --lenpen 1.0 \
        --min-len 2 \
        --max-len-b 40 \
        --unkpen 3 \
        --sen_len_constraint 3 \
	--lambda_repeat_penalty_intra 5 \
        --lambda_repeat_penalty_inter $4 \
        --no-repeat-ngram-size 3 \
        2>&1 | tee $MODEL_DIR/output_src_tgt.txt

cp $DATA_DIR/dict.src.txt $DATA_DIR/dict.tgt.txt
cp $DATA_DIR/dict.$STYLE\_tgt.txt $DATA_DIR/dict.src.txt 

grep ^T $MODEL_DIR/output_src_tgt.txt | LC_ALL=C sort -V | cut -f2- | sed 's/ ##//g' > $MODEL_DIR/tgt.txt

cp $DATA_DIR/img_seq.test $MODEL_DIR/img_seq.test
python files_merger.py $MODEL_DIR img_seq.test,hypo_src_tgt.txt,tgt.txt > $MODEL_DIR/output\_src\_F
