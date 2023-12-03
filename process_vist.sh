#!/usr/bin/env bash
# process CNN_NYT headlines corpora
DATA_DIR=data/vist/raw
OUT_DIR=data/vist/processed

cp $DATA_DIR/objs.train  $DATA_DIR/train.src
cp $DATA_DIR/objs.valid  $DATA_DIR/valid.src
cp $DATA_DIR/objs.test  $DATA_DIR/test.src

cp $DATA_DIR/texts.train  $DATA_DIR/train.tgt
cp $DATA_DIR/texts.valid  $DATA_DIR/valid.tgt
cp $DATA_DIR/texts.test  $DATA_DIR/test.tgt

fairseq-preprocess \
    --user-dir mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref $DATA_DIR/train \
    --validpref $DATA_DIR/valid \
    --testpref $DATA_DIR/test \
    --destdir $OUT_DIR \
    --srcdict data/obj_dict.txt \
    --tgtdict data/story_dict.txt \
    --workers 20
