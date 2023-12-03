#!/usr/bin/env bash
# process CNN_NYT headlines corpora

DATA_DIR=data/fairy/raw
OUT_DIR=data/fairy/processed

cp $DATA_DIR/texts.train  $DATA_DIR/train.fairy_src
cp $DATA_DIR/texts.valid  $DATA_DIR/valid.fairy_src
cp $DATA_DIR/texts.valid  $DATA_DIR/test.fairy_src
cp $DATA_DIR/texts.train  $DATA_DIR/train.fairy_tgt
cp $DATA_DIR/texts.valid  $DATA_DIR/valid.fairy_tgt
cp $DATA_DIR/texts.valid  $DATA_DIR/test.fairy_tgt


fairseq-preprocess \
    --user-dir mass --task masked_s2s \
    --source-lang fairy_src --target-lang fairy_tgt \
    --trainpref $DATA_DIR/train \
    --validpref $DATA_DIR/valid \
    --testpref $DATA_DIR/test \
    --destdir $OUT_DIR \
    --srcdict data/story_dict.txt \
    --tgtdict data/story_dict.txt \
    --workers 20

DATA_DIR=data/romance/raw
OUT_DIR=data/romance/processed

cp $DATA_DIR/texts.train  $DATA_DIR/train.romance_src
cp $DATA_DIR/texts.valid  $DATA_DIR/valid.romance_src
cp $DATA_DIR/texts.valid  $DATA_DIR/test.romance_src
cp $DATA_DIR/texts.train  $DATA_DIR/train.romance_tgt
cp $DATA_DIR/texts.valid  $DATA_DIR/valid.romance_tgt
cp $DATA_DIR/texts.valid  $DATA_DIR/test.romance_tgt


fairseq-preprocess \
    --user-dir mass --task masked_s2s \
    --source-lang romance_src --target-lang romance_tgt \
    --trainpref $DATA_DIR/train \
    --validpref $DATA_DIR/valid \
    --testpref $DATA_DIR/test \
    --destdir $OUT_DIR \
    --srcdict data/story_dict.txt \
    --tgtdict data/story_dict.txt \
    --workers 20

DATA_DIR=data/humor/raw
OUT_DIR=data/humor/processed

cp $DATA_DIR/texts.train  $DATA_DIR/train.humor_src
cp $DATA_DIR/texts.valid  $DATA_DIR/valid.humor_src
cp $DATA_DIR/texts.valid  $DATA_DIR/test.humor_src
cp $DATA_DIR/texts.train  $DATA_DIR/train.humor_tgt
cp $DATA_DIR/texts.valid  $DATA_DIR/valid.humor_tgt
cp $DATA_DIR/texts.valid  $DATA_DIR/test.humor_tgt


fairseq-preprocess \
    --user-dir mass --task masked_s2s \
    --source-lang humor_src --target-lang humor_tgt \
    --trainpref $DATA_DIR/train \
    --validpref $DATA_DIR/valid \
    --testpref $DATA_DIR/test \
    --destdir $OUT_DIR \
    --srcdict data/story_dict.txt \
    --tgtdict data/story_dict.txt \
    --workers 20
