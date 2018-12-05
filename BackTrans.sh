#!/bin/bash

# assuming we've trained our initial models

cat english_train clean_monolingual.en > english_train_cat

for i in {0..4}; do
  python3 translate.py -gpu 0 -model afen_model${i}_step_100000.pt -src Gaussian-af.txt -replace_unk -verbose -output EN_AUGMENT${i}
  cat english_train EN_AUGMENT${i} > english_train_cat${i}
  cat afrikaans_train Gaussian-af.txt > afrikaans_gen_train_cat${i}
  # if u wanna sample add that urself
  python3 preprocess.py -train_src english_train_cat${i} -train_tgt afrikaans_gen_train_cat${i} -valid_src english_valid -valid_tgt afrikaans_valid -src_vocab english_vocab -save_data enaf_aug_${i}
  next=$i+1
  python3 train.py -data enaf_aug_${i} -save_model enaf_model$next -gpu_ranks 0

  python3 translate.py -gpu 0 -model enaf_model${i}_step_100000.pt -src english_mono -replace_unk -verbose -output AF_AUGMENT${i}
  cat afrikaans_train AF_AUGMENT${i} > afrikaans_train_cat${i}
  # if u wanna sample add that urself
  python3 preprocess.py -train_src afrikaans_train_cat${i} -train_tgt english_train_cat  -valid_src afrikaans_valid -valid_tgt english_valid -tgt_vocab english_vocab -save_data afen_aug_${i}
  next=$i+1
  python3 train.py -data afen_aug_${i} -save_model afen_model$next -gpu_ranks 0
done
