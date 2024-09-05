mkdir -p mbti
mkdir -p mbti/model
cut -f1 ../preprocess/mbti_sentence.train.txt > mbti/train.tgt
cut -f2 ../preprocess/mbti_sentence.train.txt > mbti/train.src
cut -f1 ../preprocess/mbti_sentence.valid.txt > mbti/valid.tgt
cut -f2 ../preprocess/mbti_sentence.valid.txt > mbti/valid.src
python preprocess.py --train_src mbti/train.src --train_tgt mbti/train.tgt --valid_src mbti/valid.src --valid_tgt mbti/valid.tgt --save_data mbti/mbti_data
