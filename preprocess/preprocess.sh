tsv_file=mbti_sentence.tsv
test_file=mbti_sentence.test.txt
valid_file=mbti_sentence.valid.txt
train_file=mbti_sentence.train.txt

echo "Preprocessing data"
python preprocess.py ../data/mbti_1.csv | shuf > $tsv_file
echo "Splitting data"
head -n 63427 $tsv_file > $test_file
head -n 126854 $tsv_file | tail -n 63427 > $valid_file
tail -n 296000 $tsv_file > $train_file
