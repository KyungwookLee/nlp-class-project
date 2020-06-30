

tsv_file=personal_data_aug.tsv
test_file=personal_data_aug_test.tsv
valid_file=personal_data_aug_valid.tsv
train_file=personal_data_aug_train.tsv

echo "Preprocess data"
python preprocess_gina.py ../data/mbti_1.csv | shuf > $tsv_file

echo "Split data"
head -n 20820 $tsv_file > $train_file
head -n 27760 $tsv_file | tail -n 6940 > $valid_file
tail -n 6940 $tsv_file > $test_file