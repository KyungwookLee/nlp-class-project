
tsv_file=personal_data.tsv
test_file=personal_data_test.tsv
valid_file=personal_data_valid.tsv
train_file=personal_data_train.tsv




echo "Preprocess data"
python preprocess_gina.py ../data/mbti_1.csv | shuf > $tsv_file

echo "Split data"
head -n 5205 $tsv_file > $train_file
head -n 6940 $tsv_file | tail -n 1735 > $valid_file
tail -n 1735 $tsv_file > $test_file
