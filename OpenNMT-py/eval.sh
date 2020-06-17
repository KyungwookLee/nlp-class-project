cut -f1 ../preprocess/mbti_sentence.valid.txt > mbti/mbti_valid.label
num_test=(`wc -l mbti/mbti_valid.label`)
num_diff=(`diff mbti/mbti_valid.label mbti/output | grep "^>" | wc -l`)
echo "Num test: $num_test"
echo "Num diff: $num_diff"
echo "Num correct: $((num_test-num_diff))"
echo "Accuracy:" 
echo "scale=6;($num_test-$num_diff)/$num_test" | bc -l
