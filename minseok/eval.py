import torch

from transformers import BertForSequenceClassification, BertTokenizerFast
from bertviz.bertviz import head_view
from tqdm import tqdm, trange

from dataset import MBTIDataset


def predict(model, test_set):
    num_correct = 0
    model.eval()
    with torch.no_grad():
        for post, label in tqdm(test_set, desc="Testing Iteration"):
            logit, attention = model(torch.tensor(post).unsqueeze(0))
            import pdb; pdb.set_trace()
            pred = torch.argmax(logit[0].data)
            if pred.item() == label:
                num_correct += 1
    return num_correct / len(test_set)


if __name__ == "__main__":
    # Load fine-tuned model
    model = BertForSequenceClassification.from_pretrained('./checkpoint', num_labels=16, output_attentions=True)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # kaggle test set
    # test_set = MBTIDataset('./preprocess_new/original_test/mbti.tsv', tokenizer)

    # historical figures
    test_set = MBTIDataset('100speeches.tsv', tokenizer)

    test_accuracy = predict(model, test_set)
    print("Test accuracy: {:06.4f}".format(test_accuracy))