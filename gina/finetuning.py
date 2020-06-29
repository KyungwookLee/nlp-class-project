import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch.optim as optim
from tqdm import tqdm, trange
import data
import sklearn.metrics as metrics

def tokenloader(tokenizer, batch_size):
    label_train, text_train, mask_train = data.dataloader('./personal_data_aug_train.tsv', tokenizer)
    label_val, text_val, mask_val = data.dataloader('./personal_data_aug_valid.tsv', tokenizer)
    label_test, text_test, mask_test = data.dataloader('./personal_data_aug_test.tsv', tokenizer)
    
    train_set = TensorDataset(label_train, text_train, mask_train)
    val_set = TensorDataset(label_val, text_val, mask_val)
    test_set = TensorDataset(label_test, text_test, mask_test)

    train_data = DataLoader(train_set, batch_size=batch_size)
    val_data = DataLoader(val_set, batch_size=batch_size)
    test_data = DataLoader(test_set, batch_size=batch_size)

    return train_data, val_data, test_data

def training(model, train_data, val_data):
    epochs = 10        
    learning_rate = 5e-5  
    device = torch.device('cuda:5')
    batch_size = 16
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    bal_pred, bal_ans = [], []
    for epoch in trange(epochs, desc="Epoch"):
        tl, ta, vl, va = 0, 0, 0, 0
        model.train()
        model.to(device)
        for label, text, mask in tqdm(train_data, desc="Training Iteration"):
            label, text, mask = label.to(device), text.to(device), mask.to(device)
            optimizer.zero_grad()
            loss, logits = model(text, attention_mask=mask, labels=label)[:2]
            loss.backward()
            optimizer.step()
            tl+=loss
            del loss
            preds = torch.argmax(logits.data, dim=1)
            ta+=float(sum(preds==label))/batch_size
        train_losses.append(float(tl/len(train_data)))
        train_accuracies.append(float(ta/len(train_data)))
        print('avg_t_loss=',tl/len(train_data),'avg_t_acc=',ta/len(train_data))

        with torch.no_grad():
          model.eval()
          for label, text, mask in tqdm(val_data, desc="Validation Iteration"):
              label, text, mask = label.to(device), text.to(device), mask.to(device)
              loss, logits = model(text, attention_mask=mask, labels=label)
              preds = torch.argmax(logits.data, dim=1)
            #   acc = (preds == label).sum().item() / len(label) 
              vl+=loss
              va+=float(sum(preds==label))/batch_size
              bal_pred.append(preds)
              bal_ans.append(label)
            #   va+=acc
              del loss
        val_losses.append(float(vl/len(val_data)))
        val_accuracies.append(float(va/len(val_data)))
        bal_score = metrics.balanced_accuracy_score(label,preds)
        f1_score = metrics.f1_score(label,preds, average='macro')
        print('avg_v_loss=',vl/len(val_data),'avg_v_acc=',va/len(val_data),'bal_score=',bal_score,'f1_score=',f1_score)
    
    return train_losses, val_losses, train_accuracies, val_accuracies


def testing(model, test_data):
    num_correct = 0
    model.eval()
    with torch.no_grad():
        for label, text, mask in tqdm(test_data, desc="Testing Iteration"):
            label, text, mask = label.to(device), text.to(device), mask.to(device)
            logits = model(text, attention_mask=mask)
            preds = torch.argmax(logits.data, dim=1)
            num_correct += (preds == label).sum().item()
    return num_correct / len(test_data)


def train_test_model():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_data, val_data, test_data = tokenloader(tokenizer, batch_size=16)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16)
    model_name = "classification_by_person_aug"
    train_losses, val_losses, train_accuracies, val_accuracies = training(model,train_data,val_data)

    torch.save(model.state_dict(), model_name+'.pth')
    
    print("Final training loss: {:06.4f}".format(train_losses))
    print("Final validation loss: {:06.4f}".format(val_losses))
    print("Final training accuracy: {:06.4f}".format(train_accuracies))
    print("Final validation accuracy: {:06.4f}".format(val_accuracies))

    test_accuracy = testing(model, test_data)
    print("Test accuracy: {:06.4f}".format(test_accuracy))

if __name__ == "__main__":
    train_test_model()