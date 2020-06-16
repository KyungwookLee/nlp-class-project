import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, BertConfig

from tqdm import tqdm, trange

import utils


def train(model, train_loader, val_loader, optimizer):

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    loss_log = tqdm(total=0, bar_format='{desc}', position=3)
    acc_log = tqdm(total=0, bar_format='{desc}', position=4)

    for epoch in trange(epochs, desc="Epoch"):
        train_loss, train_acc = [], []
        model.train()
        for posts, masks, labels in tqdm(train_loader, desc="Training Iteration"):
            optimizer.zero_grad()
            posts, masks, labels = posts.to(device), masks.to(device), labels.to(device)
            loss, logits = model(posts, attention_mask=masks, labels=labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits.data, dim=1)
            acc = (preds == labels).sum().item() / len(labels)

            train_loss.append(loss.sum().item())
            train_acc.append(acc)

            des1 = 'Training Loss: {:06.4f}'.format(loss.cpu())
            des2 = 'Training Acc: {:.0%}'.format(acc)
            loss_log.set_description_str(des1)
            acc_log.set_description_str(des2)
            del loss

        train_losses.append(sum(train_loss) / len(train_loss))
        train_accuracies.append(sum(train_acc) / len(train_acc))
    
        val_loss, val_acc = [], []
        model.eval()
        with torch.no_grad():
            for posts, masks, labels in tqdm(val_loader, desc="Validation Iteration"):
                posts, masks, labels = posts.to(device), masks.to(device), labels.to(device)
                loss, logits = model(posts, attention_mask=masks, labels=labels)
                
                preds = torch.argmax(logits.data, dim=1)
                acc = (preds == labels).sum().item() / len(labels)
                
                val_loss.append(loss.item())
                val_acc.append(acc)

                des1 = 'Validation Loss: {:06.4f}'.format(loss.cpu())
                des2 = 'Validation Acc: {:.0%}'.format(acc)
                loss_log.set_description_str(des1)
                acc_log.set_description_str(des2)
                del loss

        val_losses.append(sum(val_loss) / len(val_loss))
        val_accuracies.append(sum(val_acc) / len(val_acc))
    
    return train_losses, val_losses, train_accuracies, val_accuracies


def predict(model, test_loader):
    num_correct = 0
    model.eval()
    with torch.no_grad():
        for posts, masks, labels in tqdm(test_loader, desc="Testing Iteration"):
            posts, masks, labels = posts.to(device), masks.to(device), labels.to(device)
            logits = model(posts, attention_mask=masks)
            preds = torch.argmax(logits.data, dim=1)
            num_correct += (preds == labels).sum().item()
    return num_correct / len(test_loader)


if __name__ == "__main__":
    # Tune hyperparameters here
    batch_size = 32         # 16 or 32          (recommended in BERT paper)
    epochs = 4              # 2, 3, or 4        
    learning_rate = 2e-5    # 5e-5, 3e-5, 2e-5  
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print("Running on:", device)

    # Load pretrained BERT model & tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16, output_attentions=False, output_hidden_states=False)
    model.to(device)
    model_name = "16classes_per_post"
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Load & store data into memory
    posts_train, masks_train, labels_train = utils.load_tsv('./preprocess_new/split_train/mbti.tsv', tokenizer)
    posts_val, masks_val, labels_val = utils.load_tsv('./preprocess_new/split_val/mbti.tsv', tokenizer)
    posts_test, masks_test, labels_test = utils.load_tsv('./preprocess_new/split_test/mbti.tsv', tokenizer)
    
    train_set = TensorDataset(posts_train, masks_train, labels_train)
    val_set = TensorDataset(posts_val, masks_val, labels_val)
    test_set = TensorDataset(posts_test, masks_test, labels_test)

    train_loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size)
    val_loader = DataLoader(val_set, sampler=SequentialSampler(val_set), batch_size=batch_size)
    test_loader = DataLoader(test_set, sampler=SequentialSampler(val_set), batch_size=batch_size)

    # Train & save the model
    train_losses, val_losses, train_accuracies, val_accuracies = train(model, train_loader, val_loader, optimizer)
    torch.save(model.state_dict(), model_name+'.pth')

    # Plot training & validation losses and accuracies
    utils.plot_values(train_losses, val_losses, title=model_name+"_losses")
    utils.plot_values(train_accuracies, val_accuracies, title=model_name+"_accuracies")
    
    print("Final training loss: {:06.4f}".format(train_losses[-1]))
    print("Final validation loss: {:06.4f}".format(val_losses[-1]))
    print("Final training accuracy: {:06.4f}".format(train_accuracies[-1]))
    print("Final validation accuracy: {:06.4f}".format(val_accuracies[-1]))

    # Evaluate on test data
    test_accuracy = predict(model, test_loader)
    print("Test accuracy: {:06.4f}".format(test_accuracy))
