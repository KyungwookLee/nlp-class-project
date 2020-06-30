import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizerFast

import utils
from dataset import MBTIDataset, MBTIBatchSampler, mbti_collate_fn
from bertviz.bertviz import head_view

from tqdm import tqdm, trange


def train(model, train_loader, val_loader, optimizer):

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    loss_log = tqdm(total=0, bar_format='{desc}', position=3)
    acc_log = tqdm(total=0, bar_format='{desc}', position=4)

    for epoch in trange(epochs, desc="Epoch"):
        train_loss, train_acc = [], []
        model.train()
        for posts, attention_mask, labels in tqdm(train_loader, desc="Training Iteration"):
            posts = posts.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            loss, logits = model(posts, attention_mask=attention_mask, labels=labels)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            preds = torch.argmax(logits.data, dim=1)
            acc = (preds == labels).sum().item() / len(labels)

            train_loss.append(loss.item())
            train_acc.append(acc)

            des1 = 'Training Loss: {:06.4f}'.format(loss.cpu())
            des2 = 'Training Acc: {:.0%}'.format(acc)
            loss_log.set_description_str(des1)
            acc_log.set_description_str(des2)
            del loss
        
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_acc = sum(train_acc) / len(train_acc)
        print()
        print("Average Training Loss:", avg_train_loss)
        print("Average Training Acc:", avg_train_acc)
        print()
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
    
        val_loss, val_acc = [], []
        model.eval()
        with torch.no_grad():
            for posts, attention_mask, labels in tqdm(val_loader, desc="Validation Iteration"):
                posts = posts.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                loss, logits = model(posts, attention_mask=attention_mask, labels=labels)
                loss = loss.mean()
                pred = torch.argmax(logits.data, dim=1)
                acc = (pred == labels).sum().item() / len(labels)
                
                val_loss.append(loss.item())
                val_acc.append(acc)

                des1 = 'Validation Loss: {:06.4f}'.format(loss.cpu())
                des2 = 'Validation Acc: {:.0%}'.format(acc)
                loss_log.set_description_str(des1)
                acc_log.set_description_str(des2)
                del loss
        
        avg_val_loss = sum(val_loss) / len(val_loss)
        avg_val_acc = sum(val_acc) / len(val_acc)
        print()
        print("Average Validation Loss:", avg_val_loss)
        print("Average Validation Acc:", avg_val_acc)
        print()
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)

    return train_losses, val_losses, train_accuracies, val_accuracies


def predict(model, test_loader):
    acc = []
    model.eval()
    with torch.no_grad():
        for posts, masks, labels in tqdm(test_loader, desc="Testing Iteration"):
            posts, masks, labels = posts.to(device), masks.to(device), labels.to(device)
            logits = model(posts, attention_mask=masks)
            preds = torch.argmax(logits[0], dim=1)
            acc.append((preds == labels).sum().item() / len(labels))
    return sum(acc) / len(acc)


if __name__ == "__main__":
    # Tune hyperparameters here
    batch_size = 16         # 16 or 32          (recommended in BERT paper)
    epochs = 4              # 2, 3, or 4        
    learning_rate = 5e-5    # 5e-5, 3e-5, 2e-5
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on:", device)

    # Load pretrained BERT model
    model_name = "16classes_per_person_bert_aug"
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = nn.DataParallel(model)
    model.to(device)
    
    # Load & store data into memory    
    train_set = MBTIDataset('./preprocess_minseok/per_person_for_bert/mbti_train.tsv', tokenizer)
    val_set = MBTIDataset('./preprocess_minseok/per_person_for_bert/mbti_val.tsv', tokenizer)
    # train_set = MBTIDataset('./gina/personal_data_aug_train.tsv', tokenizer)
    # val_set = MBTIDataset('./gina/personal_data_aug_valid.tsv', tokenizer)

    train_batch_sampler = MBTIBatchSampler(train_set, batch_size=batch_size, shuffle=True)
    val_batch_sampler = MBTIBatchSampler(val_set, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, collate_fn=mbti_collate_fn)
    val_loader = DataLoader(val_set, batch_sampler=val_batch_sampler, collate_fn=mbti_collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader), num_training_steps=len(train_loader)*epochs)

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
    # posts_test, masks_test, labels_test = utils.load_tsv('./preprocess_new/split_test/mbti.tsv', tokenizer)
    # test_set = TensorDataset(posts_test, masks_test, labels_test)
    # test_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=batch_size)
    # model.load_state_dict(torch.load(model_name+"_epoch3.pth"), strict=False)
    # test_accuracy = predict(model, test_loader)
    # print("Test accuracy: {:06.4f}".format(test_accuracy))
