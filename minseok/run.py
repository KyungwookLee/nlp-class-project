import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3, 6'
import random
from collections import defaultdict

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
    bal_accuracies, f1_scores = [], []
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
        num_pred_per_label = defaultdict(int)
        num_correct_per_label = defaultdict(int)
        num_total_per_label = defaultdict(int)
        model.eval()
        with torch.no_grad():
            for posts, attention_mask, labels in tqdm(val_loader, desc="Validation Iteration"):
                posts = posts.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                loss, logits = model(posts, attention_mask=attention_mask, labels=labels)
                loss = loss.mean()
                preds = torch.argmax(logits.data, dim=1)
                acc = (preds == labels).sum().item() / len(labels)
                
                val_loss.append(loss.item())
                val_acc.append(acc)

                # for balanced accuracy and f1-score
                for i in range(len(labels)):
                    label = utils.decode_label(labels[i].item())
                    pred = utils.decode_label(preds[i].item())
                    if preds[i] == labels[i]:
                        num_correct_per_label[label] += 1   # TP
                    num_pred_per_label[pred] += 1           # TP+FP
                    num_total_per_label[label] += 1         # TP+FN
                
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

        # balanced accuracy and f1-scores
        bal_acc = {}
        prf1 = {}
        for i in range(16):
            label = utils.decode_label(i)
            # catch div by zero
            P = num_correct_per_label[label] / num_pred_per_label[label] if num_pred_per_label[label] != 0 else 0.
            R = num_correct_per_label[label] / num_total_per_label[label] if num_total_per_label[label] != 0 else 0.
            F1 = 2*(P*R)/(P+R) if P != 0. and R != 0. else 0. 
            prf1[label] = [P, R, F1]
            bal_acc[label] = [R, num_correct_per_label[label], num_total_per_label[label]]
        f1_scores.append(prf1)
        bal_accuracies.append(bal_acc)

    return train_losses, val_losses, train_accuracies, val_accuracies, f1_scores, bal_accuracies


if __name__ == "__main__":
    # Reproduce same results
    random.seed(123)
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Tune hyperparameters here
    batch_size = 16         # 16 or 32          (recommended in BERT paper)
    epochs = 4              # 2, 3, or 4        
    learning_rate = 2e-5    # 5e-5, 3e-5, 2e-5
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print("Running on:", device)

    # Load pretrained BERT model
    model_name = "16classes_per_person_bert_bs16_lr2e-5"
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # model = nn.DataParallel(model)
    model.to(device)
    
    # Load & store data into memory    
    train_set = MBTIDataset('./preprocess_new/original_train/mbti.tsv', tokenizer)
    val_set = MBTIDataset('./preprocess_new/original_val/mbti.tsv', tokenizer)
    # train_set = MBTIDataset('./gina/personal_data_aug_train.tsv', tokenizer)
    # val_set = MBTIDataset('./gina/personal_data_aug_valid.tsv', tokenizer)

    train_batch_sampler = MBTIBatchSampler(train_set, batch_size=batch_size, shuffle=True)
    val_batch_sampler = MBTIBatchSampler(val_set, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, collate_fn=mbti_collate_fn)
    val_loader = DataLoader(val_set, batch_sampler=val_batch_sampler, collate_fn=mbti_collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader), num_training_steps=len(train_loader)*epochs)

    # Train & save the model
    train_losses, val_losses, train_accuracies, val_accuracies, f1_scores, bal_accuracies = train(model, train_loader, val_loader, optimizer)
    model.save_pretrained('./checkpoint')
    # torch.save(model.state_dict(), model_name+'.pth')

    with open(model_name+"_losses.txt", "w") as f:
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            f.write(f"Epoch: {epoch+1}\n")
            f.write(f"Train Loss: {train_loss}, Val Loss: {val_loss}\n")
            f.write(f"\n")
    
    with open(model_name+"_acc.txt", "w") as f:
        for epoch, (train_acc, val_acc) in enumerate(zip(train_accuracies, val_accuracies)):
            f.write(f"Epoch: {epoch+1}\n")
            f.write(f"Train Loss: {train_acc}, Val Loss: {val_acc}\n")
            f.write(f"\n")
   
    with open(model_name+"_bal_acc.txt", "w") as f:
        for epoch, bal_acc in enumerate(bal_accuracies):
            f.write(f"Epoch: {epoch+1}\n")
            sum_acc = 0.
            for label, (acc, cor, tot) in bal_acc.items():
                f.write(f"{label}: {acc} ({cor}/{tot})\n")
                sum_acc += acc
            f.write(f"Balanced Accuracy: {sum_acc / 16}\n")
            f.write(f"\n")
    
    with open(model_name+"_f1.txt", "w") as f:
        for epoch, f1_score in enumerate(f1_scores):
            f.write(f"Epoch: {epoch+1}\n")
            sum_f1 = 0.
            for label, (P, R, F1) in f1_score.items():
                f.write(f"{label}: P={P}, R={R}, F1={F1}\n")
                sum_f1 += F1
            f.write(f"F1_macro: {sum_f1 / 16}\n")
            f.write(f"\n")

    # Plot training & validation losses and accuracies
    utils.plot_values(train_losses, val_losses, title=model_name+"_losses")
    utils.plot_values(train_accuracies, val_accuracies, title=model_name+"_accuracies")
