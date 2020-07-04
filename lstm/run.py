from typing import List
from dataset import Language, NmtDataset, bucketed_batch_indices, collate_fn
from model import Seq2Seq

import torch
import pytorch_warmup as warmup

import random
from tqdm import tqdm, trange
from collections import defaultdict, Counter

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

task = 'person_aug'  # post: 'post', person: 'person', person-augmented: 'person_aug'
attention_type = 'dot' # 'dot' or 'concat'
embedding_dim = 128
hidden_dim = 64
bucketing = True
lr = 5e-5
weight_decay = 0#0.0001
lmbda = lambda epoch: 0.98
early_num = 10
max_len = 512  # post: 100, person: 1024, person-augmented: 512

def plot_attention(attention: torch.Tensor, trg_text: List[str], src_text: List[str], name: str):
    assert attention.shape[0] == len(trg_text) and attention.shape[1] == len(src_text)
    _, ax = plt.subplots()
    _ = ax.pcolor(attention)

    ax.set_xticks([tick + .5 for tick in range(len(src_text))], minor=False)
    ax.set_yticks([tick + .5 for tick in range(len(trg_text))], minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(src_text, rotation=90, minor=False)
    ax.set_yticklabels(trg_text, minor=False)
    plt.savefig('attention_' + name + '.png')

def train():
    max_epoch = 1
    batch_size = 32 # post: 64, person: 32, person-augmented: 32

    train_posts = Language(path='lstm.data/' + task + '_train.input')
    train_labels = Language(path='lstm.data/' + task + '_train.label')
    train_posts.build_vocab()
    train_labels.build_vocab()
    train_dataset = NmtDataset(src=train_posts, trg=train_labels, max_len=max_len)

    val_posts = Language(path='lstm.data/' + task + '_val.input')
    val_labels = Language(path='lstm.data/' + task + '_val.label')
    val_posts.set_vocab(train_posts.word2idx, train_posts.idx2word)
    val_labels.set_vocab(train_labels.word2idx, train_labels.idx2word)
    val_dataset = NmtDataset(src=val_posts, trg=val_labels, max_len=max_len)

    max_pad_len = 5
    train_sentence_length = list(map(lambda pair: len(pair[0]), train_dataset))
    train_batch_sampler = bucketed_batch_indices(train_sentence_length, batch_size=batch_size, max_pad_len=max_pad_len) if bucketing else None
    val_sentence_length = list(map(lambda pair: len(pair[0]), val_dataset))
    val_batch_sampler = bucketed_batch_indices(val_sentence_length, batch_size=batch_size, max_pad_len=max_pad_len) if bucketing else None

    model = Seq2Seq(train_posts, train_labels, attention_type=attention_type,
                    embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    #warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=10)
    
    train_dataloader = torch.utils.data.dataloader.DataLoader(train_dataset, collate_fn=collate_fn, num_workers=2, batch_size=1 if bucketing else batch_size, batch_sampler=train_batch_sampler, shuffle=not bucketing)
    val_dataloader = torch.utils.data.dataloader.DataLoader(val_dataset, collate_fn=collate_fn, num_workers=2, batch_size=1 if bucketing else batch_size, batch_sampler=val_batch_sampler, shuffle=not bucketing)
    
    #train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    best_acc, best_loss, early_stop = 0., float('inf'), 0
    for epoch in trange(max_epoch, desc="Epoch", position=0):
        model.train()
        train_epoch_losses, train_epoch_accuracies = 0., 0.
        train_epoch_acc1, train_epoch_acc2, train_epoch_acc3, train_epoch_acc4 = 0., 0., 0., 0.
        for train_posts, train_labels in tqdm(train_dataloader, desc="Iteration", position=1):
            #print(train_labels)
            optimizer.zero_grad()
            train_posts, train_labels = train_posts.to(device), train_labels.to(device)
            train_loss, train_output = model(train_posts, train_labels, teacher_force=0.5)
            train_loss.backward()
            optimizer.step()
            
            train_accuracy = float((train_output[:,0] == train_labels[:,1]).sum()) / batch_size
            #train_acc1 = float((train_output[:,0] == train_labels[:,1]).sum()) / batch_size
            #train_acc2 = float((train_output[:,1] == train_labels[:,2]).sum()) / batch_size
            #train_acc3 = float((train_output[:,2] == train_labels[:,3]).sum()) / batch_size
            #train_acc4 = float((train_output[:,3] == train_labels[:,4]).sum()) / batch_size
            #train_accuracy = float(torch.all(torch.eq(train_output, train_labels[:,1:-1]), dim=1).sum()) / batch_size
            
            train_epoch_losses += train_loss.item()
            train_epoch_accuracies += train_accuracy
            #train_epoch_acc1 += train_acc1
            #train_epoch_acc2 += train_acc2
            #train_epoch_acc3 += train_acc3
            #train_epoch_acc4 += train_acc4

            del train_loss, train_output
        
        train_last_loss = train_epoch_losses / len(train_dataloader)
        train_last_accuracy = train_epoch_accuracies / len(train_dataloader)

        #train_losses.append(train_last_loss)
        #train_accuracies.append(train_last_accuracy)

        print('Train Loss: {:06.6f}, accuracy: {:06.6f}'.format(train_last_loss, train_last_accuracy))
        #print('Train Loss: {:06.6f}, accuracy: {:06.6f}, 1: {:06.6f}, 2: {:06.6f}, 3: {:06.6f}, 4: {:06.6f}'.format(train_last_loss, train_last_accuracy, train_last_acc1, train_last_acc2, train_last_acc3, train_last_acc4))
        
        with torch.no_grad():
            model.eval()
            val_epoch_losses, val_epoch_accuracies = 0., 0.
            val_epoch_acc1, val_epoch_acc2, val_epoch_acc3, val_epoch_acc4 = 0., 0., 0., 0.
            for val_posts, val_labels in tqdm(val_dataloader, desc="Iteration", position=1):
                val_posts, val_labels = val_posts.to(device), val_labels.to(device)
                val_loss, val_output = model(val_posts, val_labels, teacher_force=0.)
                
                val_accuracy = float((val_output[:,0] == val_labels[:,1]).sum()) / batch_size
                #val_acc1 = float((val_output[:,0] == val_labels[:,1]).sum()) / batch_size
                #val_acc2 = float((val_output[:,1] == val_labels[:,2]).sum()) / batch_size
                #val_acc3 = float((val_output[:,2] == val_labels[:,3]).sum()) / batch_size
                #val_acc4 = float((val_output[:,3] == val_labels[:,4]).sum()) / batch_size
                #val_accuracy = float(torch.all(torch.eq(val_output, val_labels[:,1:-1]), dim=1).sum()) / batch_size

                val_epoch_losses += val_loss.item()
                val_epoch_accuracies += val_accuracy
                #val_epoch_acc1 += val_acc1
                #val_epoch_acc2 += val_acc2
                #val_epoch_acc3 += val_acc3
                #val_epoch_acc4 += val_acc4
                
                del val_loss, val_output
            
            val_last_loss = val_epoch_losses / len(val_dataloader)
            val_last_accuracy = val_epoch_accuracies / len(val_dataloader)
            #val_last_acc1 = val_epoch_acc1 / len(val_dataloader)
            #val_last_acc2 = val_epoch_acc2 / len(val_dataloader)
            #val_last_acc3 = val_epoch_acc3 / len(val_dataloader)
            #val_last_acc4 = val_epoch_acc4 / len(val_dataloader)

            #val_losses.append(val_last_loss)
            #val_accuracies.append(val_last_accuracy)

            print('Val Loss: {:06.6f}, accuracy: {:06.6f}'.format(val_last_loss, val_last_accuracy))
            #print('Val Loss: {:06.6f}, accuracy: {:06.6f}, 1: {:06.6f}, 2: {:06.6f}, 3: {:06.6f}, 4: {:06.6f}'.format(val_last_loss, val_last_accuracy, val_last_acc1, val_last_acc2, val_last_acc3, val_last_acc4))
            
            if val_last_loss < best_loss:
                #best_acc = val_last_accuracy
                best_loss = val_last_loss
                state = {
                    'model': model,
                    'acc': best_acc,
                    'epoch': epoch,
                }
                torch.save(model.state_dict(), "checkpoint/" + task + str(embedding_dim) + "_" + str(hidden_dim) + "_lr" + str(lr) + ".pth")
                early_stop = 0
            else:
                early_stop += 1
                if early_stop == early_num:
                    break

        scheduler.step()
        #warmup_scheduler.dampen()

def prediction():
    SOS = Language.SOS_TOKEN_IDX
    EOS = Language.EOS_TOKEN_IDX

    train_posts = Language(path='lstm.data/' + task + '_train.input')
    train_labels = Language(path='lstm.data/' + task + '_train.label')
    train_posts.build_vocab()
    train_labels.build_vocab()
    model = Seq2Seq(train_posts, train_labels, attention_type=attention_type,
                    embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load("checkpoint/" + task + str(embedding_dim) + "_" + str(hidden_dim) + "_lr" + str(lr) + ".pth", map_location=device))

    test_posts = Language(path='lstm.data/' + task + '_test.input')
    test_labels = Language(path='lstm.data/' + task + '_test.label')
    test_posts.set_vocab(train_posts.word2idx, train_posts.idx2word)
    test_labels.set_vocab(train_labels.word2idx, train_labels.idx2word)
    dataset = NmtDataset(src=test_posts, trg=test_labels, max_len=max_len)
    
    pred_list, label_list, attention_list, post_list = [], [], [], []
    with torch.no_grad():
        model.eval()
        for post, label in tqdm(dataset, desc="Prediction", position=1):
            post = torch.LongTensor(post).to(device)
            pred, attention = model.predict(post, split=False)

            pred_list.append([train_labels.idx2word[idx] for idx in pred.tolist()])
            label_list.append([train_labels.idx2word[label[1]]])
            attention_list.append(attention[0].tolist())
            post_list.append([train_posts.idx2word[idx] for idx in post])
        
    acc = accuracy_score(label_list, pred_list)
    bal_acc = balanced_accuracy_score(label_list, pred_list)
    macro_f1 = f1_score(label_list, pred_list, average='macro')

    print('Test accuracy: {:06.6f}, balanced accuracy: {:06.6f}, macro f1 score: {:06.6f}'.format(acc, bal_acc, macro_f1))

    """
    token_counter = defaultdict(Counter)
    token_score = defaultdict(lambda: defaultdict(float))
    token_score_norm = defaultdict(lambda: defaultdict(float))
    for i, post in enumerate(post_list):
        token_counter[label_list[i][0]].update(post)
        for j, token in enumerate(post):
            token_score[label_list[i][0]][token] += attention_list[i][j]
    for mbti, scores in token_score.items():
        for token, score in scores.items():
            token_score_norm[mbti][token] = score / float(token_counter[mbti][token])
    """
    

if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    random.seed(4321)
    torch.manual_seed(4321)
    train()
    prediction()
