import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, BertPreTrainedModel, BertModel, BertConfig
import torch.optim as optim
from tqdm import tqdm, trange
import data2 as data

class BertForMBTI(BertPreTrainedModel):
    """ BERT model for Squad dataset
    Implement proper a question and answering model based on BERT.
    We are not going to check whether your model is properly implemented.
    If the model shows proper performance, it doesn't matter how it works.

    BertPretrinedModel Examples:
    https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForQuestionAnswering
    """
    def __init__(self, config: BertConfig):
        """ Model Initializer
        You can declare and initialize any layer if you want.
        """
        super().__init__(config)
        ### YOUR CODE HERE
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.binary_outputs = nn.Linear(config.hidden_size,config.num_labels)
        ### END YOUR CODE

        # Don't forget initializing the weights
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask: torch.Tensor
    ):
        """ Model Forward Function
        There is no format for the return values.
        However, the input must be in the prescribed form.

        Arguments:
        input_ids -- input_ids is a tensor 
                    in shape (batch_size, sequence_length)
        attention_mask -- attention_mask is a tensor
                    in shape (batch_size, sequence_length)
        token_type_ids -- token_type ids is a tensor
                    in shape (batch_size, sequence_length)

        Returns:
        FREE-FORMAT
        """
        ### YOUR CODE HERE
        outputs = self.bert(input_ids,attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.binary_outputs(sequence_output)[:,:4,:]
        p1, p2, p3, p4 = logits.split(1,dim=1)
        p1 = p1.squeeze(-2)
        p2 = p2.squeeze(-2)
        p3 = p3.squeeze(-2)
        p4 = p4.squeeze(-2)

        return p1,p2,p3,p4
        ### END YOUR CODE




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

def mbti_collate_fn(samples):
    """ Squad sample sollate function
    This function also generates attention mask

    How to Use:
    data_loader = Dataloader(squad_feature_dataset, ..., collate_fn=squad_feature_collate_fn)
    """
    input_ids, token_type_ids, start_pos, end_pos = zip(*samples)
    attention_mask = [[1] * len(input_id) for input_id in input_ids]

    input_ids = pad_sequence([torch.Tensor(input_id).to(torch.long) for input_id in input_ids], \
                             padding_value=0, batch_first=True)
    token_type_ids = pad_sequence([torch.Tensor(token_type_id).to(torch.long) for token_type_id in token_type_ids], \
                                  padding_value=1, batch_first=True)
    attention_mask = pad_sequence([torch.Tensor(mask).to(torch.long) for mask in attention_mask], \
                                  padding_value=0, batch_first=True)

    start_pos = torch.Tensor(start_pos).to(torch.long)
    end_pos = torch.Tensor(end_pos).to(torch.long)
    return input_ids, attention_mask, token_type_ids, start_pos, end_pos


def training(model, train_data, val_data):
    epochs = 10        
    learning_rate = 1e-4  
    device = torch.device('cuda:7')
    batch_size = 16
    model = BertForMBTI.from_pretrained('bert-base-uncased')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    # bal_pred, bal_ans = [], []

    for epoch in trange(epochs, desc="Epoch"):
        tl, ta, vl, va = 0, 0, 0, 0
        model.train()
        model.to(device)
        for label, text, mask in tqdm(train_data, desc="Training Iteration"):
            label, text, mask = label.to(device, dtype=torch.float), text.to(device), mask.to(device)
            optimizer.zero_grad()
            p1,p2,p3,p4=model(text, attention_mask=mask)
            sfm = nn.Softmax(dim=1)
            p1,p2,p3,p4 = sfm(p1),sfm(p2),sfm(p3),sfm(p4)
            bceloss = nn.BCELoss()
            l1 = bceloss(p1[:,0],label[:,0]) 
            l2 = bceloss(p2[:,0],label[:,0]) 
            l3 = bceloss(p3[:,0],label[:,0]) 
            l4 = bceloss(p4[:,0],label[:,0]) 
            loss = l1+l2+l3+l4
            loss.backward()
            optimizer.step()
            tl+=loss
            del loss
            preds1 = torch.argmax(p1.data, dim=1)
            preds2 = torch.argmax(p2.data, dim=1)
            preds3 = torch.argmax(p3.data, dim=1)
            preds4 = torch.argmax(p4.data, dim=1)
            preds = torch.stack([preds1,preds2,preds3,preds4]).T
            preds = (preds==label).sum(dim=-1)==4
            ta+=preds.sum()/batch_size
            
        train_losses.append(float(tl/len(train_data)))
        train_accuracies.append(float(ta/len(train_data)))
        print('avg_t_loss=',tl/len(train_data),'avg_t_acc=',ta/len(train_data))

        with torch.no_grad():
          model.eval()
          for label, text, mask in tqdm(val_data, desc="Validation Iteration"):
              label, text, mask = label.to(device, dtype=torch.float), text.to(device), mask.to(device)
              p1,p2,p3,p4=model(text, attention_mask=mask)
              preds1 = torch.argmax(p1.data, dim=1)
              preds2 = torch.argmax(p2.data, dim=1)
              preds3 = torch.argmax(p3.data, dim=1)
              preds4 = torch.argmax(p4.data, dim=1)
              sfm = nn.Softmax(dim=1)
              p1,p2,p3,p4 = sfm(p1),sfm(p2),sfm(p3),sfm(p4)
              bceloss = nn.BCELoss()
              l1 = bceloss(p1[:,0],label[:,0]) 
              l2 = bceloss(p2[:,0],label[:,0]) 
              l3 = bceloss(p3[:,0],label[:,0]) 
              l4 = bceloss(p4[:,0],label[:,0]) 
              loss = l1+l2+l3+l4
              preds = torch.stack([preds1,preds2,preds3,preds4]).T
              preds = (preds==label).sum(dim=-1)==4
              va+=preds.sum()/batch_size
              vl+=loss
              del loss
            #   bal_pred.append(preds)
            #   bal_ans.append(label)
        val_losses.append(float(vl/len(val_data)))
        val_accuracies.append(float(va/len(val_data)))
        # bal_score = metrics.balanced_accuracy_score(label,preds)
        # f1_score = metrics.f1_score(label,preds, average='macro')
        # print('avg_v_loss=',vl/len(val_data),'avg_v_acc=',va/len(val_data),'bal_score=',bal_score,'f1_score=',f1_score)
        print('avg_v_loss=',vl/len(val_data),'avg_v_acc=',va/len(val_data))
    
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