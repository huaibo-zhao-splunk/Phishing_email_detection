import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import BertModel, BertTokenizerFast, BertConfig
# from transformers import DistilBertModel, DistilBertTokenizerFast
import os
import tqdm
import time
# specify GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("/home/ec2-user/SageMaker/email_phishing/Huaibo/data/data_v5_train.csv", 'r') as f:
    df_train = pd.read_csv(f)
with open("/home/ec2-user/SageMaker/email_phishing/Huaibo/data/data_v5_test.csv", 'r') as f:
    df_test = pd.read_csv(f)
# df = df.dropna()
# df_train = df.loc[df['is_train'] == 1]
# df_test = df.loc[df['is_train'] == 0]
# df_train = df_train.sample(frac=1, random_state=111)
# df_test = df_test.sample(frac=1, random_state=111)

labels = 'label'
train_text = df_train.text.values
train_labels = df_train[labels].values
val_text = df_test.text.values
val_labels = df_test[labels].values
test_text = df_test.text.values
test_labels = df_test[labels].values


# MODEL_NAME="/home/ec2-user/SageMaker/email_phishing/Huaibo/model/bert_classification_en"
config = BertConfig.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained("bert-base-uncased", config=config)
# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# freeze all the parameters

# for param in bert.parameters():
#     param.requires_grad = False
num_layers = config.num_hidden_layers

# Freeze all layers except the last transformer layer
for name, param in bert.named_parameters():
    if 'encoder.layer.' in name:
        layer_index = int(name.split('.')[2])  # extract the layer index from the parameter name
        if layer_index == num_layers - 1:
            param.requires_grad = True  # unfreeze the parameters in the last transformer layer
        else:
            param.requires_grad = False 
            
            
            

print("Start tokenizing")

# MODEL_NAME="/home/ec2-user/SageMaker/email_phishing/Huaibo/model/bert_classification_en"
# bert = BertModel.from_pretrained(MODEL_NAME, local_files_only=True)
# # Load the BERT tokenizer
# tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

max_seq_len = 512

# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# for validation set
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

# for test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size
batch_size = 64

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

print("Finished tokenizing")
            

class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

        #pass the inputs to the model  
        cls_hs = self.bert(sent_id, attention_mask=mask)
#         print(cls_hs[0][:,0,:])
        x = self.fc1(cls_hs[0][:,0,:])

        x = self.relu(x)

        x = self.dropout(x)
        
        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x
    

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)
# Load fine-tuned parallel model file
# MODEL_DIRECTORY = "/home/ec2-user/SageMaker/email_phishing/Huaibo/model/bert_phishing"
model = nn.DataParallel(model)
# model.load_state_dict(torch.load(os.path.join(MODEL_DIRECTORY, "pytorch_model.pt"), map_location=torch.device(device)))
model = model.to(device)

# optimizer from hugging face transformers
from transformers import AdamW

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-3)

from sklearn.utils.class_weight import compute_class_weight

#compute the class weights
# class_wts = compute_class_weight(class_weight='balanced', classes=[0,1], y=train_labels)

# convert class weights to tensor
# weights= torch.tensor(class_wts,dtype=torch.float)
# weights = weights.to(device)

# loss function
# cross_entropy  = nn.NLLLoss(weight=weights) 
cross_entropy  = nn.NLLLoss()
# number of training epochs
epochs = 10


# function to train the model
def train():
  
  model.train()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save model predictions
  total_preds=[]
  
  # iterate over batches
  for step, batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
#     if step % 50 == 0 and not step == 0:
#       print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    # push the batch to gpu
    batch = [r.to(device) for r in batch]
 
    sent_id, mask, labels = batch

    # clear previously calculated gradients 
    model.zero_grad()        

    # get model predictions for the current batch
    preds = model(sent_id, mask)

    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)

    # add on to the total loss
    total_loss = total_loss + loss.item()

    # backward pass to calculate the gradients
    loss.backward()

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # update parameters
    optimizer.step()

    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
  
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  #returns the loss and predictions
  return avg_loss, total_preds


# function for evaluating the model
def evaluate():
  
  print("\nEvaluating...")
  
  # deactivate dropout layers
  model.eval()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save the model predictions
  total_preds = []

  # iterate over batches
  for step,batch in enumerate(val_dataloader):
    
    # Progress update every 50 batches.
#     if step % 50 == 0 and not step == 0:
      
#       # Calculate elapsed time in minutes.
#       elapsed = format_time(time.time() - t0)
            
#       # Report progress.
#       print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    # push the batch to gpu
    batch = [t.to(device) for t in batch]

    sent_id, mask, labels = batch

    # deactivate autograd
    with torch.no_grad():
      
      # model predictions
      preds = model(sent_id, mask)

      # compute the validation loss between actual and predicted values
      loss = cross_entropy(preds,labels)

      total_loss = total_loss + loss.item()

      preds = preds.detach().cpu().numpy()

      total_preds.append(preds)

  # compute the validation loss of the epoch
  avg_loss = total_loss / len(val_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds



MODEL_DIRECTORY = "/home/ec2-user/SageMaker/email_phishing/Huaibo/model/BERT"

def finetune():
   # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses=[]
    valid_losses=[]

    #for each epoch
    for epoch in range(epochs):
        
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        
        #train model
        train_loss, _ = train()
        
        #evaluate model
        valid_loss, _ = evaluate()
        
        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            tokenizer.save_pretrained(MODEL_DIRECTORY)
            print("tokenizer saved in " + MODEL_DIRECTORY, flush=True)
            torch.save(model.state_dict(),os.path.join(MODEL_DIRECTORY, "pytorch_model.pt"))
            print("model saved in " + MODEL_DIRECTORY, flush=True)
    #         torch.save(model.state_dict(), 'saved_weights.pt')
        
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')


if __name__ == "__main__":
   finetune()