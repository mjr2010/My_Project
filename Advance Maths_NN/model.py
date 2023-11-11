import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

def req_data(batches):
    global data
    global device
    device = 'cpu'
    data = batches
    return None

# define class object of bigram language model
# this bigram language model has 3 layers, an embedding layer, a fully connected layer (with a ReLU activation function), and another fully connected layer
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size): # this function is called everytime an instance of bigram language model is created
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size) 
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x): # how the input data should be passed through model
        x = self.embedding(x)
        x = torch.mean(x, dim=1)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def generate_text(model, vocab_size, start_token, length): # generate new tokens based on given tokens, keep in mind the model hasn't been given enough data. its just a sample generator
        generated_text = [start_token]
        x = torch.tensor([start_token])
        for i in range(length):
            scores = model(x)
            prob = nn.functional.softmax(scores, dim=-1) # softmax function tries to predict what will be likely the next token in a sequence
            x = torch.multinomial(prob, 1)
            generated_text.append(x.item())
        generated_text = [chr(ord('A')+i) for i in generated_text]  
        return ''.join(generated_text)

    def bigram_train(self, x, y, num_epochs=10, learning_rate=0.01): # train the modell. here we have only 4 batch and each block-size is 8. so, in total we have 32 tokens chosen randomly as input tokens and 32 corresponding target tokens
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate) # adam optimizer from torch library to mimizie the loss function
        criterion = nn.CrossEntropyLoss() 
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            scores = self(x)
            loss = criterion(scores.view(-1,20), y.view(-1))
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

def bigram():
    vocab_size = 20  # 20 characters
    embedding_size = 32
    hidden_size = 64
    model = BigramLanguageModel(vocab_size, embedding_size, hidden_size)
    x = torch.randint(low=0, high=vocab_size, size=(128, 2))
    y  = torch.randint(low=0, high=vocab_size, size=(128,))
    model.bigram_train(x, y, num_epochs=10, learning_rate=0.01) # train the model with learning rate of 0.01 and 10 epoches

class Transformer(nn.Module):
    """
    transformer model object. transformer is based on attention mechanism decoder and encoder. it takes an input, tokenize it in embedding layer and in attention it
    learns the relation between each character in a sequence. SInce attention has a bigger window size compared to LSTM and GRU its able to learn relations better.
    """
    # the transformer object initialization
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,):
        
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model
        
        # postional encoder layer which is defined below
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000)
        # embedding layer from torch.nn module
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,)
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        '''
        feed forward function of the transformer
        '''
        # src size is a tuple (batch_size, src sequence length)
        # tgt size is a tuple (batch_size, tgt sequence length)

        # embedding layers, map tokens to vectors
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        # positional encoding, learn the relation between tokens 
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
    
        src = src.permute(1,0,2) # chnage the shpae of inputs
        tgt = tgt.permute(1,0,2) # change the shape on targets

        # store the output of the given input to the model
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        return out
    
    def get_tgt_mask(self, size) -> torch.tensor:
        '''
        function to mask some tokens in the target 
        '''
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        '''
        function to determine padding 
        '''
        return (matrix == pad_token)
    
class PositionalEncoding(nn.Module):
    '''
    postional encoding object. the purpose is to learn relation between tokens in a given sequence
    '''
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_p) # dropout layer from torch.nn module
        
        pos_encoding = torch.zeros(max_len, dim_model)  # encoding
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) 
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) 
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term) # positional encoding
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term) # positional encoding
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        '''
        dropout layer which takes encoded tokens                     
        '''
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
    
def train(model, opt, loss_fn, dataloader):
    """
    function to train the model for train dataset
    """
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        X, y = batch[:, 0], batch[:, 1]
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)
        
        # window size of inputes should be shifted by one to get the target value 
        y_input = y[:,:-1]
        y_expected = y[:,1:]
        
        # get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)
        
        # calculate predictions using inputs which are multiple batches of train set 
        pred = model(X, y_input, tgt_mask)

        # change the shape of pred to have batch size first again
        pred = pred.permute(1, 2, 0)      
        loss = loss_fn(pred, y_expected)
        
        # optimization, backward propagation to set the weights for each neuron in the model to get minimzie loss function
        opt.zero_grad()
        loss.backward()
        opt.step()
    
        total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)

def validation(model, loss_fn, dataloader):
    """
    function to validate the accuracy of the model
    """
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X, dtype=torch.long, device="cpu"), torch.tensor(y, dtype=torch.long, device="cpu")

            # we shift the target by one 
            y_input = y[:,:-1]
            y_expected = y[:,1:]
            
            # get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to("cpu")
            
            # calculate predictions using inputs which are multiple batches of validation set
            pred = model(X, y_input, tgt_mask)
            
            # change the shape of pred to have batch size first again
            pred = pred.permute(1, 2, 0)      
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)

def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    """
    fit the model to train set and validation set, optimize parameters and calculate loss function 
    """
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    
    for epoch in range(epochs):
        
        train_loss = train(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]
        
        validation_loss = validation(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        
    return train_loss_list, validation_loss_list

def f1(train_dataloader, val_dataloader):
    model1 = Transformer(num_tokens=4, dim_model=8, num_heads=2, num_encoder_layers=4, num_decoder_layers=4, dropout_p=0.1).to("cpu")
    opt = torch.optim.SGD(model1.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    train_loss1, validation_loss1 = fit(model1, opt, loss_fn, train_dataloader, val_dataloader, 10)
    x_points = np.arange(1,11)
    plt.title('model1 train and valdation loss')
    plt.plot(x_points, train_loss1, 'b--', label = 'tain set')
    plt.plot(x_points, validation_loss1, 'g--', label = 'validation set')
    plt.xlabel('$epoches$')
    plt.ylabel('$loss$')
    plt.legend(loc = 'upper right')
    plt.show();

def f2(train_dataloader, val_dataloader):
    model2 = Transformer(num_tokens=4, dim_model=8, num_heads=2, num_encoder_layers=2, num_decoder_layers=2, dropout_p=0.1).to("cpu")
    opt = torch.optim.SGD(model2.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    train_loss2, validation_loss2 = fit(model2, opt, loss_fn, train_dataloader, val_dataloader, 10)
    x_points = np.arange(1,11)
    plt.title('model1 train and valdation loss')
    plt.plot(x_points, train_loss2, 'b--', label = 'tain set')
    plt.plot(x_points, validation_loss2, 'g--', label = 'validation set')
    plt.xlabel('$epoches$')
    plt.ylabel('$loss$')
    plt.legend(loc = 'upper right')
    plt.show();

def f3(train_dataloader, val_dataloader):
    model3 = Transformer(num_tokens=4, dim_model=8, num_heads=2, num_encoder_layers=2, num_decoder_layers=4, dropout_p=0.1).to(device)
    opt = torch.optim.SGD(model3.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    train_loss3, validation_loss3 = fit(model3, opt, loss_fn, train_dataloader, val_dataloader, 10)
    x_points = np.arange(1,11)
    plt.title('model 3 train and valdation loss')
    plt.plot(x_points, train_loss3, 'b--', label = 'tain set')
    plt.plot(x_points, validation_loss3, 'g--', label = 'validation set')
    plt.xlabel('$epoches$')
    plt.ylabel('$loss$')
    plt.legend(loc = 'upper right')
    plt.show();