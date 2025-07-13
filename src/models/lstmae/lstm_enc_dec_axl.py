import torch
import torch.nn as nn

from models.algorithm_utils import PyTorchUtils

class LSTMEDModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, hidden_size: int = 5,
                 n_layers: tuple= (1, 1), use_bias: tuple= (True, True), dropout: tuple= (0, 0),
                 seed: int= None, gpu: int= None):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.to_device(self.encoder)
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.to_device(self.decoder)

        #########################################################################
        ## Bidirectional setting
        # self.hidden2output = nn.Linear(self.hidden_size*2, self.n_features)

        ## Default setting
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        
        ## Multiple Output Layers setting
        # self.hidden2output = nn.Sequential(
        #     nn.Linear(self.hidden_size,int(self.hidden_size*2/3)),
        #     nn.Linear(int(self.hidden_size*2/3),int(self.hidden_size*4/9)),
        #     nn.Linear(int(self.hidden_size*4/9), self.n_features))
        #########################################################################
        self.to_device(self.hidden2output)

    def _init_hidden(self, batch_size):
        #########################################################################
        ## Default setting
        return (self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()),
                self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()))
        
        ## Bidirectional setting
        # return (self.to_var(torch.Tensor(self.n_layers[0]*2, batch_size, self.hidden_size).zero_()),
        #         self.to_var(torch.Tensor(self.n_layers[0]*2, batch_size, self.hidden_size).zero_()))
        #########################################################################
         
    def forward(self, ts_batch, return_latent: bool = False):   #change for latent space- change return_latent: bool=True
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = enc_hidden

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):
        #########################################################################
            ## Default setting
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            ## Bidirectional setting
            #output[:, i, :] = self.hidden2output(dec_hidden[0].view(batch_size, -1))
        #########################################################################
            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return (output, enc_hidden[1][-1]) if return_latent else output
