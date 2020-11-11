import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool1d, Module, BatchNorm1d, Dropout


class FaceCNet(Module):
    def __init__(self, batch_sz, input_dim, hidden_dim, hidden_dim2, seq_length, n_layers, output_dim):
        super(FaceCNet, self).__init__()
        self.batch_size = batch_sz
        self.input_dim = input_dim
        self.seq_len = seq_length
        self.hidden_dim = hidden_dim  # number of hidden states
        self.hidden_dim2 = hidden_dim2
        self.n_layers = n_layers  # number of LSTM layers (stacked)
        self.output_dim = output_dim
        self.hidden = None
        self.l_lstm = torch.nn.LSTM(input_size=self.input_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=self.n_layers,
                                    batch_first=True)
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_batchnorm = torch.nn.BatchNorm1d(num_features = self.hidden_dim)
        self.l_linear1 = torch.nn.Linear(self.hidden_dim, self.hidden_dim2)
        self.l_linear2 = torch.nn.Linear(self.hidden_dim2, self.output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def init_hidden(self):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim)
        cell_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim)
        if torch.cuda.is_available():
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        #batch_size, seq_len, _ = x.size()
        lstm_out, self.hidden = self.l_lstm(x)
        #lstm_out is of shape [batch_size, seq_length, features_dim]
        #change to [batch_size, features_dim, seq_length] for batch norm
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        lstm_out = self.l_batchnorm(lstm_out)
        lstm_out = torch.dropout(lstm_out, 0.5, True)
        # lstm_out(with batch_first = True) is
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error
        #lstm_out = lstm_out.permute(1, 0, 2)
        # [seq_length, batch_size, dimension]

        lstm_out = self.l_linear1(lstm_out)
        lstm_out = self.l_linear2(lstm_out)
        lstm_out = torch.dropout(lstm_out, 0.5, True)
        sig_out = self.sigmoid(lstm_out)
        if( sig_out.shape[0] < self.batch_size ) :
            return sig_out

        sig_out = sig_out.view(self.batch_size, -1)
        sig_out = sig_out[:, -2:]
       
        return sig_out