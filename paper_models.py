# %%
# Script for setup of the models used in this paper

from seqdata.core import *
from seqdata.models.core import *
from seqdata.learner import *
from seqdata.dataloaders import *
from seqdata.dualrnn import *
from fastai.basics import *
from fastai.callback.all import *

# %% NarProg model

def model_narprog_small(n_u,n_x,n_y,init_sz,**kwargs):
    rnn_size = 120
    rnn_layers = 1
    diag_rnn_size = 50
    diag_rnn_layers = 1
    diag_nonlinear_layers = 0

    diag_rnn = Diag_RNN(n_u+n_x+n_y,output_size=rnn_size,
                            output_layer=rnn_layers,
                            hidden_size=diag_rnn_size,
                            rnn_layer=diag_rnn_layers,
                            linear_layer = diag_nonlinear_layers,stateful=False,**kwargs)
    model = NarProg(n_u,n_u+n_x+n_y,n_y,init_sz,rnn_layer=rnn_layers,hidden_size=rnn_size,diag_model=diag_rnn,**kwargs)
    return model

def model_narprog_medium(n_u,n_x,n_y,init_sz,**kwargs):
    rnn_size = 220
    rnn_layers = 1
    diag_rnn_size = 100
    diag_rnn_layers = 1
    diag_nonlinear_layers = 0

    diag_rnn = Diag_RNN(n_u+n_x+n_y,output_size=rnn_size,
                            output_layer=rnn_layers,
                            hidden_size=diag_rnn_size,
                            rnn_layer=diag_rnn_layers,
                            linear_layer = diag_nonlinear_layers,stateful=False,**kwargs)
    model = NarProg(n_u,n_u+n_x+n_y,n_y,init_sz,rnn_layer=rnn_layers,hidden_size=rnn_size,diag_model=diag_rnn,**kwargs)
    return model

def model_narprog_large(n_u,n_x,n_y,init_sz,**kwargs):
    rnn_size = 360
    rnn_layers = 2
    diag_rnn_size = 200
    diag_rnn_layers = 1
    diag_nonlinear_layers = 0

    diag_rnn = Diag_RNN(n_u+n_x+n_y,output_size=rnn_size,
                            output_layer=rnn_layers,
                            hidden_size=diag_rnn_size,
                            rnn_layer=diag_rnn_layers,
                            linear_layer = diag_nonlinear_layers,stateful=False,**kwargs)
    model = NarProg(n_u,n_u+n_x+n_y,n_y,init_sz,rnn_layer=rnn_layers,hidden_size=rnn_size,diag_model=diag_rnn,**kwargs)
    return model
# %% NAR TCN model

class Chomp1d(nn.Module):
    # PyTorch module that truncates discrete convolution output for the purposes of causal convolutions
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TConv(nn.Module):
    # Module representing a single causal convolution (truncated 1D convolution)
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TConv, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.init_weights()

    def init_weights(self):  # Initializes weights to positive values
        self.conv1.weight.data.normal_(0, 0.01)

    def forward(self, x):
        return self.net(x)


class TConvBlock(nn.Module):
    # Module representing a temporal convolution block which consists of:
    # - causal convolutions
    # - sequence of conv layers with dilations that increase exponentially

    def __init__(self, c_in, c_out, k, dilations):
        super(TConvBlock, self).__init__()
        self.dsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None  # Downsample layer for residual if required
        self.lookback = 0
        layers = []

        # Adds sequence of causal convolutions to module based on input dilations
        for i in range(len(dilations)):
            d = dilations[i]
            if i == 0:  # Downsample w.r.t channel size at the first convolution
                layers += [TConv(c_in, c_out, k, stride=1, dilation=d, padding=(k - 1) * d)]
            else:
                layers += [TConv(c_out, c_out, k, stride=1, dilation=d, padding=(k - 1) * d)]

            self.lookback += (k - 1) * d    # Calculates total lookback window for layer

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Model forward pass including residual connection
        out = self.network(x)
        res = x if self.dsample is None else self.dsample(x)
        return out + res
    
class End2End_4(nn.Module):
    def __init__(self,n_u,n_x,n_y, init_sz):
        # Final End2EndNet design with fewer layers, fewer channels, no dropout,
        # and control inputs at the front of the network

        # Input: Time series of past robot state, past control input, and future control input (bs x 16 x (P+F))
        # Output: Time series of future truncated robot state (bs x 6 x F)

        super().__init__()
        K = 5
        dilations = [1, 2, 4, 8]
        self.P = init_sz
        self.n_u = n_u
        self.n_y = n_y

        self.tconv1 = TConvBlock(n_u+n_x+n_y, 32, K, dilations)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(32, 32, K, dilations)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(32, 32, K, dilations)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(32, n_y, K, dilations)

    def forward(self, input):
        input[:,self.P:,self.n_u:] = 0
        input = input.transpose(1,2)
        
        x1 = self.relu1(self.bn1(self.tconv1(input)))
        x2 = x1 + self.relu2(self.bn2(self.tconv2(x1)))
        x3 = self.relu3(self.bn3(self.tconv3(x2)))
        out = self.tconv4(x3)

        out = out.transpose(1,2)
        return out

class End2End_8(nn.Module):
    def __init__(self,n_u,n_x,n_y, init_sz):
        # Final End2EndNet design with fewer layers, fewer channels, no dropout,
        # and control inputs at the front of the network

        # Input: Time series of past robot state, past control input, and future control input (bs x 16 x (P+F))
        # Output: Time series of future truncated robot state (bs x 6 x F)

        super().__init__()
        K = 5
        dilations = [1, 2, 4, 8]
        self.P = init_sz
        self.n_u = n_u
        self.n_y = n_y

        self.tconv1 = TConvBlock(n_u+n_x+n_y, 16, K, dilations)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(16, 16, K, dilations)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(16, 32, K, dilations)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(32, 32, K, dilations)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.relu4 = torch.nn.ReLU()
        self.tconv5 = TConvBlock(32, 32, K, dilations)
        self.bn5 = torch.nn.BatchNorm1d(32)
        self.relu5 = torch.nn.ReLU()
        self.tconv6 = TConvBlock(32, 32, K, dilations)
        self.bn6 = torch.nn.BatchNorm1d(32)
        self.relu6 = torch.nn.ReLU()
        self.tconv7 = TConvBlock(32, 32, K, dilations)
        self.bn7 = torch.nn.BatchNorm1d(32)
        self.relu7 = torch.nn.ReLU()
        self.tconv8 = TConvBlock(32, n_y, K, dilations)

    def forward(self, input):
        input[:,self.P:,self.n_u:] = 0
        input = input.transpose(1,2)
        
        x1 = self.relu1(self.bn1(self.tconv1(input)))
        x2 = x1 + self.relu2(self.bn2(self.tconv2(x1)))
        x3 = self.relu3(self.bn3(self.tconv3(x2)))
        x4 = x3 + self.relu4(self.bn4(self.tconv4(x3)))
        x5 = self.relu5(self.bn5(self.tconv5(x4)))
        x6 = x5 + self.relu6(self.bn6(self.tconv6(x5)))
        x7 = self.relu7(self.bn7(self.tconv7(x6)))
        out = self.tconv8(x7)

        out = out.transpose(1,2)

        return out

class End2End_12(nn.Module):
    def __init__(self,n_u,n_x,n_y, init_sz):
        # Final End2EndNet design with fewer layers, fewer channels, no dropout,
        # and control inputs at the front of the network

        # Input: Time series of past robot state, past control input, and future control input (bs x 16 x (P+F))
        # Output: Time series of future truncated robot state (bs x 6 x F)

        super().__init__()
        K = 5
        dilations = [1, 2, 4, 8]
        self.P = init_sz
        self.n_u = n_u
        self.n_y = n_y

        self.tconv1 = TConvBlock(n_u+n_x+n_y, 16, K, dilations)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(16, 16, K, dilations)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(16, 32, K, dilations)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(32, 32, K, dilations)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.relu4 = torch.nn.ReLU()
        self.tconv5 = TConvBlock(32, 32, K, dilations)
        self.bn5 = torch.nn.BatchNorm1d(32)
        self.relu5 = torch.nn.ReLU()
        self.tconv6 = TConvBlock(32, 32, K, dilations)
        self.bn6 = torch.nn.BatchNorm1d(32)
        self.relu6 = torch.nn.ReLU()
        self.tconv7 = TConvBlock(32, 32, K, dilations)
        self.bn7 = torch.nn.BatchNorm1d(32)
        self.relu7 = torch.nn.ReLU()
        self.tconv8 = TConvBlock(32, 32, K, dilations)
        self.bn8 = torch.nn.BatchNorm1d(32)
        self.relu8 = torch.nn.ReLU()
        self.tconv9 = TConvBlock(32, 32, K, dilations)
        self.bn9 = torch.nn.BatchNorm1d(32)
        self.relu9 = torch.nn.ReLU()
        self.tconv10 = TConvBlock(32, 32, K, dilations)
        self.bn10 = torch.nn.BatchNorm1d(32)
        self.relu10 = torch.nn.ReLU()
        self.tconv11 = TConvBlock(32, 32, K, dilations)
        self.bn11 = torch.nn.BatchNorm1d(32)
        self.relu11 = torch.nn.ReLU()
        self.tconv12 = TConvBlock(32, n_y, K, dilations)

    def forward(self, input):
        input[:,self.P:,self.n_u:] = 0
        input = input.transpose(1,2)

        x1 = self.relu1(self.bn1(self.tconv1(input)))
        x2 = x1 + self.relu2(self.bn2(self.tconv2(x1)))
        x3 = self.relu3(self.bn3(self.tconv3(x2)))
        x4 = x3 + self.relu4(self.bn4(self.tconv4(x3)))
        x5 = self.relu5(self.bn5(self.tconv5(x4)))
        x6 = x5 + self.relu6(self.bn6(self.tconv6(x5)))
        x7 = self.relu7(self.bn7(self.tconv7(x6)))
        x8 = x7 + self.relu8(self.bn8(self.tconv8(x7)))
        x9 = self.relu9(self.bn9(self.tconv9(x8)))
        x10 = x9 + self.relu10(self.bn10(self.tconv10(x9)))
        x11 = self.relu11(self.bn11(self.tconv11(x10)))
        out = self.tconv12(x11)

        out = out.transpose(1,2)

        return out
    

def model_e2e_tcn_small(n_u,n_x,n_y,init_sz):
    return End2End_4(n_u,n_x,n_y,init_sz)
def model_e2e_tcn_medium(n_u,n_x,n_y,init_sz):
    return End2End_8(n_u,n_x,n_y,init_sz)
def model_e2e_tcn_large(n_u,n_x,n_y,init_sz):
    return End2End_12(n_u,n_x,n_y,init_sz)
# %% AR LSTM model

#| export
class StatefulRNN(nn.Module):
    def __init__(self, rnn,init_weights=False):
        super().__init__()
        self.bs = 1

        self.rnn = rnn
        self.reset_state()
        if init_weights: self.apply(self._weights_init)

    def forward(self, inp, h_init=None):
        bs,_,_ = inp.shape
        if h_init is None: h_init = self._get_hidden(bs)

        out,hidden = self.rnn(inp, h_init)

        self.hidden =  to_detach(hidden, cpu=False, gather=False)
        self.bs = bs

        return out, hidden

    def _get_hidden(self,bs):
        '''retrieve internal hidden state, check if model device has changed'''
        if self.hidden is None: return None
        if bs!=self.bs: return None
        if self.hidden[0][0].device != one_param(self).device: return None
        return self.hidden
    
    def _weights_init(self, m): 
        # same initialization as keras. Adapted from the initialization developed 
        # by JUN KODA (https://www.kaggle.com/junkoda) in this notebook
        # https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization
        for name, params in m.named_parameters():
            if "weight_ih" in name: 
                nn.init.xavier_normal_(params)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(params)
            elif 'bias_ih' in name:
                params.data.fill_(0)
                # Set forget-gate bias to 1
                n = params.size(0)
                params.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                params.data.fill_(0)

    def reset_state(self):
        self.hidden = None     
#| export
    
def model_ar_rnn_washout_small(n_u,n_x,n_y,init_sz,**kwargs):
    return ARProg(n_u,n_x,n_y,init_sz,num_layers=1,hidden_size=120,**kwargs)
def model_ar_rnn_washout_medium(n_u,n_x,n_y,init_sz,**kwargs):
    return ARProg(n_u,n_x,n_y,init_sz,num_layers=1,hidden_size=250,**kwargs)#220
def model_ar_rnn_washout_large(n_u,n_x,n_y,init_sz,**kwargs):
    return ARProg(n_u,n_x,n_y,init_sz,num_layers=5,hidden_size=200,**kwargs)

# %% AR Lstm Init model

def model_ar_rnn_init_small(n_u,n_x,n_y,init_sz,**kwargs):
    rnn_size = 110
    rnn_layers = 1
    diag_rnn_size = 50
    diag_rnn_layers = 1
    diag_nonlinear_layers = 0

    diag_rnn = Diag_RNN(n_u+n_x+n_y,output_size=rnn_size,
                            output_layer=rnn_layers,
                            hidden_size=diag_rnn_size,
                            rnn_layer=diag_rnn_layers,
                            linear_layer = diag_nonlinear_layers,**kwargs)

    model = ARProg_Init(n_u,n_x,n_y,init_sz,num_layers=rnn_layers,hidden_size=rnn_size,diag_model=diag_rnn,**kwargs)
    return model

def model_ar_rnn_init_medium(n_u,n_x,n_y,init_sz,**kwargs):
    rnn_size = 220#190
    rnn_layers = 1
    diag_rnn_size = 100
    diag_rnn_layers = 1
    diag_nonlinear_layers = 0

    diag_rnn = Diag_RNN(n_u+n_x+n_y,output_size=rnn_size,
                            output_layer=rnn_layers,
                            hidden_size=diag_rnn_size,
                            rnn_layer=diag_rnn_layers,
                            linear_layer = diag_nonlinear_layers,**kwargs)

    model = ARProg_Init(n_u,n_x,n_y,init_sz,num_layers=rnn_layers,hidden_size=rnn_size,diag_model=diag_rnn,**kwargs)
    return model

def model_ar_rnn_init_large(n_u,n_x,n_y,init_sz,**kwargs):
    rnn_size = 350
    rnn_layers = 2
    diag_rnn_size = 170
    diag_rnn_layers = 1
    diag_nonlinear_layers = 0

    diag_rnn = Diag_RNN(n_u+n_x+n_y,output_size=rnn_size,
                            output_layer=rnn_layers,
                            hidden_size=diag_rnn_size,
                            rnn_layer=diag_rnn_layers,
                            linear_layer = diag_nonlinear_layers,**kwargs)
    model = ARProg_Init(n_u,n_x,n_y,init_sz,num_layers=rnn_layers,hidden_size=rnn_size,diag_model=diag_rnn,**kwargs)
    return model
# %% MLP-SSM model

from torchid.ss.dt.models import NeuralStateUpdate, NeuralOutput
from torchid.ss.dt.simulator import StateSpaceSimulator
from torchid.ss.dt.estimators import LSTMStateEstimator
from torchid.datasets import SubsequenceDataset

class SSM(nn.Module):
    
    def __init__(self,n_u,n_x,n_y, init_sz,state_size=10, hidden_size=64):
        super().__init__()
        store_attr()
        f_xu = NeuralStateUpdate(n_x=state_size, n_u=n_u, hidden_size=hidden_size)
        g_x = NeuralOutput(n_x=state_size, n_y=n_x+n_y) 
        self.model = StateSpaceSimulator(f_xu, g_x)
        self.estimator = LSTMStateEstimator(n_u=n_u, n_y=n_x+n_y, n_x=state_size,hidden_size=hidden_size)

    def forward(self, inp):
        inp = inp.transpose(0, 1)

        u = inp[..., :self.n_u]
        y = inp[..., self.n_u:]

        out_init = self.estimator(u[:self.init_sz,...], y[:self.init_sz,...])
        out_prog = self.model(out_init,u[self.init_sz:,...])

        result=torch.cat([torch.zeros(self.init_sz,out_prog.shape[1],out_prog.shape[2],device=inp.device),out_prog],0) 

        return result.transpose(0, 1)[...,-self.n_y:]
    
def model_mlp_ssm_small(n_u,n_x,n_y,init_sz,**kwargs):
    state_size = 10
    hidden_size=110
    model = SSM(n_u,n_x,n_y,init_sz,state_size,hidden_size=hidden_size,**kwargs)
    return model
def model_mlp_ssm_medium(n_u,n_x,n_y,init_sz,**kwargs):
    state_size = 50
    hidden_size=200
    model = SSM(n_u,n_x,n_y,init_sz,state_size,hidden_size=hidden_size,**kwargs)
    return model
def model_mlp_ssm_large(n_u,n_x,n_y,init_sz,**kwargs):
    state_size = 200
    hidden_size=500
    model = SSM(n_u,n_x,n_y,init_sz,state_size,hidden_size=hidden_size,**kwargs)
    return model

def get_model_constr(model_type,model_size):
    if model_type== 'nargru':
        if model_size == 'small':
            return model_narprog_small
        elif model_size == 'medium':
            return model_narprog_medium
        elif model_size == 'large':
            return model_narprog_large
    elif model_type== 'argru_init':
        if model_size == 'small':
            return model_ar_rnn_init_small
        elif model_size == 'medium':
            return model_ar_rnn_init_medium
        elif model_size == 'large':
            return model_ar_rnn_init_large
    elif model_type== 'argru_washout':
        if model_size == 'small':
            return model_ar_rnn_washout_small
        elif model_size == 'medium':
            return model_ar_rnn_washout_medium
        elif model_size == 'large':
            return model_ar_rnn_washout_large


dict_models = {
    'narprog_small':model_narprog_small,
    'narprog_medium':model_narprog_medium,
    'narprog_large':model_narprog_large,
    'e2e_tcn_small':model_e2e_tcn_small,
    'e2e_tcn_medium':model_e2e_tcn_medium,
    'e2e_tcn_large':model_e2e_tcn_large,
    'ar_rnn_washout_small':model_ar_rnn_washout_small,
    'ar_rnn_washout_medium':model_ar_rnn_washout_medium,
    'ar_rnn_washout_large':model_ar_rnn_washout_large,
    'ar_rnn_init_small':model_ar_rnn_init_small,
    'ar_rnn_init_medium':model_ar_rnn_init_medium,
    'ar_rnn_init_large':model_ar_rnn_init_large,
    'mlp_ssm_small':model_mlp_ssm_small,
    'mlp_ssm_medium':model_mlp_ssm_medium,
    'mlp_ssm_large':model_mlp_ssm_large
}