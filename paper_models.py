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
    'ar_rnn_washout_small':model_ar_rnn_washout_small,
    'ar_rnn_washout_medium':model_ar_rnn_washout_medium,
    'ar_rnn_washout_large':model_ar_rnn_washout_large,
    'ar_rnn_init_small':model_ar_rnn_init_small,
    'ar_rnn_init_medium':model_ar_rnn_init_medium,
    'ar_rnn_init_large':model_ar_rnn_init_large,
}