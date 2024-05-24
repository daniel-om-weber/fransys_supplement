import paper_dataloaders
import paper_models

from seqdata.core import *
from seqdata.models.core import *
from seqdata.learner import *
from seqdata.dataloaders import *
from seqdata.dualrnn import *
from seqdata.hpopt import *
from fastai.basics import *
from fastai.callback.all import *
from seqdata.quaternions import *
from pathlib import Path
from ray import tune
import pickle

from paper_dataloaders import *
from paper_models import *

if __name__ == "__main__":

    def create_lrn(dls,config):
        dls = dict_dls_fransys[config['dl']]()
        n_u, n_y, n_x, init_sz, _ = get_dls_parameters(dict_dls_fransys[config['dl']])
        
        if config['model']== 'nargru':
            if config['size'] == 'small':
                model_type = model_narprog_small
            elif config['size'] == 'medium':
                model_type = model_narprog_medium
            elif config['size'] == 'large':
                model_type = model_narprog_large
        elif config['model']== 'argru_init':
            if config['size'] == 'small':
                model_type = model_ar_rnn_init_small
            elif config['size'] == 'medium':
                model_type = model_ar_rnn_init_medium
            elif config['size'] == 'large':
                model_type = model_ar_rnn_init_large
        elif config['model']== 'argru_washout':
            if config['size'] == 'small':
                model_type = model_ar_rnn_washout_small
            elif config['size'] == 'medium':
                model_type = model_ar_rnn_washout_medium
            elif config['size'] == 'large':
                model_type = model_ar_rnn_washout_large
        
        model = model_type(n_u,n_x,n_y,init_sz,
                           hidden_p=config['hidden_p'] if config['has_hidden_p'] else 0,
                           weight_p=config['weight_p'] if config['has_weight_p'] else 0)
        
        norm_mean,norm_std = extract_mean_std_from_dls(dls)
        norm_mean = norm_mean[...,-n_y:]
        norm_std = norm_std[...,-n_y:]
        if config['model'] == 'argru_init':
            model.rnn_prognosis.init_normalize_values(norm_mean,norm_std)
            
        elif config['model'] == 'argru_washout':
            model.rnn_model.init_normalize_values(norm_mean,norm_std)

        lrn = Learner(dls,model,loss_func=SkipNLoss(mae,init_sz),
                        metrics=[SkipNLoss(rmse,init_sz)],
                      wd=config['wd'],
                     opt_func=partial(RAdam,beta=config['opt_beta']))
        
        if config['model']== 'nargru':
            prog_model = model.rnn_prognosis
        elif config['model']== 'argru_init':
            prog_model = model.rnn_prognosis.model.rnn 
        elif config['model']== 'argru_washout':
            prog_model = model.rnn_model.model.rnn

        lrn.add_cb(TimeSeriesRegularizer(alpha=config['tsr_alpha'],
                                         beta=config['tsr_beta'],
                                         modules=[prog_model]))

        
        lrn.add_cb(CancelNaNCallback())
        lrn.add_cb(TbpttResetCB())
        return lrn
        
    def update_search_space(search_space,stored_hp,key_func):
        def __inner(config,key):
            return stored_hp[key_func(config)][key]
        
        hp_entry = first(stored_hp.values())
        for key in hp_entry.keys():
            search_space.setdefault(key, tune.sample_from(partial(__inner,key=key)))

        
    search_space = {
        
        'dl': tune.grid_search(list(dict_dls_fransys.keys())),
        'model': tune.grid_search(['nargru','argru_init','argru_washout']),
        'size': tune.grid_search(['small','medium','large']),
        'pct_start': 0.2,
        'n_epoch': 100
    }

    hpopt_configs = pickle.load(open('configs_4A.p', 'rb'))
    update_search_space(search_space,hpopt_configs,lambda config:tuple([config['dl'],config['model'],config['size']]))

    hp_opt = HPOptimizer(create_lrn,None)
    hp_opt.start_ray()
    
    hp_opt.optimize(resources_per_trial={"gpu": 1/2},
                    num_samples=10,
                    config=search_space,
                    metric='_rmse',
                    mode='min',
                    name='FranSys_P4A_models_hpopt',
                    storage_path='~/ray_results')


# %%
