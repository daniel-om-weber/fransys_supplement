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

    models = ['nargru','argru_washout','argru_init']
    sizes =['small','medium','large']
    dls = list(dict_dls_fransys.keys())

    for model in models:
        for size in sizes:
            for dl in dls:
                search_space = {
                    'wd': tune.loguniform(1e-5,1e1),
                    'tsr_alpha': tune.loguniform(1e-3,1e3),
                    'tsr_beta': tune.loguniform(1e-3,1e3),
                    'hidden_p': tune.uniform(0,0.8),
                    'has_hidden_p': tune.choice([True,False]),
                    'weight_p': tune.uniform(0,0.8),
                    'has_weight_p': tune.choice([True,False]),
                    
                    'opt_beta': tune.loguniform(1e1,1e4),
                    'lr': tune.loguniform(1e-4,1e-2),
                    
                    'model': model,
                    'size': size,
                    'dl': dl,
                    'pct_start': 0.2,
                    'n_epoch': 100
                }
            
                hp_opt = HPOptimizer(create_lrn,None)
                hp_opt.start_ray()

                if size == 'large':
                    time_h = 6
                else:
                    time_h = 3
                
                from ray.tune.schedulers import AsyncHyperBandScheduler
                scheduler = AsyncHyperBandScheduler(grace_period=8, max_t=100)
                hp_opt.optimize(resources_per_trial={"gpu": 1/3 if model == 'argru_washout' else 1/2},#maximize gpu utilization for argru_washout
                                num_samples=-1,
                                time_budget_s=time_h*60*60,
                                config=search_space,
                                metric='_rmse',
                                mode='min',
                                scheduler=scheduler,
                                raise_on_failed_trial=False,
                                name='FranSys_P4A_hpopt',
                                storage_path='~/ray_results')


# %%
