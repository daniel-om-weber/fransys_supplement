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
    
        pred_sz = config['pred_sz']
        if pred_sz is not None:
            dls = dict_dls_fransys[config['dl']](pred_sz= pred_sz)
        else:
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

        lrn = Learner(dls,model,loss_func=SkipNLoss(mae,init_sz),
                        metrics=[SkipNLoss(rmse,init_sz)],
                      wd=config['wd'],
                     opt_func=partial(RAdam,beta=config['opt_beta']))

        
        schedule_type = partial(sched_ramp,p_left=config['p_left'], p_right=config['p_right'])
        if config['schedule_pred']:
            lrn.add_cb(CB_TruncateSequence(init_sz+config['init_pred'],schedule_type))
        
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

    def get_init_pred_sample(config):
        dl = config['dl']
        match dl:
            case 'Quadrotor':
                return np.random.uniform(130,200)
            case 'Robot':
                return np.random.uniform(100,150)
            case 'Ship':
                return np.random.uniform(1,60)

    hpopt_configs = pickle.load(open('configs_4A.p', 'rb'))
    sizes = ['small','medium','large']
    for size in sizes:
        time_h = 4 if size == 'large' else 2
        for dl in list(dict_dls_fransys.keys()):

            search_space = {
                'dl': dl,
                'init_pred': tune.sample_from(get_init_pred_sample),#tune.uniform(30,180),
                'p_left': tune.uniform(0.2,0.5),
                'p_right': tune.uniform(0.6,0.9),
                'model': 'nargru',
                'size': size,
                'pred_sz': 200,
                'schedule_pred': True,
                'pct_start': 0.2,
                'n_epoch': 100
            }
            update_search_space(search_space,hpopt_configs,lambda config:tuple([config['dl'],config['model'],config['size']]))

        
            hp_opt = HPOptimizer(create_lrn,None)
            hp_opt.start_ray()

            
            hp_opt.optimize(resources_per_trial={"gpu": 1 if size == 'large' else 1/2},
                            num_samples=-1,
                            time_budget_s=time_h*60*60,#4*60*60,
                            config=search_space,
                            metric='_rmse',
                            mode='min',
                            raise_on_failed_trial=False,
                            name='FranSys_P4B_hpopt',
                            storage_path='~/ray_results')


# %%
