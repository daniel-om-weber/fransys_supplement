

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,3'

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

        
        if config['schedule_pred']:
           schedule_type = partial(sched_ramp,p_left=config['p_left'], p_right=config['p_right'])
           lrn.add_cb(CB_TruncateSequence(init_sz+config['init_pred'],schedule_type))
        
        if config['model']== 'nargru':
            prog_model = model.rnn_prognosis

            cb = NarProgCallback([model.rnn_diagnosis,model.rnn_prognosis],
                                p_state_sync=config['p_state_sync'] if config['with_state_sync'] else 0, 
                                p_diag_loss=config['p_diag_loss'] if config['with_diag_loss'] else 0,
                                p_osp_sync=0,
                                p_osp_loss=0,
                                p_tar_loss=0,
                                sync_type=config['sync_type'])
            lrn.add_cb(cb)
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

    def config_id(config): 
        return tuple([config['dl'],config['model'],config['size']])

    def update_search_space(search_space,stored_hp):
        def __inner(config,key):
            return stored_hp[config_id(config)][key]
        
        hp_entry = first(stored_hp.values())
        for key in hp_entry.keys():
            search_space.setdefault(key, tune.sample_from(partial(__inner,key=key)))

    configs_4a = pickle.load(open('configs_4A.p', 'rb'))
    configs_4b = pickle.load(open('configs_4B.p', 'rb'))
    configs_4c = pickle.load(open('configs_4C.p', 'rb'))
    
    hp_opt = HPOptimizer(create_lrn,None)
    hp_opt.start_ray()
    search_space = {
        'ablation':tune.grid_search([0,1,2]),
        'with_state_sync': tune.sample_from(lambda config: ((config['ablation'] == 1) or ((config['ablation'] == 0) and configs_4c[config_id(config)]['with_state_sync']))),
        'with_diag_loss': tune.sample_from(lambda config: ((config['ablation'] == 2) or ((config['ablation'] == 0) and configs_4c[config_id(config)]['with_diag_loss']))),

        'pred_sz': tune.grid_search([60,100,200,300,500]),
        'dl': tune.grid_search(list(dict_dls_fransys.keys())),
        'model': 'nargru',
        'size': tune.grid_search(['small','medium','large']),
        'schedule_pred': tune.grid_search([False,True]),
        'pct_start': 0.2,
        'n_epoch': 100
    }
    update_search_space(search_space,configs_4a)
    update_search_space(search_space,configs_4b)
    update_search_space(search_space,configs_4c)

    hp_opt.optimize(resources_per_trial={"gpu": 1/2 },
                    num_samples=7,
                    config=search_space,
                    metric='_rmse',
                    mode='min',
                    keep_checkpoints_num=1,
                    name='FranSys_P4C_models',
                    storage_path='~/ray_results')


# %%
