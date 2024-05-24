# %%
# Script for setup of the dataloaders used in this paper

from seqdata.core import *
from seqdata.models.core import *
from seqdata.learner import *
from seqdata.dataloaders import *
from seqdata.dualrnn import *
from fastai.basics import *
from fastai.callback.all import *

# from msp import *
import pickle
import h5py


def FileListSplitter(f_list):
    f_str = [str(f) for f in f_list]
    def _inner(item):
        if type(item) is dict:
            return item['path'] in f_str
        else:
            return str(item) in f_str
    return FuncSplitter(_inner)

def extract_mean_std_from_dls(dls):
    normalize = first(dls.after_batch,f=lambda x: type(x)==Normalize)
    norm_mean = normalize.mean.detach().cpu()
    norm_std = normalize.std.detach().cpu()
    return norm_mean,norm_std

def load_dls_normalize_values(key,file_name='dls_normalize.p'):
    ''' Load mean and std of dls dictionary '''

    #use the absolute path, so seperate processes refer to the same file
    f_abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),file_name)
    try:
        with open( f_abs_path, "rb" ) as f:
            dls_normalize_values = pickle.load( f )
        if key in dls_normalize_values:
            return dls_normalize_values[key]
    except OSError as e:
        print(f'{f_abs_path} not found')
    return None,None

def save_dls_normalize_values(key,value,file_name='dls_normalize.p'):
    ''' save mean and std of dls dictionary to file'''
    
    #use the absolute path, so seperate processes refer to the same file
    f_abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),file_name)
    try:
        with open( f_abs_path, "rb" ) as f:
            dls_normalize_values = pickle.load( f )
    except OSError as e:
        dls_normalize_values = {}
    dls_normalize_values[key] = value
        
    with open( f_abs_path, "wb" ) as f:
        pickle.dump(dls_normalize_values,f)

# %%
# data_path = '/mnt/data/'
# data_path = os.path.expanduser('~/data/')
data_path = Path(os.path.expanduser('~/local_data/'))

# %%
# create the file lists of all datasets

# Quadrotor Pelican dataset
f_pelican = get_hdf_files(data_path / 'pelican')
def filter_last_number(filename):
    last_number = re.findall(r'(\d+)\.', filename)
    return bool(last_number) and int(last_number[-1])

pelican_fnames_train = ['hdf5flight24.hdf5',
                        'hdf5flight25.hdf5',
                        'hdf5flight38.hdf5',
                        'hdf5flight20.hdf5',
                        'hdf5flight26.hdf5',
                        'hdf5flight14.hdf5',
                        'hdf5flight21.hdf5',
                        'hdf5flight11.hdf5',
                        'hdf5flight40.hdf5',
                        'hdf5flight9.hdf5',
                        'hdf5flight23.hdf5',
                        'hdf5flight19.hdf5',
                        'hdf5flight27.hdf5',
                        'hdf5flight12.hdf5',
                        'hdf5flight6.hdf5',
                        'hdf5flight50.hdf5',
                        'hdf5flight36.hdf5',
                        'hdf5flight48.hdf5',
                        'hdf5flight28.hdf5',
                        'hdf5flight44.hdf5',
                        'hdf5flight34.hdf5',
                        'hdf5flight32.hdf5',
                        'hdf5flight3.hdf5',
                        'hdf5flight45.hdf5',
                        'hdf5flight33.hdf5',
                        'hdf5flight4.hdf5']

pelican_fnames_valid =[  'hdf5flight10.hdf5',
                         'hdf5flight15.hdf5',
                         'hdf5flight2.hdf5',
                         'hdf5flight18.hdf5',
                         'hdf5flight51.hdf5',
                         'hdf5flight52.hdf5',
                         'hdf5flight35.hdf5',
                         'hdf5flight13.hdf5',
                         'hdf5flight22.hdf5',
                         'hdf5flight53.hdf5']

pelican_fnames_test = [  'hdf5flight8.hdf5',
                         'hdf5flight16.hdf5',
                         'hdf5flight5.hdf5',
                         'hdf5flight7.hdf5',
                         'hdf5flight41.hdf5',
                         'hdf5flight1.hdf5',
                         'hdf5flight17.hdf5',
                         'hdf5flight37.hdf5',
                         'hdf5flight30.hdf5',
                         'hdf5flight49.hdf5',
                         'hdf5flight29.hdf5',
                         'hdf5flight31.hdf5',
                         'hdf5flight39.hdf5',
                         'hdf5flight54.hdf5',
                         'hdf5flight47.hdf5',
                         'hdf5flight43.hdf5',
                         'hdf5flight42.hdf5',
                         'hdf5flight46.hdf5']
    
f_pelican_train = f_pelican.filter(lambda x: x.name in pelican_fnames_train)
f_pelican_valid = f_pelican.filter(lambda x: x.name in pelican_fnames_valid)
f_pelican_test = f_pelican.filter(lambda x: x.name in pelican_fnames_test)

# f_pelican_train = f_pelican.filter(lambda x: filter_last_number(x.name) <= 50)
# f_pelican_test = f_pelican.filter(lambda x: filter_last_number(x.name) > 50)
f_pelican_train_looper = f_pelican.filter(lambda x: '19' not in x.name)
f_pelican_test_looper = f_pelican.filter(lambda x: '19' in x.name)

# Quadrotor Physics Informed dataset
f_pi_quadrotor = get_hdf_files(data_path / 'pi_quadrotor')
pi_quadrotor_fnames_test = ['ovalz_10.hdf5','ovalz_4.hdf5','8z_5.hdf5',
                            '8z_6.hdf5','line8z_4.hdf5','wz_12.hdf5',
                            'v_8.hdf5','vT_5.hdf5']
f_pi_quadrotor_train = f_pi_quadrotor.filter(lambda x: x.name not in pi_quadrotor_fnames_test)
f_pi_quadrotor_test = f_pi_quadrotor.filter(lambda x: x.name in pi_quadrotor_fnames_test)

# ship dataset
f_ship = get_hdf_files(data_path / 'patrol_ship')
f_ship_train = f_ship.filter(lambda x: 'train' in x.parent.name)
f_ship_valid = f_ship.filter(lambda x: 'valid' in x.parent.name)
f_ship_test_ind = f_ship.filter(lambda x: 'test_ind' in x.parent.name)
f_ship_test_ood = f_ship.filter(lambda x: 'test_ood' in x.parent.name)

# robot dataset
f_robot_forward = get_hdf_files(data_path / 'robot').filter(lambda x: 'forward' in x.name)
f_robot_forward_train = f_robot_forward.filter(lambda x: 'train' in x.name)
f_robot_forward_valid = f_robot_forward.filter(lambda x: 'valid' in x.name)
f_robot_forward_test = f_robot_forward.filter(lambda x: 'test' in x.name)

f_robot_inverse = get_hdf_files(data_path / 'robot').filter(lambda x: 'inverse' in x.name)
f_robot_inverse_train = f_robot_inverse.filter(lambda x: 'train' in x.name)
f_robot_inverse_valid = f_robot_inverse.filter(lambda x: 'valid' in x.name)
f_robot_inverse_test = f_robot_inverse.filter(lambda x: 'test' in x.name)

# Wiener Hammerstein dataset
f_wienerhammerstein = get_hdf_files(data_path / 'WienerHammerstein')
f_wienerhammerstein_train = f_wienerhammerstein.filter(lambda x: 'train' in x.name)
f_wienerhammerstein_valid = f_wienerhammerstein.filter(lambda x: 'valid' in x.name)
f_wienerhammerstein_test = f_wienerhammerstein.filter(lambda x: 'test' in x.name)

# Mass Spring Damper dataset
f_mass_spring_undamped = get_hdf_files(data_path / 'mass_spring_undampened')
f_mass_spring_undamped_train = f_mass_spring_undamped[:-3]
f_mass_spring_undamped_test = f_mass_spring_undamped[-3:]

f_mass_spring_damped = get_hdf_files(data_path / 'mass_spring_damped')
f_mass_spring_damped_train = f_mass_spring_damped[:-1]
f_mass_spring_damped_test = f_mass_spring_damped[-1:]

# Battery dataset
f_battery = get_hdf_files(data_path / 'SPMe_artificial',recurse=True)
f_battery_cells_train = f_battery.filter(lambda x: int(x.parent.parent.name[-2:]) <= 28)
f_battery_cells_valid = f_battery.filter(lambda x: int(x.parent.parent.name[-2:]) > 28)
f_battery_cells_train_qocv = f_battery_cells_train.filter(lambda x: x.parent.name == 'qocv' and int(x.name[-9:-5])< 5) 
f_battery_cells_train_driving = f_battery_cells_train.filter(lambda x: x.parent.name == 'training') 
f_battery_cells_train_cc_discharge = f_battery_cells_train.filter(lambda x: x.parent.name == 'discharge' and int(x.name[-9:-5]) < 5)
f_battery_train = f_battery_cells_train_qocv + f_battery_cells_train_driving
f_battery_valid = f_battery_cells_valid.filter(lambda x: x.parent.name == 'training') 

# %% contructors for the dataloaders
def create_dls_prediction(
        u, #list of input signal names
        y, #list of output signal names
        item_list, #list of hdf files
        x=[], #optional list of state signal names
        init_sz = 10, #initial window size
        pred_sz = 190, #prediction window size
        stp_sz = 1, #step size between consecutive windows
        bs = 128, #batch size
        valid_splitter = RandomSplitter(0.4), #splitter for the validation set
        valid_stp_sz = None, #step size between consecutive validation windows
        input_delay = False, #if true, the input is delayed by one step
        cached = True, #if true, the data is cached in RAM
        num_workers = 5, #number of processes for the dataloader, 0 for no multiprocessing
        max_batches_training = 300, #limits the number of  training batches in a single epoch
        max_batches_valid = None, #limits the number of validation batches in a single epoch
        dls_id = None #identifier for the dataloader to load the normalization values
    ):
    win_sz = init_sz+pred_sz+1#+1 to compensate for the shifting of the input
    if valid_stp_sz is None: valid_stp_sz = pred_sz
        
    #identify dls by variable and file names
    if dls_id is None: dls_id = (str(u+x+y),str([f.name for f in item_list]))
    (norm_mean,norm_std) = load_dls_normalize_values(dls_id)

    if input_delay: #if true, the input is delayed by one step
        blocks = (SequenceBlock.from_hdf(u+x+y,TensorSequencesInput,clm_shift=[-1]*len(u+x+y),cached=cached),
                    SequenceBlock.from_hdf(y,TensorSequencesOutput,clm_shift=[1]*len(y),cached=cached))
    else:
        blocks = (SequenceBlock.from_hdf(u+x+y,TensorSequencesInput,clm_shift=([0]*len(u)+[-1]*len(x+y)),cached=cached),
                    SequenceBlock.from_hdf(y,TensorSequencesOutput,clm_shift=[1]*len(y),cached=cached))
    
    dl_kwargs=[{'max_batches':max_batches_training},{'max_batches':max_batches_valid}]
    seq = DataBlock(blocks=blocks,
                     get_items=CreateDict([DfApplyFuncSplit(
                            valid_splitter,
                            DfHDFCreateWindows(win_sz=win_sz,stp_sz=1,clm=u[0]),
                            DfHDFCreateWindows(win_sz=win_sz,stp_sz=valid_stp_sz,clm=u[0])
                        )]),
                     batch_tfms=Normalize(mean=norm_mean,std=norm_std,axes=(0, 1)),
                     splitter=valid_splitter)
    dls = seq.dataloaders(item_list,bs=bs,num_workers=num_workers,
                          dl_type=BatchLimit_Factory(TfmdDL),dl_kwargs=dl_kwargs)
    if norm_mean is None:
        (norm_mean,norm_std) = extract_mean_std_from_dls(dls)
        save_dls_normalize_values(dls_id,(norm_mean,norm_std))
    return dls


def create_dls_simulation(
        u, #list of input signal names
        y, #list of output signal names
        item_list, #list of hdf files
        win_sz = 100, #initial window size
        stp_sz = 1, #step size between consecutive windows
        bs = 128, #batch size
        valid_splitter = RandomSplitter(0.4), #splitter for the validation set
        cached = True, #if true, the data is cached in RAM
        num_workers = 3, #number of processes for the dataloader, 0 for no multiprocessing
        max_batches_training = 300, #limits the number of  training batches in a single epoch
        max_batches_valid = 30 #limits the number of validation batches in a single epoch
    ):
    blocks = (SequenceBlock.from_hdf(u,TensorSequencesInput,cached=cached),
            SequenceBlock.from_hdf(y,TensorSequencesOutput,cached=cached))
        
    #identify dls by variable and file names
    dls_id = (str(u),str([f.name for f in item_list]))
    (norm_mean,norm_std) = load_dls_normalize_values(dls_id)
    
    dl_kwargs=[{'max_batches':max_batches_training},{'max_batches':max_batches_valid}]
    seq = DataBlock(blocks=blocks,
                     get_items=CreateDict([DfApplyFuncSplit(
                            valid_splitter,
                            DfHDFCreateWindows(win_sz=win_sz,stp_sz=1,clm=u[0]),
                            DfHDFCreateWindows(win_sz=win_sz,stp_sz=win_sz,clm=u[0])
                        )]),
                     batch_tfms=Normalize(mean=norm_mean,std=norm_std,axes=(0, 1)),
                     splitter=valid_splitter)
    dls = seq.dataloaders(item_list,bs=bs,num_workers=num_workers,
                          dl_type=BatchLimit_Factory(TfmdDL),dl_kwargs=dl_kwargs)
        
    if norm_mean is None:
        (norm_mean,norm_std) = extract_mean_std_from_dls(dls)
        save_dls_normalize_values(dls_id,(norm_mean,norm_std))
    return dls

# %% Quadrotor Pelican Dataloader
pelican_u_motors = [f'motors{i}' for i in range(1,4+1)]
pelican_u_motors_cmd = [f'motors_cmd{i}' for i in range(1,4+1)]
pelican_y_euler = [f'euler{i}' for i in range(1,3+1)]
pelican_y_euler_rates = [f'euler_rates{i}' for i in range(1,3+1)]
pelican_y_pos = [f'pos{i}' for i in range(1,3+1)]
pelican_y_vel = [f'vel{i}' for i in range(1,3+1)]
pelican_y_rate = [f'pqr{i}' for i in range(1,3+1)]

# Our Own Quadrotor Pelican Data Split
dls_config_pelican_prediction = {'u':pelican_u_motors,'y':pelican_y_euler_rates+pelican_y_vel,
                        'item_list':f_pelican_train+f_pelican_valid,'valid_splitter':FileListSplitter(f_pelican_valid),
                        'init_sz':50,'pred_sz':190,'stp_sz':1,'valid_stp_sz':40,'bs':128,'dls_id':'pelican_prediction'}
create_dls_pelican_prediction = partial(create_dls_prediction, **dls_config_pelican_prediction)

dls_config_pelican_prediction_test = {'u':pelican_u_motors,'y':pelican_y_euler_rates+pelican_y_vel,
                        'item_list':f_pelican,'valid_splitter':FileListSplitter(f_pelican_test),
                        'init_sz':50,'pred_sz':190,'stp_sz':1,'valid_stp_sz':190,'bs':256,'dls_id':'pelican_prediction'}
create_dls_pelican_prediction_test = partial(create_dls_prediction, **dls_config_pelican_prediction_test)

# Mohajerin et al. Quadrotor Pelican Data Split
dls_config_pelican_mohajerin_prediction = {'u':pelican_u_motors,'y':pelican_y_euler_rates+pelican_y_vel,
                        'item_list':f_pelican,'valid_splitter':RandomSplitter(0.4),
                        'init_sz':10,'pred_sz':190,'stp_sz':1,'valid_stp_sz':40,'bs':128,'dls_id':'pelican_prediction'}
create_dls_pelican_mohajerin_prediction = partial(create_dls_prediction, **dls_config_pelican_mohajerin_prediction)

# Looper et al. Quadrotor Pelican Data Split
dls_config_pelican_looper_prediction = {'u':pelican_u_motors_cmd,'y':pelican_y_rate+pelican_y_vel,'x':pelican_y_pos+pelican_y_euler,
                        'item_list':f_pelican,'valid_splitter':FileListSplitter(f_pelican_test_looper),
                        'init_sz':10,'pred_sz':90,'stp_sz':90,'valid_stp_sz':40,'bs':16,'dls_id':'pelican_prediction'}
create_dls_pelican_looper_prediction = partial(create_dls_prediction, **dls_config_pelican_looper_prediction)

# %%

def get_dls_parameters(func):
    default_parameters = func.keywords
    n_u = len(default_parameters['u'])
    n_y = len(default_parameters['y'])
    n_x = len(default_parameters['x']) if 'x' in default_parameters else 0
    init_sz = default_parameters['init_sz'] if 'init_sz' in default_parameters else default_parameters['win_sz']
    pred_sz = default_parameters['pred_sz'] if 'pred_sz' in default_parameters else default_parameters['win_sz']

    return n_u, n_y, n_x, init_sz, pred_sz
# %% Quadrotor Physics Informed Dataloader

quadrotor_pi_u = ['u_0','u_1','u_2','u_3']

quadrotor_pi_x_v = ['v_x','v_y','v_z']
quadrotor_pi_x_q = ['q_w','q_x','q_y','q_z']
quadrotor_pi_x_w = ['w_x','w_y','w_z']
quadrotor_pi_x = quadrotor_pi_x_v + quadrotor_pi_x_q + quadrotor_pi_x_w

quadrotor_pi_y_vdot = ['vdot_x','vdot_y','vdot_z']
quadrotor_pi_y_wdot = ['wdot_x','wdot_y','wdot_z']
quadrotor_pi_y = quadrotor_pi_y_vdot + quadrotor_pi_y_wdot

dls_config_quadrotor_pi_prediction = {'u':quadrotor_pi_u,'y':quadrotor_pi_y,'x':quadrotor_pi_x,
                        'item_list':f_pi_quadrotor,'valid_splitter':FileListSplitter(f_pi_quadrotor_test),
                        'init_sz':50,'pred_sz':500,'stp_sz':1,'bs':128,'dls_id':'quad_pi_prediction'}
create_dls_quadrotor_pi_prediction = partial(create_dls_prediction, **dls_config_quadrotor_pi_prediction)

# %% Patrol Ship Dataloader
ship_u = ['n','deltal','deltar','Vw']
ship_y = ['alpha_x','alpha_y','u','v','p','r','phi']
ship_y_hybrid =  ['u','v','p','r','phi']

dls_config_ship_prediction = {'u':ship_u,'y':ship_y,
                        'item_list':f_ship_train+f_ship_valid,'valid_splitter':FileListSplitter(f_ship_valid),
                        'init_sz':60,'pred_sz':60,'stp_sz':1,'valid_stp_sz':20,'bs':64,'dls_id':'ship_prediction'}
create_dls_ship_prediction = partial(create_dls_prediction, **dls_config_ship_prediction)


dls_config_ship_prediction_test_ind = {'u':ship_u,'y':ship_y,
                        'item_list':f_ship_train+f_ship_valid+f_ship_test_ind,'valid_splitter':FileListSplitter(f_ship_test_ind),
                        'init_sz':60,'pred_sz':60,'stp_sz':1,'valid_stp_sz':60,'bs':256,'dls_id':'ship_prediction'}
create_dls_ship_prediction_test_ind = partial(create_dls_prediction, **dls_config_ship_prediction_test_ind)

dls_config_ship_prediction_test_ood = {'u':ship_u,'y':ship_y,
                        'item_list':f_ship_train+f_ship_valid+f_ship_test_ood,'valid_splitter':FileListSplitter(f_ship_test_ood),
                        'init_sz':60,'pred_sz':60,'stp_sz':1,'valid_stp_sz':60,'bs':256,'dls_id':'ship_prediction'}
create_dls_ship_prediction_test_ood = partial(create_dls_prediction, **dls_config_ship_prediction_test_ood)

dls_config_ship_prediction_hybrid = {'u':ship_u,'y':ship_y_hybrid,
                        'item_list':f_ship_train+f_ship_valid,'valid_splitter':FileListSplitter(f_ship_valid),
                        'init_sz':60,'pred_sz':900,'stp_sz':1,'valid_stp_sz':20,'bs':64,'dls_id':'ship_hybrid_prediction'}
create_dls_ship_prediction_hybrid = partial(create_dls_prediction, **dls_config_ship_prediction_hybrid)

dls_config_ship_prediction_hybrid_test = {'u':ship_u,'y':ship_y_hybrid,
                        'item_list':f_ship_train+f_ship_valid+f_ship_test_ind,'valid_splitter':FileListSplitter(f_ship_test_ind),
                        'init_sz':60,'pred_sz':900,'stp_sz':1,'valid_stp_sz':1,'bs':64,'dls_id':'ship_hybrid_prediction'}
create_dls_ship_prediction_hybrid_test = partial(create_dls_prediction, **dls_config_ship_prediction_hybrid_test)

# %% Robot Dataloader
robot_u = [f'u{i}' for i in range(6)]
robot_u_inverse = [f'u{i}' for i in range(18)]
robot_y = [f'y{i}' for i in range(6)]

dls_config_robot_forward_prediction = {'u':robot_u,'y':robot_y,
                        'item_list':f_robot_forward_train+f_robot_forward_valid,'valid_splitter':FileListSplitter(f_robot_forward_valid),
                        'init_sz':100,'pred_sz':150,'stp_sz':1,'valid_stp_sz':4,'bs':128,'dls_id':'robot_forward_prediction'}
create_dls_robot_forward_prediction = partial(create_dls_prediction, **dls_config_robot_forward_prediction)

dls_config_robot_forward_prediction_long = {'u':robot_u,'y':robot_y,
                        'item_list':f_robot_forward_train+f_robot_forward_valid,'valid_splitter':FileListSplitter(f_robot_forward_valid),
                        'init_sz':100,'pred_sz':1000,'stp_sz':1,'valid_stp_sz':4,'bs':128,'dls_id':'robot_forward_prediction'}
create_dls_robot_forward_prediction_long = partial(create_dls_prediction, **dls_config_robot_forward_prediction_long)

dls_config_robot_forward_prediction_test = {'u':robot_u,'y':robot_y,
                        'item_list':f_robot_forward,'valid_splitter':FileListSplitter(f_robot_forward_test),
                        'init_sz':100,'pred_sz':150,'stp_sz':1,'valid_stp_sz':150,'bs':256,'dls_id':'robot_forward_prediction'}
create_dls_robot_forward_prediction_test = partial(create_dls_prediction, **dls_config_robot_forward_prediction_test)

dls_config_robot_forward_simulation = {'u':robot_u,'y':robot_y,
                        'item_list':f_robot_forward,'valid_splitter':FileListSplitter(f_robot_forward_test),
                        'win_sz':300,'stp_sz':1,'valid_stp_sz':4,'bs':128,'dls_id':'robot_forward_simulation'}
create_dls_robot_forward_simulation = partial(create_dls_simulation, **dls_config_robot_forward_simulation)


dls_config_robot_inverse_prediction = {'u':robot_u_inverse,'y':robot_y,
                        'item_list':f_robot_inverse,'valid_splitter':FileListSplitter(f_robot_inverse_test),
                        'init_sz':100,'pred_sz':150,'stp_sz':1,'valid_stp_sz':4,'bs':128,'dls_id':'robot_inverse_prediction'}
create_dls_robot_inverse_prediction = partial(create_dls_prediction, **dls_config_robot_inverse_prediction)

# %% Wiener Hammerstein Dataloader

dls_config_wiener_hammerstein_prediction = {'u':['u'],'y':['y'],
                        'item_list':f_wienerhammerstein_train+f_wienerhammerstein_valid,
                        'valid_splitter':FileListSplitter(f_wienerhammerstein_valid),
                        'init_sz':100,'pred_sz':400,'stp_sz':1,'bs':128,'dls_id':'wh_prediction'}
create_dls_wiener_hammerstein_prediction = partial(create_dls_prediction, **dls_config_wiener_hammerstein_prediction)

dls_config_wiener_hammerstein_prediction_test = {'u':['u'],'y':['y'],
                        'item_list':f_wienerhammerstein_train+f_wienerhammerstein_valid+f_wienerhammerstein_test,
                        'valid_splitter':FileListSplitter(f_wienerhammerstein_test),
                        'init_sz':100,'pred_sz':400,'stp_sz':1,'bs':128,'dls_id':'wh_prediction'}
create_dls_wiener_hammerstein_prediction_test = partial(create_dls_prediction, **dls_config_wiener_hammerstein_prediction_test)

dls_config_wiener_hammerstein_simulation = {'u':['u'],'y':['y'],
                        'item_list':f_wienerhammerstein_train+f_wienerhammerstein_valid,
                        'valid_splitter':FileListSplitter(f_wienerhammerstein_valid),
                        'win_sz':400,'stp_sz':1,'bs':128,'dls_id':'wh_simulation'}
create_dls_wiener_hammerstein_simulation = partial(create_dls_simulation, **dls_config_wiener_hammerstein_simulation)

# %% Mass Spring Damped Dataloader

mass_spring_u = ['u(t)']
mass_spring_y = ['x(t)']

dls_config_mass_spring_damped_prediction = {'u':mass_spring_u,'y':mass_spring_y,
                        'item_list':f_mass_spring_damped,
                        'valid_splitter':FileListSplitter(f_mass_spring_damped_test),
                        'init_sz':100,'pred_sz':300,'stp_sz':200,'bs':128}
create_dls_mass_spring_damped_prediction = partial(create_dls_prediction, **dls_config_mass_spring_damped_prediction)

dls_config_mass_spring_damped_simulation = {'u':mass_spring_u,'y':mass_spring_y,
                        'item_list':f_mass_spring_damped,
                        'valid_splitter':FileListSplitter(f_mass_spring_damped_test),
                        'win_sz':400,'stp_sz':200,'bs':128}
create_dls_mass_spring_damped_simulation = partial(create_dls_simulation, **dls_config_mass_spring_damped_simulation)

dls_config_mass_spring_undamped_prediction = {'u':mass_spring_u,'y':mass_spring_y,
                        'item_list':f_mass_spring_undamped,
                        'valid_splitter':FileListSplitter(f_mass_spring_undamped_test),
                        'init_sz':100,'pred_sz':300,'stp_sz':200,'bs':128}
create_dls_mass_spring_undamped_prediction = partial(create_dls_prediction, **dls_config_mass_spring_undamped_prediction)

dls_config_mass_spring_undamped_simulation = {'u':mass_spring_u,'y':mass_spring_y,
                        'item_list':f_mass_spring_undamped,
                        'valid_splitter':FileListSplitter(f_mass_spring_undamped_test),
                        'win_sz':400,'stp_sz':200,'bs':128}
create_dls_mass_spring_undamped_simulation = partial(create_dls_simulation, **dls_config_mass_spring_undamped_simulation)

# %% Battery Dataloader
battery_u = ['current']
battery_y = ['voltage']

dls_config_battery_prediction = {'u':battery_u,'y':battery_y,
                        'item_list':f_battery_train+f_battery_valid,
                        'valid_splitter':FileListSplitter(f_battery_valid),
                        'init_sz':200,'pred_sz':2000,'stp_sz':200,'bs':64,'dls_id':'battery_prediction'}
create_dls_battery_prediction = partial(create_dls_prediction, **dls_config_battery_prediction)

# %% Dictionary of all dataloaders

dict_dls_comparison = {
    'pelican':create_dls_pelican_prediction,
    'quadrotor_pi':create_dls_quadrotor_pi_prediction,
    'robot_forward':create_dls_robot_forward_prediction,
    'robot_inverse':create_dls_robot_inverse_prediction,
    'wiener_hammerstein':create_dls_wiener_hammerstein_prediction,
    'mass_spring_damped':create_dls_mass_spring_damped_prediction,
    'mass_spring_undamped':create_dls_mass_spring_undamped_prediction,
    'battery':create_dls_battery_prediction
}

dict_dls_prediction = {
    'pelican':create_dls_pelican_prediction,
    'pelican_mohajerin':create_dls_pelican_mohajerin_prediction,
    'pelican_looper':create_dls_pelican_looper_prediction,
    'quadrotor_pi':create_dls_quadrotor_pi_prediction,
    'robot_forward':create_dls_robot_forward_prediction,
    'robot_inverse':create_dls_robot_inverse_prediction,
    'wiener_hammerstein':create_dls_wiener_hammerstein_prediction,
    'mass_spring_damped':create_dls_mass_spring_damped_prediction,
    'mass_spring_undamped':create_dls_mass_spring_undamped_prediction,
    'Ship':create_dls_ship_prediction,
    'battery':create_dls_battery_prediction
}
dict_dls_fransys = {
    'Quadrotor':create_dls_pelican_prediction,
    'Robot':create_dls_robot_forward_prediction,
    'Ship':create_dls_ship_prediction,
}
dict_dls_fransys_test = {
    'Quadrotor':create_dls_pelican_prediction_test,
    'Robot':create_dls_robot_forward_prediction_test,
    'Ship':create_dls_ship_prediction_test_ind,
}

dict_dls_simulation = {
    'robot_forward':create_dls_robot_forward_simulation,
    'wiener_hammerstein':create_dls_wiener_hammerstein_simulation,
    'mass_spring_damped':create_dls_mass_spring_damped_simulation,
    'mass_spring_undamped':create_dls_mass_spring_undamped_simulation
}