# %%
import scipy.io as sio
import os
import h5py
import numpy as np

# %%
import requests,io,os
import zipfile,rarfile
from pathlib import Path

def unzip_download(url,extract_dir = '.'):
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_dir)
        
def unrar_download(url,extract_dir = '.'):
    response = requests.get(url)
    with rarfile.RarFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_dir)
        
def download(url,target_dir = '.'):
    fname = Path(url).name
    if os.path.isfile(fname): return
    response = requests.get(url)
    p_name = Path(target_dir).joinpath(fname)
    with open(p_name, "wb") as file:
        file.write(response.content)
    return p_name

# %%
data_path = Path(os.path.expanduser('~/local_data/'))

# %% [markdown]
# # Ship

# %%
from easyDataverse import Dataverse

download_dir = 'ship_downloaded'
target_dir = data_path / 'patrol_ship'

dataverse = Dataverse('https://darus.uni-stuttgart.de/')
dataset = dataverse.load_dataset(
    pid='doi:10.18419/darus-2905',
    filedir=download_dir,
)

# %%
import shutil
import pandas as pd

structure_mapping = {
    'patrol_ship_routine/processed/train': 'train',
    'patrol_ship_routine/processed/validation': 'validation',
    'patrol_ship_routine/processed/test': 'test_ind',
    'patrol_ship_ood/processed/test': 'test_ood'
}

# Ensure desired directories exist
for subdir in structure_mapping.values():
    os.makedirs(os.path.join(target_dir, subdir), exist_ok=True)

def convert_tab_to_hdf5(tab_path, hdf5_path):
    df = pd.read_csv(tab_path, sep='\t')  # Adjust the separator as needed
    with h5py.File(hdf5_path, 'w') as hdf:
        for column in df.columns:
            data = df[column].astype(np.float32).values
            hdf.create_dataset(column, data=data, dtype='f4')

# Walk through the current directory structure and process files
for subdir, dirs, files in os.walk(download_dir):
    for file in files:
        if file.endswith('.tab'):
            current_file_path = os.path.join(subdir, file)
            
            # Determine the relative path
            relative_subdir = os.path.relpath(subdir, download_dir)
            
            # Find the corresponding desired subdir
            if relative_subdir in structure_mapping:
                desired_subdir = structure_mapping[relative_subdir]
                
                # Construct desired file paths
                base_filename = file.replace('.tab', '')
                desired_hdf5_path = os.path.join(target_dir, desired_subdir, base_filename + '.hdf5')
                
                convert_tab_to_hdf5(current_file_path, desired_hdf5_path)
#remove downloaded files
shutil.rmtree(download_dir)

# %% [markdown]
# # Quadrotor

# %%
url_pelican = 'http://wavelab.uwaterloo.ca/wp-content/uploads/2017/09/AscTec_Pelican_Flight_Dataset.mat'
downloaded_fname = download(url_pelican)

# %%
target_dir = data_path / 'pelican/'

def write_signal(fname, sname, signal):
    with h5py.File(fname, 'a') as f:
        for i in range(signal.shape[1]):
            ds_name = f'{sname}{i+1}'
            sig = signal[:, i]
            f.create_dataset(ds_name, data=sig, dtype='float64', chunks=(1000,))

flight_data = sio.loadmat(downloaded_fname,simplify_cells=True)
flights = flight_data['flights']
os.makedirs(target_dir, exist_ok=True)

for k, flight in enumerate(flights, start=1):
    fname = os.path.join(target_dir, f'hdf5flight{k}.hdf5')
    
    if os.path.exists(fname):
        os.remove(fname)
    
    write_signal(fname, 'vel', flight['Vel'])
    write_signal(fname, 'pos', flight['Pos'][1:, :])
    write_signal(fname, 'euler', flight['Euler'][1:, :])
    write_signal(fname, 'euler_rates', flight['Euler_Rates'])
    write_signal(fname, 'motors', flight['Motors'][1:, :])
    write_signal(fname, 'motors_cmd', flight['Motors_CMD'][1:, :])
    write_signal(fname, 'pqr', flight['pqr'][:-1, :])

# %%
#cleanup downloaded quadrotor file
os.remove(downloaded_fname)

# %% [markdown]
# # Robot

# %%
url_robot = "https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/Robot_Identification_Benchmark_Without_Raw_Data.rar"
unrar_download(url_robot)

# %%
target_dir = data_path / 'robot/'
os.makedirs(target_dir, exist_ok=True)

train_valid_split = 0.8

path_forward = "./forward_identification_without_raw_data.mat"
path_inverse = "./inverse_identification_without_raw_data.mat"

fs = 10  # Hz

def w_ds(group, ds_name, data, dtype='f4', chunks=None):
    group.create_dataset(ds_name, data=data, dtype=dtype, chunks=chunks)

def write_array(group, ds_name: str, data: np.array, dtype='f4', chunks=None) -> None:
    for i in range(data.shape[0]):
        w_ds(group, f'{ds_name}{i}', data[i], dtype, chunks)

# Convert the matlab sequences to hdf5 files
for idx, path in enumerate([path_forward, path_inverse]):
    mf = sio.loadmat(path)
    for mode in ['train', 'test']:
        if mode == 'test':
            with h5py.File(target_dir / f'{"forward" if idx == 0 else "inverse"}_{mode}.hdf5', 'w') as f:
                w_ds(f, 'dt', np.ones_like(mf[f'time_{mode}'][0]) / fs)
                write_array(f, 'u', mf[f'u_{mode}'])
                write_array(f, 'y', mf[f'y_{mode}'])
        else:
            with h5py.File(target_dir / f'{"forward" if idx == 0 else "inverse"}_train.hdf5', 'w') as train_f, \
                 h5py.File(target_dir / f'{"forward" if idx == 0 else "inverse"}_valid.hdf5', 'w') as valid_f:
                    dt = np.ones_like(mf[f'time_{mode}'][0]) / fs
                    total_entries = len(dt)
                    split_index = int(total_entries * train_valid_split)

                    w_ds(train_f, 'dt', dt[:split_index])
                    write_array(train_f, 'u', mf[f'u_{mode}'][:,:split_index])
                    write_array(train_f, 'y', mf[f'y_{mode}'][:,:split_index])
                     
                    w_ds(valid_f, 'dt', dt[split_index:])
                    write_array(valid_f, 'u', mf[f'u_{mode}'][:,split_index:])
                    write_array(valid_f, 'y', mf[f'y_{mode}'][:,split_index:])

# %%
#cleanup downloaded robot files 
os.remove(path_forward)
os.remove(path_inverse)

# %%



