# FranSys: A Fast Non-Autoregressive Recurrent Neural Network for Multi-Step Ahead Prediction
This repository contains the code to recreate the experiments in the paper "FranSys - A Fast Non-Autoregressive Recurrent Neural Network for Multi-Step Ahead Prediction" by Daniel O.M. Weber, Clemens Gühmann, and Thomas Seel.
In addition to the code, we also provide a [pre-built Docker image](https://hub.docker.com/layers/pheenix/fransys_supplement/submission/images/sha256-7f69a6828aa79a6f19a5031134628ebe78ae32a8bb95e1b4da413c0395a345d2?tab=layers), which reproduces the experimental environment with code, dependencies, and datasets independently from changes of any software versions or package managers.

## Abstract
Neural network-based nonlinear system identification is crucial for various multi-step ahead prediction tasks, including model predictive control and digital twins. These applications demand models that are not only accurate but also efficient in training and deployment. While current state-of-the-art neural network-based methods can identify accurate models, they often become prohibitively slow when scaled to achieve high accuracy, limiting their use in resource-constrained or time-critical applications.
We propose FranSys, a Fast recurrent neural network-based method for multi-step ahead prediction in non-autoregressive System Identification. FranSys comprises three key innovations: (1) the first non-autoregressive RNN model structure for multi-step ahead prediction that enables much faster training and inference compared to autoregressive RNNs by separating state estimation and prediction into two specialized sub-models, (2) a state distribution alignment training technique that enhances generalizability and (3) a prediction horizon scheduling method that accelerates training by progressively increasing the prediction horizon.
We evaluate FranSys on three publicly available benchmark datasets representing diverse systems, comparing its speed and accuracy against state-of-the-art RNN-based multi-step ahead prediction methods. The evaluation includes various prediction horizons, model sizes, and hyperparameter optimization settings, using both our own implementations and those from related work.
Results demonstrate that FranSys is 10 to 100 times faster in training and inference with the same and often higher accuracy on test data than state-of-the-art RNN-based multi-step ahead prediction methods, particularly with long prediction horizons. This substantial speed improvement enables the application of larger neural network-based models with longer prediction horizons on resource-constrained systems in time-critical tasks, such as model predictive control and online learning of digital twins.

For more details, please refer to our paper: [link to the paper or arXiv preprint]

## Installation

This project uses our custom library called `seqdata`(further developed with the name `tsfast`), which is built on top of PyTorch and fastai for processing sequential data. To install the required dependencies and set up the environment, you can either use a container or install the software in your Python environment. For optimal reproducibility we recommend to use the [pre-built image from Docker Hub](https://hub.docker.com/layers/pheenix/fransys_supplement/submission/images/sha256-7f69a6828aa79a6f19a5031134628ebe78ae32a8bb95e1b4da413c0395a345d2?tab=layers).

### Option 1: Using Docker or Singularity

If you prefer to use containerization, you can either build the Docker image using the provided Dockerfile or use the [pre-built image from Docker Hub](https://hub.docker.com/layers/pheenix/fransys_supplement/submission/images/sha256-7f69a6828aa79a6f19a5031134628ebe78ae32a8bb95e1b4da413c0395a345d2?tab=layers):

#### Using the pre-built Docker image with Docker
Pull the pre-built image from Docker Hub:
```
docker pull pheenix/fransys_supplement:submitted
```

To run the container with Jupyter Lab, use the following command:
```
docker run -it --rm -p 8888:8888 -v /path/to/local/data:$HOME/local_data pheenix/fransys_supplement:submitted
```

Replace `/path/to/local/data` with the path to your local data directory. This directory will be mounted inside the container at `$HOME/local_data`, allowing you to access your local data files.

The container will start Jupyter Lab, which you can access by opening the provided URL in your web browser.

#### Using the pre-built Docker image with Singularity
Singularity can be used similarly with the following command:
```  
singularity run --cwd=/workspace --nv docker://pheenix/fransys_supplement:submission
```
Note that Singularity does not encapsulate the host's filesystem. 

#### Building the Docker image
If you prefer to build the docker image yourself, you can use the provided Dockerfile:

1. Clone this repository and navigate to the project directory:
 ```
 git clone https://github.com/daniel-om-weber/fransys_supplement.git
 cd fransys_supplement
 ```

2. Build the Docker image:
 ```
 docker build -t fransys_supplement .
 ```



### Option 2: Using pip

1. Clone this repository and navigate to the project directory:
 ```
 git clone https://github.com/daniel-om-weber/fransys_supplement.git
 cd fransys_supplement
 ```

2. (Optional) Create and activate the conda environment:
 It is recommended that the packages be installed in their own virtual environment. This can be created and managed with conda. 
 Make sure you have conda installed on your system. If not, you can download and install Miniconda from the official website: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

 ```
 conda create --name env_fransys python=3.10
 conda activate env_fransys
 ```

3. Install the `seqdata` library and additional dependencies:
 ```
 pip install seqdata/. jupyterlab h5py ipympl seaborn rarfile easyDataverse
 ```

If any installation steps are unclear, you can check the installation commands in the Dockerfile.


## Dataset Preparation

The code in this repository uses three publicly available benchmark datasets:
1. Ship Maneuvering Dataset (SHIP/SHIP-OOD)
2. Quadrotor Dataset (QUAD)
3. Industrial Robot Dataset (ROBOT)

The dataset preparation script `0_prepare_datasets.py` automatically downloads and preprocesses these datasets. If you are using the provided Dockerfile or the pre-built Docker image, the datasets are already prepared inside the container in the `$HOME/local_data` directory.

### [Ship Maneuvering Dataset (SHIP/SHIP-OOD)](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2905)
The Ship Maneuvering Dataset consists of simulated ship maneuvering data with environmental disturbances. It includes input signals such as propeller speed, wind speed, and rudder angles, and output signals like linear and angular velocities, roll angle, and wind attack angle. The dataset is split into training, validation, and two test sets (SHIP and SHIP-OOD, with the latter having different input and state distributions).

### [Quadrotor Dataset (QUAD)](https://github.com/wavelab/pelican_dataset)
The Quadrotor Dataset contains real-world flight data from a quadrocopter equipped with a motion capture system. The input signals are the four motor rotation speeds, and the output signals include linear and angular velocities. The dataset is split into contiguous subsets for training, validation, and testing.

### [Industrial Robot Dataset (ROBOT)](https://kluedo.ub.rptu.de/frontdoor/index/index/docId/7284)
The Industrial Robot Dataset features data from a real industrial robotic arm with six joints. The dataset provides both forward and inverse kinematics scenarios, with motor torques as input signals and joint angles as output signals. Due to the relatively short length of the dataset compared to the system complexity, it is prone to overfitting.

If you are running the code outside the container, make sure to execute the `0_prepare_datasets.py` script to download and preprocess the datasets before running the experiments.

## Experiments

This repository contains the code for reproducing the experiments presented in the paper. Each experiment has a corresponding set of Python scripts and Jupyter notebooks. The main experiments are:

1. Comparison of NAR-RNN with autoregressive RNN models (P4A)
2. Evaluation of State Distribution Alignment (P4C)
3. Evaluation of Prediction Horizon Scheduling (P4B)
4. Evaluation of FranSys on unseen test data (P4D)

### 0. Dataset Loading and Normalization
- `0_eval_all_dls.ipynb`: This notebook tests if all datasets can be loaded, performs test runs to train models, and creates a `dls_normalize.p` file with the mean and standard deviation values of each dataloader if it does not exist. The values used in the paper are already provided in the `dls_normalize.p` file.

### 1. Comparison of NAR-RNN with Autoregressive RNN Models (P4A)
- `1_P4A_hpopt.py`: This script optimizes the hyperparameters for the first experiment.
- `1_P4A_hpopt.ipynb`: This notebook evaluates the found hyperparameters and stores them in `configs_4A.p`, which is already provided with the values used in the paper.
- `2_P4A_models_no_hpopt.py`: This script trains the models for the first experiment without hyperparameter optimization.
- `3_P4A_models_with_hpopt.py`: This script trains the models for the first experiment with hyperparameter optimization.
- `3_P4A_Plots.ipynb`: This notebook creates the plots for the first experiment.

### 2. Evaluation of State Distribution Alignment (P4C)
- `6_P4C_hpopt.py`: This script optimizes the hyperparameters for the second experiment.
- `6_P4C_hpopt.ipynb`: This notebook evaluates the found hyperparameters and stores them in `configs_4C.p`.
- `7_P4C_models.py`: This script trains the models for the second experiment.
- `7_P4C_Plots.ipynb`: This notebook creates the plots for the second experiment.

### 3. Evaluation of Prediction Horizon Scheduling (P4B)
- `4_P4B_hpopt.py`: This script optimizes the hyperparameters for the third experiment.
- `4_P4B_hpopt.ipynb`: This notebook evaluates the found hyperparameters and stores them in `configs_4B.p`.
- `5_P4B_models.py`: This script trains the models for the third experiment.
- `9_P4B_ablation.py`: This script performs the detailed ablation study for the third experiment.
- `5_P4B_Plots.ipynb`: This notebook creates the plots for the third experiment.

### 4. Evaluation of FranSys on Unseen Test Data (P4D)
- `8_P4D_models.py`: This script trains the models for the fourth experiment.
- `8_P4D_Plots.ipynb`: This notebook creates the plots for the fourth experiment.

To reproduce the experiments, follow the installation instructions and ensure that the datasets are prepared. Then, run the scripts in the order specified for each experiment. The hyperparameter optimization scripts can be skipped, as the optimized hyperparameters are already provided in the respective configuration files.

Please note that the hyperparameter optimization scripts may take a long time to complete, depending on your computational resources. We parallelize these tasks using the Ray library to speed up the process. The provided configuration files contain the hyperparameters used in the paper, so you can directly use them to train the models and generate the plots.


## Citation
If you use FranSys, the prepared benchmark datasets, or the code from this repository in your research, please cite our paper:

```bibtex
@article{weber2024fransys,
  title={FranSys - A Fast Non-Autoregressive Recurrent Neural Network for Multi-Step Ahead Prediction},
  author={Weber, Daniel O.M. and Gühmann, Clemens and Seel, Thomas},
  journal={arXiv preprint arXiv:2024.XXXXX},
  year={2023}
}
```
