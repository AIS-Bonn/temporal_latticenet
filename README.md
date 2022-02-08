# Temporal LatticeNet

This repository is based on the lattice_net repository from Radu Alexandru Rosu: https://github.com/RaduAlexandru/lattice_net .

## Requirements

The easiest way to install Temporal LatticeNet is using the included dockerfile.
You will need to have Docker>=19.03 and nvidia drivers (https://github.com/NVIDIA/nvidia-docker) installed.
Afterwards, you can build the docker image which contains all the LatticeNet dependencies using:

## Docker

```sh
$ git clone --recursive https://github.com/AIS-Bonn/temporal_latticenet.git
$ cd temporal_latticenet/seq_docker
$ ./build.sh temporal_lattice_img #this will take some time because some packages need to be build from source
$ ./run.sh temporal_lattice_img  
$ git clone --recursive https://github.com/RaduAlexandru/easy_pbr
$ cd easy_pbr && make && cd ..
$ git clone --recursive https://github.com/RaduAlexandru/data_loaders 
$ cd data_loaders && make && cd ..
$ git clone --recursive --branch sequenceLearning https://github.com/RaduAlexandru/lattice_net
$ cd lattice_net && make && cd ..
```

The folder *temporal_latticenet* is mounted during run, so that changes inside the docker are reflected to the folder. 
If you don't want this you have to change the *run.sh*.

Additionally you can mount the datasets you want to use. You can change that in the *run.sh*.

### SSH inside Docker

Some dependent repositories need an ssh key, because they rely on ssh cloning:

```sh
$ ssh-keygen -t ed25519 -C "your@example.com" #generate the new ssh key inside the docker and add it to your github account
$ eval "$(ssh-agent -s)" # activate ssh agent
$ ssh-add ~/.ssh/id_ed25519  # add the key to the ssh-agent
```

## Defining the Network

The config files (**.cfg**) in the folder *seq_config/* manage the different modes of the network. We have different ones for the training and testing.
The bool **sequence_learning** defines if the network should use temporal dependencies or not. 
Meanwhile in **rnn_modules** is a array with four entries, where each entry is one of the following strings: Linear/MaxPool/CGA/CLI/LSTM/GRU/None . Our AFlow module is still called CLI in this implementation. 

## Datasets

Currently only [SemanticKITTI](http://www.semantic-kitti.org/) is supported.            
[Paris-Lille-3D](https://npm3d.fr/paris-lille-3d) was implemented aswell, but the training was not pursued, due to problems with the dataset.  


## Pretrained Models

The pretrained models can be found in the folder *pretrained_models*. To load them you have to change the **load_checkpoint_model** in the config file. The script will then print out a message confirming it. Additionally, you have to set the correct sigma (0.6), load the correct model (the one for sigma 0.6), the correct values_mode, the correct rnn_modules and the correct sequence definition.

The **best models** were:                 
With reflectance as input:                            
GRU-GRU-AFlow-GRU: *model_moving_setKitti_sigma0.6_typegru-gru-cli-gru_frames4_scope3_epoch_2.pt*                     
GRU-GRU-AFlow-AFLOW: *model_moving_setKitti_sigma0.6_typegru-gru-cli-cli_frames4_scope3_epoch_2.pt*                          
LSTM-LSTM-AFlow-LSTM: *model_moving_setKitti_sigma0.6_typelstm-lstm-cli-lstm_frames4_scope3_epoch_2.pt*                     
GRU-GRU-/-GRU: *model_moving_setKitti_sigma0.6_typegru-gru-none-gru_frames4_scope3_epoch_2.pt*                    

Without reflectance as input:                       
GRU-GRU-AFlow-GRU: *noREF_model_moving_setKitti_sigma0.6_typegru-gru-cli-gru_frames4_scope3_epoch_2.pt*                     