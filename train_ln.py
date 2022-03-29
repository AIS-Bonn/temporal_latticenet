#!/usr/bin/env python3.6

# debugging
#import tracemalloc
#from tracemalloc import Filter
#tracemalloc.start()
#import faulthandler # for debugging
import torch

import sys, os, argparse, time
from tqdm import tqdm

from dataloader.kitti_dataloader import *
from dataloader.parisLille_dataloader import *
from easypbr  import *
from latticenet  import ModelParams
from latticenet_py.lattice.lovasz_loss import LovaszSoftmax

from callbacks.callback import *
from callbacks.viewer_callback import *
from callbacks.visdom_callback import *
from callbacks.state_callback import *
from callbacks.wandb_callback import *
from callbacks.phase import *

from cfgParser import *
from pathlib import Path
from seq_lattice.models import *

class CloudReadingException(Exception):
    pass

from datetime import datetime

#torch.manual_seed(0)
#torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(edgeitems=3)
wandb_entity = "peerschuett"
experiment_name = "temporal_latticenet_tests"

# train_border and valid_border are integers, that define how many clouds are skipped, e.g. train_border = 6 means we start at the sixth cloud
def create_loader(dataset_name, config_parser, sequence_learning = False, shuffle = False, train_border = 0, valid_border = 0):
    if(dataset_name=="semantickitti"):
        train_dataset = SemanticKittiDataset(split = "train", config_parser = config_parser, sequence_learning = sequence_learning)
        valid_dataset = SemanticKittiDataset(split = "valid", config_parser = config_parser, sequence_learning = sequence_learning)
    elif(dataset_name=="parislille"):
        train_dataset = ParisLille3DDataset(split = "train", config_parser = config_parser, sequence_learning = sequence_learning)
        valid_dataset = ParisLille3DDataset(split = "valid", config_parser = config_parser, sequence_learning = sequence_learning)
    else:
        sys.exit("Dataset name not recognized. It is " + dataset_name)
    
    train_sampler = list(range(len(train_dataset)))[train_border:] if train_border > 0 else None
    valid_sampler = list(range(len(valid_dataset)))[valid_border:] if valid_border > 0 else None
    #shuffle = False if train_border > 0 else shuffle

    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers = 8, batch_size=1, shuffle = shuffle, sampler = train_sampler)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, num_workers = 8, batch_size=1, shuffle = False, sampler = valid_sampler)

    return train_dataloader, valid_dataloader, train_dataset, valid_dataset


def run(dataset_name = "semantickitti"):
    cwd = os.getcwd()
    if dataset_name == "semantickitti":
        print("\n-------- Using SemanticKitti Dataset --------")
        config_file=cwd+"/seq_config/lnn_train_semantic_kitti.cfg"
        print("Config file: ", config_file)
    elif(dataset_name=="parislille"):
        sys.exit("Currently ParisLille3D isn't supported!")
        print("\n-------- Using ParisLille3D Dataset --------")
        config_file=cwd+"/seq_config/lnn_train_paris_lille.cfg"
    else:
        sys.exit("Dataset name not recognized. It is {}. Available options are semantickitti or parislille.".format(dataset_name) )

    if not torch.cuda.is_available():
        sys.exit("The GPU is not available!")
        
    # Read the config file
    config_parser = cfgParser(config_file)
    model_params=ModelParams.create(config_file)  
    loader_params = config_parser.get_loader_vars()
    label_mngr_params = config_parser.get_label_mngr_vars()
    model_config = config_parser.get_model_vars()
    train_config = config_parser.get_train_vars()
    lattice_gpu_config = config_parser.get_lattice_gpu_vars()
    loader_config = config_parser.get_loader_vars()

    # Print some nice information
    print("Lattice sigma: ", str(lattice_gpu_config["sigma_0"])[0:3])
    print("Sequences: #scans: {}, cloud scope: {}".format((loader_config['frames_per_seq'] if model_config["sequence_learning"] else 1), loader_config['cloud_scope']))
    print("Features: ", model_config["values_mode"])
    if train_config["save_checkpoint"]:
        Path(train_config["checkpoint_path"]).mkdir(parents=True, exist_ok=True)
        print("The checkpoints will be saved to: ", str(train_config["checkpoint_path"]))

    # initialize the LabelMngr and the viewer
    m_ignore_index = label_mngr_params["unlabeled_idx"]
    labels_file=str(label_mngr_params["labels_file"])
    colorscheme_file=str(label_mngr_params["color_scheme_file"])
    frequency_file=str(label_mngr_params["frequency_file_all"]) if loader_params["include_moving_classes"] else  str(label_mngr_params["frequency_file"])
    label_mngr=LabelMngr(labels_file, colorscheme_file, frequency_file, m_ignore_index )
    if train_config["with_viewer"]:
        view=Viewer.create(config_file)

    # Initialize the networks model
    lattice=Lattice.create(config_file, "lattice") # create Lattice
    model = None
    if not loader_params["include_moving_classes"] and (train_config["dataset_name"] == "semantickitti"):
        model=LNN_SEQ(20, model_params, config_parser).to("cuda")
    elif (train_config["dataset_name"] == "semantickitti"):
        #print("Including moving classes - therefore 26 classes")
        model=LNN_SEQ(26, model_params, config_parser).to("cuda")
    elif not loader_params["include_moving_classes"] and (train_config["dataset_name"] == "parislille"):
        model=LNN_SEQ(10, model_params, config_parser).to("cuda") # parislille has only 10 classes
    elif (train_config["dataset_name"] == "parislille"):
        model=LNN_SEQ(12, model_params, config_parser).to("cuda") 

    # Define the loss functions
    loss_fn, loss=LovaszSoftmax(ignore_index=m_ignore_index), None
    secondary_fn=torch.nn.NLLLoss(ignore_index=m_ignore_index)  #combination of nll and dice  https://arxiv.org/pdf/1809.10486.pdf

    #create dataloaders for both phases
    loader_train, loader_valid,_,_ = create_loader(train_config["dataset_name"], config_parser, model_config["sequence_learning"], loader_params["shuffle"],train_border=0, valid_border=0)
    phases= [
        Phase('train', loader_train, grad=True),
        Phase('valid', loader_valid, grad=False)
    ]

    # initialize all callbacks
    cb_list = []
    if(train_config["with_visdom"]):
        cb_list.append(VisdomCallback(None))
    if(train_config["with_viewer"]):
        cb_list.append(ViewerCallback())
    # # wandb.watch enables the tracking of gradients
    if(train_config["with_wandb"]):
        cb_list.append(WandBCallback(experiment_name,config_file,wandb_entity,model))
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)

    

    nr_batches_processed, nr_epochs, first_time = 0,0,True  # set some parameters that track the progress
    
    # Train/Validation loop
    while True:
        # which phase is currently relevant?
        for phase in phases:
            if (nr_epochs > train_config["training_epochs"]-1) and phase.grad:
                return
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)
            is_training = phase.grad
            model.train(is_training)
            torch.cuda.empty_cache()

            pbar = tqdm(total=len(phase.loader.dataset)) 
            loader_iter = phase.loader.__iter__()

            for batch_idx, (positions_seq, values_seq, target_seq, path_seq, _) in enumerate(loader_iter):
                assert positions_seq is not None, "positions_seq for batch_idx {} is None!".format(batch_idx)

                for i in range(0,len(positions_seq)):
                    positions = positions_seq[i].squeeze(0).to("cuda") 
                    values = values_seq[i].squeeze(0).to("cuda") 
                    target = target_seq[i].squeeze(0).to("cuda") 
                    assert positions.shape[0] == target.shape[0], "Position shape {} and target shape {} have to be the same in the first dimension!".format(positions.shape[0], target.shape[0]) 
                    
                    #forward
                    with torch.set_grad_enabled(is_training):
                        early_return = (i != len(positions_seq)-1)
                        if i == len(positions_seq)-1:
                            cb.before_forward_pass(lattice=lattice) #sets the appropriate sigma for the lattice

                        pred_logsoftmax, pred_raw, lattice = model(lattice, positions, values, early_return, with_gradient = is_training) # lattice here is ls

                        #if its the first time we do a forward on the model we need to load the checkpoint
                        if first_time and i==len(positions_seq)-1:
                            first_time=False
                            # initialize optimizer
                            optimizer=torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"], amsgrad=True)
                            # initialize scheduler
                            if train_config["scheduler"] == "CosineAnnealingWarmRestarts":
                                restart_epochs = train_config["restart_epochs"]
                                print("Scheduler: CosineAnnealingWarmRestarts with restarts after {} epochs.".format(restart_epochs))
                                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=restart_epochs)
                            elif train_config["scheduler"] == "ReduceLROnPlateau":
                                print("Scheduler: ReduceLROnPlateau ")
                                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.1)
                            else:
                                sys.exit("Scheduler has to be either ReduceLROnPlateau or CosineAnnealingWarmRestarts. I got {}".format(train_config["scheduler"]))
                                                        
                            if train_config["load_checkpoint"]:
                                # now that all the parameters are created we can fill them with a model from a file
                                model_path = os.path.join(train_config["checkpoint_path"], train_config["load_checkpoint_model"])
                                print("Loading state dict: ", model_path)
                                # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3 -> load all parts of the state dict that exist in the current model
                                model.load_state_dict(torch.load(model_path))
                                model.train(phase.grad)
                                model.reset_sequence()
                                lattice=Lattice.create(config_file, "lattice") #lattice has to be reset aswell
                                
                                #need to rerun forward with the new parameters to get an accurate prediction
                                for k in range(0,len(positions_seq)):
                                    early_return = (k != len(positions_seq)-1)
                                    positions = positions_seq[k].squeeze(0).to("cuda")
                                    values = values_seq[k].squeeze(0).to("cuda") 
                                    target = target_seq[k].squeeze(0).to("cuda") 
                                    pred_logsoftmax, pred_raw, lattice = model(lattice, positions, values, early_return, is_training)

                        # Calculate loss
                        if i == len(positions_seq)-1:
                            # we only want to calculate loss, IoU etc for the last cloud of the sequence
                            loss_dice = 0.5*loss_fn(pred_logsoftmax, target)
                            loss_ce = 0.5*secondary_fn(pred_logsoftmax, target)
                            loss = loss_dice + loss_ce
                                                        
                            cloud = create_cloud(positions, target, path_seq[-1][0], label_mngr, pred_logsoftmax) # the viewer uses this cloud structure
                            cb.after_forward_pass(pred_softmax=pred_logsoftmax, target=target, cloud=cloud, loss=loss.item(), loss_dice=loss_dice.item(), 
                                                phase=phase, lr=optimizer.param_groups[0]["lr"], iteration = phase.iter_nr, ignore_index = m_ignore_index, nr_vertices = lattice.nr_lattice_vertices()) #visualizes the prediction 

                    
                    #backward
                    if is_training and (i == len(positions_seq)-1):
                        # CosineAnnealingWarmRestarts has to be changed after each iteration
                        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                            scheduler.step(phase.epoch_nr + float(phase.samples_processed_this_epoch) / (len(phase.loader.dataset)) )
                        
                        optimizer.zero_grad()
                        cb.before_backward_pass()
                        loss.backward()                        
                        cb.after_backward_pass()
                        optimizer.step()
                    
                    # reset the hidden state and the lattice after each sequence
                    if (i == len(positions_seq)-1):
                        pbar.update(1)
                        model.reset_sequence()
                        lattice=Lattice.create(config_file, "lattice")

                # End of epoch
                if batch_idx == len(loader_iter)-1:
                    pbar.close()
                    # If we use ReduceLROnPlateau, we only do steps here 
                    if not is_training: #we reduce the learning rate when the test iou plateus
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(phase.loss_acum_per_epoch) #for ReduceLROnPlateau
                    
                    date_time = datetime.now().strftime("%d%m%Y_%H%M")  
                    model_name = "{}_{}_{}_{}_sigma{}_type{}_frames{}_scope{}_epoch{}".format(date_time, "multi" if loader_config["include_moving_classes"] == True else "single", "Kitti" if dataset_name=="semantickitti" else "Paris", "Ref" if model_config["values_mode"] == "reflectance" else "xyz" ,str(lattice_gpu_config["sigma_0"])[0:3],"-".join(model_config["rnn_modules"]) if not loader_params["accumulate_clouds"] else "ACCUM",loader_params["frames_per_seq"], loader_params["cloud_scope"], nr_epochs)                   
                    
                    if is_training and train_config["save_checkpoint"]: 
                        check_PATH = str(train_config["checkpoint_path"]) + model_name + ".pt"
                        torch.save(model.state_dict(), check_PATH)
                        print("Saved checkpoint under: ", check_PATH)

                    cb.epoch_ended(phase=phase, model=model, save_checkpoint=train_config["save_checkpoint"], checkpoint_path=train_config["checkpoint_path"], name = model_name) 
                    cb.phase_ended(phase=phase) 


                if train_config["with_viewer"]:
                    view.update()

                nr_batches_processed+=1

            if phase.grad:
                nr_epochs += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the network on a dataset.')
    parser.add_argument('--dataset', type=str, nargs = "?", const = "semantickitti", 
                    help='the dataset name, options are semantickitti OR parislille')

    args = parser.parse_args()

    if args.dataset:
        run(args.dataset)  
    else: # when you do not give any arguments the parser just assumes you want semantickitti
        run()

    # This is what you would have, but the following is useful:
    # # These are temporary, for debugging, so meh for programming style.
    # import sys, trace

    # # If there are segfaults, it's a good idea to always use stderr as it
    # # always prints to the screen, so you should get as much output as
    # # possible.
    # sys.stdout = sys.stderr

    # # Now trace execution:
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.run('main()')
