#!/usr/bin/env python3.6
import torch

import sys, os, argparse, time
from tqdm import tqdm
print(time.asctime())
print("PID: ", os.getpid()) # you can use any application to remind you if the script breaks by checking for this PID

from dataloader.kitti_dataloader import *
from dataloader.parisLille_dataloader import *
from easypbr  import *
from latticenet  import ModelParams
from latticenet_py.lattice.lovasz_loss import LovaszSoftmax

from callbacks.callback import *
from callbacks.viewer_callback import *
from callbacks.visdom_callback import *
from callbacks.state_callback import *
from callbacks.phase import *

from cfgParser import *
from pathlib import Path
from seq_lattice.models import *

class CloudReadingException(Exception):
    pass

from datetime import datetime

#torch.manual_seed(0)
torch.set_printoptions(edgeitems=3)

def write_prediction(pred_softmax, cloud, pred_path):
    mesh_pred=cloud.clone()
    l_pred=pred_softmax.detach().argmax(axis=1).cpu().numpy()
    l_pred = np.expand_dims(l_pred, axis=1)
    mesh_pred.color_from_label_indices(l_pred)
    mesh_pred.L_pred=l_pred
    mesh_pred.save_to_file(pred_path)
    
def write_gt(cloud, gt_path):
    mesh_gt=cloud.clone()
    mesh_gt.color_from_label_indices(cloud.L_gt)
    mesh_gt.save_to_file(gt_path)


# train_border and valid_border are integers
def create_loader(dataset_name, config_parser, sequence_learning = False, shuffle = False, train_border = None, valid_border = None):
    if(dataset_name=="semantickitti"):
        test_dataset = SemanticKittiDataset(split = "test", config_parser = config_parser, sequence_learning = sequence_learning)
    elif(dataset_name=="parislille"):
        test_dataset = ParisLille3DDataset(split = "test", config_parser = config_parser, sequence_learning = sequence_learning)
    else:
        err="Dataset name not recognized. It is " + dataset_name
        sys.exit(err)

    test_sampler = list(range(len(test_dataset)))[valid_border:] if valid_border is not None else None
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers = 8, batch_size=1, shuffle = False, sampler = test_sampler)

    return test_dataloader, test_dataset


def run(dataset_name = "semantickitti"):
    if dataset_name == "semantickitti":
        print("\n-------- Using SemanticKitti Dataset --------")
        config_file="/workspace/temporal_latticenet/seq_config/lnn_eval_semantic_kitti.cfg"
        print("Config file: ", config_file)
    elif(dataset_name=="parislille"):
        sys.exit("Currently ParisLille3D isn't supported!")
        print("\n-------- Using ParisLille3D Dataset --------")
        config_file="/workspace/temporal_latticenet/seq_config/lnn_eval_paris_lille.cfg"
    else:
        sys.exit("Dataset name not recognized. It is {}. Available options are semantickitti or parislille.".format(dataset_name) )

    if not torch.cuda.is_available():
        sys.exit("The GPU is not available!")

    config_parser = cfgParser(config_file)
    model_config = config_parser.get_model_vars()
    eval_config = config_parser.get_eval_vars()
    model_params=ModelParams.create(config_file)    
    loader_params = config_parser.get_loader_semantic_kitti_vars()
    label_mngr_params = config_parser.get_label_mngr_vars()
    lattice_gpu_config = config_parser.get_lattice_gpu_vars()
    loader_config = config_parser.get_loader_vars()

    # Print some nice information
    print("Lattice sigma: ", str(lattice_gpu_config["sigma_0"])[0:3])
    print("Sequences: #scans: {}, cloud scope: {}".format((loader_config['frames_per_seq'] if model_config["sequence_learning"] else 1), loader_config['cloud_scope']))
    print("Features: ", model_config["values_mode"])
    if eval_config["do_write_predictions"]:
        Path(eval_config["output_predictions_path"]).mkdir(parents=True, exist_ok=True) 
        print("The predictions will be saved to: ", str(eval_config["output_predictions_path"]))

    # initialize all callbacks
    cb_list = []
    if(eval_config["with_viewer"]):
        cb_list.append(ViewerCallback())
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)

    # initialize the LabelMngr and the viewer
    m_ignore_index = label_mngr_params["unlabeled_idx"]
    labels_file=str(label_mngr_params["labels_file"])
    colorscheme_file=str(label_mngr_params["color_scheme_file"])
    frequency_file=str(label_mngr_params["frequency_file_all"]) if loader_params["include_moving_classes"] else  str(label_mngr_params["frequency_file"])
    label_mngr=LabelMngr(labels_file, colorscheme_file, frequency_file, m_ignore_index )
    if eval_config["with_viewer"]:
        view=Viewer.create(config_file)

    
    # Initialize the networks model
    lattice=Lattice.create(config_file, "lattice") # create Lattice
    model = None
    if not loader_params["include_moving_classes"] and (eval_config["dataset_name"] == "semantickitti"):
        model=LNN_SEQ(20, model_params, config_parser).to("cuda")
    elif (eval_config["dataset_name"] == "semantickitti"):
        #print("Including moving classes - therefore 26 classes")
        model=LNN_SEQ(26, model_params, config_parser).to("cuda")
    elif not loader_params["include_moving_classes"] and (eval_config["dataset_name"] == "parislille"):
        model=LNN_SEQ(10, model_params, config_parser).to("cuda") # parislille has only 10 classes
    elif (eval_config["dataset_name"] == "parislille"):
        model=LNN_SEQ(12, model_params, config_parser).to("cuda") 

    # Define the loss functions
    loss_fn, loss=LovaszSoftmax(ignore_index=m_ignore_index), None
    secondary_fn=torch.nn.NLLLoss(ignore_index=m_ignore_index)  #combination of nll and dice  https://arxiv.org/pdf/1809.10486.pdf

    #create dataloaders for both phases
    loader_test,_ = create_loader(eval_config["dataset_name"], config_parser, model_config["sequence_learning"], loader_params["shuffle"])
    phases= [
        Phase('test', loader_test, grad=False)
    ]

    nr_batches_processed, nr_epochs, first_time = 0,0,True  # set some parameters that track the progress
    
    while True:

        for phase in phases:
            # cb.epoch_started(phase=phase)
            # cb.phase_started(phase=phase)
            model.train(phase.grad)
            is_training = phase.grad

            pbar = tqdm(total=len(phase.loader.dataset))
            loader_iter = phase.loader.__iter__()

            for batch_idx, (positions_seq, values_seq, target_seq, path_seq, len_seq) in enumerate(loader_iter):
                assert positions_seq is not None, "positions_seq for batch_idx {} is None!".format(batch_idx)

                for i in range(0,len(positions_seq)):

                    positions = positions_seq[i].squeeze(0).to("cuda") #.detach().clone().to("cuda")
                    values = values_seq[i].squeeze(0).to("cuda") #.detach().clone().to("cuda")
                    target = target_seq[i].squeeze(0).to("cuda") #.detach().clone().to("cuda")
                    assert positions.shape[0] == target.shape[0], "Position shape {} and target shape {} have to be the same in the first dimension!".format(positions.shape[0], target.shape[0]) 
                    
                    #forward pass
                    with torch.set_grad_enabled(is_training):
                        early_return = (i != len(positions_seq)-1)
                        if i == len(positions_seq)-1:
                            cb.before_forward_pass(lattice=lattice) #sets the appropriate sigma for the lattice
                        
                        pred_logsoftmax, pred_raw, lattice = model(lattice, positions, values, early_return, with_gradient = is_training) # lattice here is ls


                        #if its the first time we do a forward on the model we need to load here the checkpoint
                        if first_time and i==len(positions_seq)-1:
                            first_time = False
                            # now that all the parameters are created we can fill them with a model from a file
                            model_path = os.path.join(eval_config["checkpoint_path"], eval_config["load_checkpoint_model"])
                            print("Loading state dict from ", model_path)
                            model.load_state_dict(torch.load(model_path))
                            model.train(phase.grad)
                            model.reset_sequence()
                            lattice=Lattice.create(config_file, "lattice")
                            
                            #need to rerun forward with the new parameters to get an accurate prediction
                            for k in range(0,len(positions_seq)):
                                early_return = (k != len(positions_seq)-1)
                                positions = positions_seq[k].squeeze(0).to("cuda")
                                values = values_seq[k].squeeze(0).to("cuda") 
                                target = target_seq[k].squeeze(0).to("cuda") 
                                pred_logsoftmax, pred_raw, lattice = model(lattice, positions, values, early_return, is_training)
     
                        if (i == (len(positions_seq)-1)):
                            pbar.update(1)  
                            cloud = create_cloud(positions, target, path_seq[-1][0], label_mngr, pred_logsoftmax) # the viewer uses this cloud structure
                            cb.after_forward_pass(pred_softmax=pred_logsoftmax, target=target, cloud=cloud, loss=0, loss_dice=0, phase=phase, lr=0) #visualizes the prediction                          

                        if eval_config["do_write_predictions"] and i==len(positions_seq)-1: 

                            #if isinstance(phase.loader, DataLoaderSemanticKitti):
                            # full path in which we save the cloud depends on the data loader. If it's semantic kitti we save also with the sequence, if it's scannet
                            # cloud_path_full=scan_path[0]
                            cloud_path_full=cloud.m_disk_path
                            # cloud_path=os.path.join(os.path.dirname(cloud_path), "../../")
                            basename=os.path.splitext(os.path.basename(cloud_path_full))[0]
                            cloud_path_base=os.path.abspath(os.path.join(os.path.dirname(cloud_path_full), "../../"))
                            cloud_path_head=os.path.relpath( cloud_path_full, cloud_path_base  )
                            # print("cloud_path_head is ", cloud_path_head)
                            # print("cloud_path head dirnmake ", os.path.dirname(os.path.dirname(cloud_path_head)) )
                            # print("basename is ", basename)
                            path_before_file=os.path.join(eval_config["output_predictions_path"], "sequences",  os.path.dirname(os.path.dirname(cloud_path_head)), "predictions")
                            os.makedirs(path_before_file, exist_ok=True)
                            
                            # write ply files that represents the prediction
                            to_save_path=os.path.join(path_before_file, basename )
                            #print("saving in ", to_save_path)
                            pred_path=to_save_path+"_pred.ply"
                            gt_path=to_save_path+"_gt.ply"
                            # print("writing prediction to ", pred_path)
                            #write_prediction(pred_logsoftmax, cloud, pred_path)
                            #write_gt(cloud, gt_path)


                            #write labels file (just a file containing for each point the predicted label)
                            l_pred = pred_logsoftmax.clone().detach().argmax(axis=1).cpu().numpy()
                            l_pred = l_pred[-1*len_seq[-1]:] # for the ACCUM case I need to get only the points of the last point cloud
                            l_pred = l_pred.reshape((-1))
                            l_pred = l_pred.astype(np.uint32)
                            #print(l_pred)
                            labels_file= os.path.join(path_before_file, (basename+".label") )    
                            #print("Saving label file here: ", labels_file)            
                            #print(labels_file)
                            l_pred.tofile(labels_file)
                            with open(labels_file, 'w') as f:
                                for idx in range(l_pred.shape[0]):
                                    line= str(l_pred[idx]) + "\n"
                                    f.write(line)
                           

                            ################################################################################
                            ##############IMPORTANT for competition################
                            #after running this test.py script and getting all the .label files. You need to run the remap_semantic_labels from https://github.com/PRBonn/semantic-kitti-api/
                            # you need to run with the --inverse flag and the correct .config (depending if you use 20 classes or 26 classes with the moving objects) in order to get the original labels and only then you can upload to the codalab server
                            #example: 
                            # ./remap_semantic_labels.py --predictions /media/rosu/Data/data/semantic_kitti/predictions/motion_seg --split test --datacfg config/semantic-kitti-all.yaml --inverse
                            # 26: ./remap_semantic_labels.py --predictions ../temporal_latticenet/predictions/tests/ --split valid --datacfg config/semantic-kitti-all.yaml --inverse
                            # in remap: label = np.fromfile(label_file, dtype=np.uint32, sep = "\n")
                            # ./validate_submission.py --task segmentation /media/rosu/Data/data/semantic_kitti/for_server/big_network_early_linear_cloud_nr_3_scope_1/motion_seg_remapped.zip /media/rosu/Data/data/kitti/data_odometry_velodyne/dataset 

                            #to validate on the valid set 
                            # ./evaluate_semantics.py --dataset /media/rosu/Data/data/kitti/data_odometry_velodyne/dataset  --predictions /media/rosu/Data/data/semantic_kitti/predictions/motion_seg_validation --split valid
                            # 26: ./evaluate_semantics.py --dataset ../semantic_kitti/dataset/ --predictions ../temporal_latticenet/predictions/tests/ --split valid -dc config/semantic-kitti-all.yaml
                            ################################################################################


                           
                            # #write GT labels file (just a file containing for each point the predicted label)
                            # gt = np.squeeze(cloud.L_gt)
                            # labels_file= os.path.join(path_before_file, (basename+".gt") )                
                            # with open(labels_file, 'w') as f:
                            #     for i in range(gt.shape[0]):
                            #         line= str(gt[i]) + "\n"
                            #         f.write(line)

                            # #check the predictions from tangentconv and get how much different we are from it. We want to show an image of the biggest change in accuracy
                            # #we want the difference to gt to be small and the difference to tangent conv to be big
                            # tangentconv_path="/home/user/rosu/data/semantic_kitti/predictions_from_related_work/tangent_conv_semantic_kitti_single_frame_final_predictions_11_21"
                            # cloud_path_without_seq=os.path.abspath(os.path.join(os.path.dirname(cloud_path_full), "../"))
                            # cloud_path_with_seq=os.path.relpath( cloud_path_full, cloud_path_without_seq  )
                            # seq=os.path.dirname(cloud_path_with_seq)
                            # path_to_tangentconv_pred=os.path.join(tangentconv_path, seq,  (basename + ".label")  )
                            # print("path_to_tangentconv_pred", path_to_tangentconv_pred)

                            # f = open(path_to_tangentconv_pred, "r")
                            # tangentconv_labels = np.fromfile(f, dtype=np.uint32)

                            # #compute score 
                            # l_pred=pred_softmax.detach().argmax(axis=1).cpu().numpy()
                            # gt = np.squeeze(cloud.L_gt)
                            # # print("gt shape", gt.shape)
                            # # print("l_pref shape", l_pred.shape)
                            # # print("tangentconv_labels shape", tangentconv_labels.shape)
                            # point_is_valid = gt!=0
                            # nr_valid_points=point_is_valid.sum()
                            # point_is_different_than_gt = gt != l_pred
                            # diff_to_gt = (np.logical_and(point_is_different_than_gt, point_is_valid)).sum()
                            # point_is_different_than_tangentconv = tangentconv_labels != l_pred
                            # diff_to_tangentconv = (np.logical_and(point_is_different_than_tangentconv, point_is_valid)).sum()
                            # # print("diff to gt  is ", diff_to_gt)
                            # # print("diff to tangentconv  is ", diff_to_tangentconv)
                            # score=diff_to_tangentconv-diff_to_gt ##we try to maximize this score
                            # score /=nr_valid_points #normalize by the number of points becuase otherwise the score will be squeed towards grabbing point clouds that are just gigantic because they have more points
                            # print("score is ", score)

                            # #store the score and the path in a list
                            # predictions_list.append(cloud_path_head)
                            # scores_list.append(score)
                            # # print("predictions_list",predictions_list)
                            # # print("score_lists",scores_list)

                            # #sort based on score https://stackoverflow.com/a/6618543
                            # predictions_sorted=[predictions_list for _,predictions_list in sorted(zip(scores_list,predictions_list))]
                            # scores_sorted=np.sort(scores_list)
                            # # print("predictions_sorted",predictions_sorted)
                            # # print("scores_sorted",scores_sorted)
                            # # print("predictions_list",predictions_list)
                            # # print("score_lists",scores_list)

                            # #write the sorted predictions to file 
                            # best_predictions_file=os.path.join(eval_params.output_predictions_path(), "best_preds.txt")
                            # with open(best_predictions_file, 'w') as f:
                            #     for i in range(len(predictions_sorted)):
                            #         line= predictions_sorted[i] +  "    score: " +  str(scores_sorted[i]) + "\n"
                            #         f.write(line)
                        cloud = None
                    
                    # reset the hash map after each sequence
                    if (i == len(positions_seq)-1):
                        model.reset_sequence()
                        lattice=Lattice.create(config_file, "lattice")

                if batch_idx == len(loader_iter)-1:
                    pbar.close()
                    return

                if eval_config["with_viewer"]:
                    view.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the network on a dataset.')
    parser.add_argument('--dataset', type=str, nargs = "?", const = "semantickitti", 
                    help='the dataset name, options are semantickitti OR parislille')

    args = parser.parse_args()

    if args.dataset:
        run(args.dataset)  
    else: # when you do not give any arguments the parser just assumes you want semantickitti
        run()
