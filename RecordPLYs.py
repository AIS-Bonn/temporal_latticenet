#import faulthandler; faulthandler.enable()

import torch


import os
import time
print(time.asctime())

from cfgParser import *

import sys
import os
try:
  import torch
except ImportError:
  pass
from easypbr  import *
from tqdm import tqdm

from easypbr  import *
from os import listdir
from os.path import isfile, join
import natsort 
from dataloader.kitti_dataloader import *


def load_pred_label_SP(sequence, index, dir = "/workspace/Data/SemanticKitti/spsequencenet_predictions"):
  filename = os.path.join(dir,'sequences_prob', '{:02d}'.format(sequence), 'predictions', '{:06d}.label'.format(index))
  with open(filename, 'r') as file:
    labels = file.read().split('\n')

  return np.array(labels[0:-1]).astype('int64')


def load_pred_label(sequence, index, dir = "/workspace/temporal_latticenet/predictions_validation/gru_gru_aflow_gru"):
  filename = os.path.join(dir,'sequences', '{:02d}'.format(sequence), 'predictions', '{:06d}.label'.format(index))
  with open(filename, 'r') as file:
    labels = file.read().split('\n')

  return np.array(labels[0:-1]).astype('int64')


def visMultipleDatasetPred(myview, label_mngr, pic_folder = "./recordings"):
  base_path = "/workspace/temporal_latticenet/predictions_validation"
  extras = ["gru_gru_aflow_gru"]
  tmp = "sequences/08/predictions/"

  directories = []
  for extra in extras:
    directories.append(os.path.join(base_path, extra))

  config_file="/workspace/temporal_latticenet/seq_config/lnn_eval_semantic_kitti.cfg"
  print(config_file)
  config_parser = cfgParser(config_file)
  dataset = SemanticKittiDataset(split = "valid", config_parser = config_parser, sequence_learning = True)
  
  # the visualizations in the paper are sequence 8 with clouds 998, 3872 and 30
  #205#241#3872#0#30#998#240#2815
  valid_border=30
  valid_sampler = list(range(len(dataset)))[valid_border:] if valid_border is not None else None
  dataloader = torch.utils.data.DataLoader(dataset, num_workers = 8, batch_size=1, shuffle = False,  sampler = valid_sampler)

  myview.m_camera.from_string("9.24242  6.11435 -7.09283 0.0925219  0.890751  0.210117 -0.392239  9.26866  6.09659 -7.06882 60 0.3 6013.13")

  loader_iter = dataloader.__iter__()

  mymesh = Mesh()
  mymesh.m_label_mngr=label_mngr

  pbar = tqdm(total=len(dataloader.dataset))

  for batch_idx, (positions_seq, values_seq, target_seq, path_seq,_) in enumerate(loader_iter):
    pbar.update(1)

    path = path_seq[-1][0].split("/")

    gru_gru_cli_gru = load_pred_label(int(path[-3]), int(path[-1][0:-4]), directories[0])
    spsequencenet = load_pred_label_SP(int(path[-3]), int(path[-1][0:-4]))
    gT = target_seq[-1].squeeze(0).clone().numpy() 


    mymesh_minus_0 = create_cloud(positions_seq[-1].squeeze(0), target_seq[-1].squeeze(0), "", label_mngr)
    #mymesh_minus_0.m_vis.set_color_semanticgt()
    mymesh_minus_0.m_vis.m_solid_color = [0.5, 0.5, 0.5]
    mymesh_minus_0.m_vis.set_color_solid()     
    mymesh_minus_0.m_vis.m_point_size=5.0        
    mymesh_minus_0.m_vis.m_show_points=True
    Scene.show(mymesh_minus_0,"mesh")
    

    mymesh_minus_1 = create_cloud(positions_seq[-2].squeeze(0), target_seq[-2].squeeze(0), "", label_mngr)
    #mymesh_minus_1.m_vis.set_color_semanticgt()
    mymesh_minus_1.m_vis.m_solid_color = [0.5, 0.5, 0.5]
    mymesh_minus_1.m_vis.set_color_solid()  
    mymesh_minus_1.m_vis.m_point_size=5.0        
    mymesh_minus_1.m_vis.m_show_points=True
    Scene.show(mymesh_minus_1,"mesh-1")
    

    mymesh_minus_2 = create_cloud(positions_seq[-3].squeeze(0), target_seq[-3].squeeze(0), "", label_mngr)
    #mymesh_minus_2.m_vis.set_color_semanticgt()
    mymesh_minus_2.m_vis.m_solid_color = [0.5, 0.5, 0.5]
    mymesh_minus_2.m_vis.set_color_solid()  
    mymesh_minus_2.m_vis.m_point_size=5.0        
    mymesh_minus_2.m_vis.m_show_points=True
    Scene.show(mymesh_minus_2,"mesh-2")


    mymesh = mymesh.clone()
    mymesh.m_label_mngr=label_mngr
    mymesh.V = positions_seq[-1].squeeze(0).clone()#[gT == 21]
    mymesh.L_pred = torch.tensor(gru_gru_cli_gru)#[gT == 21]
    mymesh.L_gt = target_seq[-1].squeeze(0).clone()#[gT == 21]    
    mymesh.m_vis.m_point_size=14.0#8.0        
    mymesh.m_vis.m_show_points=False
    mymesh.m_vis.set_color_semanticpred()
    Scene.show(mymesh,"gru_gru_cli_gru")


    mymesh5 = mymesh.clone()
    mymesh5.m_label_mngr=label_mngr     
    mymesh5.m_vis.m_show_points=True
    mymesh5.L_pred = torch.tensor(spsequencenet)#[gT == 21]
    Scene.show(mymesh5,"spsequencenet")

    mymesh3 = mymesh.clone()
    mymesh3.m_label_mngr=label_mngr     
    mymesh3.m_vis.m_show_points=True
    mymesh3.m_vis.set_color_semanticgt()
    Scene.show(mymesh3,"gT")

    while True:
      myview.update()

      if myview.m_new_cloud:
        #recorder.record(str(batch_idx)+".png", "./recordings/")
        myview.m_new_cloud = False
        break


if __name__ == "__main__":
  config_file="/workspace/temporal_latticenet/seq_config/lnn_eval_semantic_kitti.cfg"
  config_parser = cfgParser(config_file)

  label_mngr_params = config_parser.get_label_mngr_vars()
  m_ignore_index = label_mngr_params["unlabeled_idx"]
  labels_file=str(label_mngr_params["labels_file"])
  colorscheme_file=str(label_mngr_params["color_scheme_file"])
  frequency_file=str(label_mngr_params["frequency_file"])
  label_mngr=LabelMngr(labels_file, colorscheme_file, frequency_file, m_ignore_index )

  myview=Viewer.create(config_file) #first because it needs to init context
  
  recorder=myview.m_recorder
  Scene.set_floor_visible(False)

  visMultipleDatasetPred(myview, label_mngr)