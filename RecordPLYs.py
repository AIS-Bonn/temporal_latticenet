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


def load_pred_label(sequence, index):
    filename = os.path.join(dir,'sequences', '{:02d}'.format(sequence), 'labels', '{:06d}.label'.format(index))
    assert(os.path.isfile(filename)), str("Filename not found: ", filename)
    npz = np.fromfile(filename, dtype=np.uint16)
    
    # Lower 16 bits: Label. 
    # Upper 16 bits: Instance id
    #labels = (npz[0::2].reshape(len(npz)//2, 1)).astype(np.int16)
    #labels = remap_lut[labels] # the labels in the normal dataset are not in the range 0-19 or 0-25 (see website for more information)
    
    #instance_id = npz[1::2]
    return npz


def recordWholeSequence(meshes_path, myview, label_mngr, pic_folder = "./recordings"):
  mymesh = Mesh()
  mymesh.m_label_mngr=label_mngr

  idx=0
  while idx < len(files):
      mymesh.load_from_file(os.path.join(meshes_path, files[idx]) )
      # these set all vertices to the same color
      #mymesh.m_vis.m_solid_color = [0.5, 0.5, 0.5]
      #mymesh.m_vis.set_color_solid()
      
      mymesh.m_vis.m_point_size=5.0        
      mymesh.m_vis.m_show_points=True

      Scene.show(mymesh,"mesh")
      myview.update()

      #move camera
      myview.m_camera.orbit_y(0.5)
      if idx==0:
        myview.m_camera.set_lookat([0.0, 0.0, 0.0])
        myview.m_camera.push_away(0.5)

      
      #recorder.record(str(idx)+".png", "./solid_gt_predictions")
      recorder.record(str(idx)+".png", pic_folder)

      idx+=1


def recordChosenClouds(meshes_path, myview, label_mngr, idx_list, pic_folder = "./recordings"):
  mymesh = Mesh()
  mymesh.m_label_mngr=label_mngr

  for idx in idx_list:
      mymesh.load_from_file(os.path.join(meshes_path, files[idx]) )
    
      #mymesh.m_vis.m_point_color=[0.5, 0.5, 0.5]        
      mymesh.m_vis.m_point_size=5.0        
      mymesh.m_vis.m_show_points=True

      Scene.show(mymesh,"mesh")
      myview.update()

      #move camera
      myview.m_camera.orbit_y(0.5)
      if idx==idx_list[0]:
        myview.m_camera.set_lookat([0.0, 0.0, 0.0])
        myview.m_camera.push_away(0.5)

      
      #recorder.record(str(idx)+".png", "./solid_gt_predictions")
      recorder.record(str(idx)+".png", pic_folder)

      idx+=1


def visCertainSequence(meshes_path, myview, label_mngr, idx_list, pic_folder = "./recordings"):
  config_file="/workspace/schuett_temporal_lattice/seq_config/lnn_train_semantic_kitti.cfg"
  config_parser = cfgParser(config_file)
  dataset = SemanticKittiDataset(split = "train", config_parser = config_parser, sequence_learning = True)

  idx = 900#800
  seq = dataset[idx]
  #idx = 3457

  positions_seq = seq[0]
  positions = positions_seq[0].squeeze(0)
  target_seq = seq[2]
  target = target_seq[0].squeeze(0)
  mymesh = create_cloud(positions, target, "", label_mngr)
  

  positions_seq = seq[0]
  positions = positions_seq[1].squeeze(0)
  target_seq = seq[2]
  target = target_seq[1].squeeze(0)
  mymesh1 = create_cloud(positions, target, "", label_mngr)
  

  positions_seq = seq[0]
  positions = positions_seq[2].squeeze(0)
  target_seq = seq[2]
  target = target_seq[2].squeeze(0)
  mymesh2 = create_cloud(positions, target, "", label_mngr)

  
  while True:
      #mymesh.load_from_file(os.path.join(meshes_path, files[idx]) )
      #print(os.path.join(meshes_path, files[idx]))
      
      # scene disable grid
      #print(dir(mymesh.m_vis))
      #exit()
      #Scene.set_floor_visible(False)

      # these set all vertices to the same color
      #mymesh.m_vis.m_solid_color = [0.5, 0.5, 0.5]
      #mymesh.m_vis.set_color_solid()
      mymesh.m_vis.set_color_semanticgt()
      

      #mymesh.m_vis.m_point_color=[0.5, 0.5, 0.5]        
      mymesh.m_vis.m_point_size=5.0        
      mymesh.m_vis.m_show_points=True

      Scene.show(mymesh,"mesh")
      #myview.update()
      #recorder.record(str(idx)+".png", "./solid_gt_predictions")           
      #mymesh.m_vis.m_show_points=True


      #mymesh1.load_from_file(os.path.join(meshes_path, files[idx+1]) )
      #mymesh1.m_vis.m_solid_color = [0.5, 0.5, 0.5]
      #mymesh1.m_vis.set_color_solid()  
      mymesh1.m_vis.set_color_semanticgt()   
      mymesh1.m_vis.m_point_size=5.0        
      mymesh1.m_vis.m_show_points=True
      Scene.show(mymesh1,"mesh+1")
      #myview.update()
      #print("Test", flush = True)
      #recorder.record(str(idx+1)+".png", "./solid_gt_predictions")
      #mymesh1.m_vis.m_show_points=True

      #mymesh2.load_from_file(os.path.join(meshes_path, files[idx+2]) )
      #mymesh2.m_vis.m_solid_color = [0.5, 0.5, 0.5]
      #mymesh2.m_vis.set_color_solid()     
      mymesh2.m_vis.set_color_semanticgt()
      mymesh2.m_vis.m_point_size=5.0        
      mymesh2.m_vis.m_show_points=True
      Scene.show(mymesh2,"mesh+2")
      #myview.update()
      #recorder.record(str(idx+2)+".png", "./solid_gt_predictions")
      #mymesh2.m_vis.m_show_points=True

      #mymesh3.load_from_file(os.path.join(meshes_path, files[idx+2]) )
      #mymesh.m_vis.m_solid_color = [0.5, 0.5, 0.5]
      #mymesh.m_vis.set_color_solid()     
      # mymesh3.m_vis.m_point_size=5.0        
      # mymesh3.m_vis.m_show_points=True
      # Scene.show(mymesh3,"pred")
      # myview.update()

      while True:
        myview.update()

      idx+=1

  
def visWholeDatasetGT(myview, label_mngr):
  config_file="/workspace/schuett_temporal_lattice/seq_config/lnn_train_semantic_kitti.cfg"
  config_parser = cfgParser(config_file)
  dataset = SemanticKittiDataset(split = "valid", config_parser = config_parser, sequence_learning = True)
  dataloader = torch.utils.data.DataLoader(dataset, num_workers = 0, batch_size=1, shuffle = False)

  loader_iter = dataloader.__iter__()

  mymesh = Mesh()
  mymesh.m_label_mngr=label_mngr
  #print(dir(mymesh.m_vis))

  car=Mesh("/workspace/schuett_temporal_lattice/visualization_assets/car_passat_big.ply")
  car.V=car.V
  #car.model_matrix.rotate_axis_angle_local([1,0,0],-90)
  #car.model_matrix.rotate_axis_angle_local([0,1,0],-90)
  #car.translate_model_matrix([0, -1.5, 0])
  #car.translate_model_matrix([0, 0, -11.1])
  car.model_matrix.rotate_axis_angle_local([1,0,0],-90)
  car.model_matrix.rotate_axis_angle_local([0,1,0],-90)
  car.translate_model_matrix([0, -1.1, 0])

  for batch_idx, (positions_seq, values_seq, target_seq, path_seq,_) in enumerate(loader_iter):

    mymesh = mymesh.clone()
    mymesh.V = positions_seq[-1].squeeze(0).clone()
    mymesh.L_gt = target_seq[-1].squeeze(0).clone()    
    mymesh.m_vis.m_point_size=5.0        
    mymesh.m_vis.m_show_points=True
    mymesh.m_vis.set_color_semanticgt()

    Scene.show(car,"car")

    Scene.show(mymesh,"mesh")
    
    myview.update()

  
def load_pred_label_SP(sequence, index, dir = "/workspace/semantic_kitti/val_muti_v1"):
  filename = os.path.join(dir,'sequences_prob', '{:02d}'.format(sequence), 'predictions', '{:06d}.label'.format(index))
  print(filename)
  assert(os.path.isfile(filename)), str("Filename not found: ", filename)
  #print(filename)
  with open(filename, 'r') as file:
    labels = file.read().split('\n')

  return np.array(labels[0:-1]).astype('int64')


def load_pred_label(sequence, index, dir = "/workspace/schuett_temporal_lattice/predictions/gru_gru_cli_gru"):
  filename = os.path.join(dir,'sequences', '{:02d}'.format(sequence), 'predictions', '{:06d}.label'.format(index))
  assert(os.path.isfile(filename)), str("Filename not found: ", filename)
  #print(filename)
  with open(filename, 'r') as file:
    labels = file.read().split('\n')

  return np.array(labels[0:-1]).astype('int64')


def load_kpconv_label(sequence, index, dir = "//workspace/schuett_temporal_lattice/kpconv_pred/val_preds"):
  file_name = '{:02d}'.format(sequence)+ '_'+ '{:07d}.npy'.format(index)
  filename = os.path.join(dir,file_name)
  #print(filename)
  assert(os.path.isfile(filename)), str("Filename not found: ", filename)
  
  labels = np.load(filename)
  #print(npz.shape)
  return labels



def visWholeDatasetPred(myview, label_mngr, pic_folder = "./recordings"):
  config_file="/workspace/schuett_temporal_lattice/seq_config/lnn_eval_semantic_kitti.cfg"
  config_parser = cfgParser(config_file)
  dataset = SemanticKittiDataset(split = "valid", config_parser = config_parser, sequence_learning = True)
  dataloader = torch.utils.data.DataLoader(dataset, num_workers = 0, batch_size=1, shuffle = False)

  loader_iter = dataloader.__iter__()

  mymesh = Mesh()
  mymesh.m_label_mngr=label_mngr
  #print(dir(mymesh.m_vis))

  # car=Mesh("/workspace/schuett_temporal_lattice/visualization_assets/car_passat_big.ply")
  # car.V=car.V
  # #car.model_matrix.rotate_axis_angle_local([1,0,0],-90)
  # #car.model_matrix.rotate_axis_angle_local([0,1,0],-90)
  # #car.translate_model_matrix([0, -1.5, 0])
  # #car.translate_model_matrix([0, 0, -11.1])
  # car.model_matrix.rotate_axis_angle_local([1,0,0],-90)
  # car.model_matrix.rotate_axis_angle_local([0,1,0],-90)
  # car.translate_model_matrix([0, -1.1, 0])

  for batch_idx, (positions_seq, values_seq, target_seq, path_seq,_) in enumerate(loader_iter):
    path = path_seq[-1][0].split("/")
    pred_label = torch.tensor(load_pred_label(int(path[5]), int(path[-1][0:-4])))

    mymesh = mymesh.clone()
    mymesh.V = positions_seq[-1].squeeze(0).clone()
    mymesh.L_pred = pred_label
    mymesh.L_gt = target_seq[-1].squeeze(0).clone()    
    mymesh.m_vis.m_point_size=5.0        
    mymesh.m_vis.m_show_points=True
    mymesh.m_vis.set_color_semanticpred()

    #Scene.show(car,"car")
    Scene.show(mymesh,"mesh")
    
    recorder.record(str(batch_idx)+".png", pic_folder)
    
    myview.update()

def visMultipleDatasetPred(myview, label_mngr, pic_folder = "./recordings"):
  base_path = "/workspace/schuett_temporal_lattice/predictions"
  extras = ["gru_gru_cli_gru", "gru_gru_cli_cli", "lstm_lstm_cli_lstm"]
  tmp = "sequences/08/predictions/"

  directories = []
  for extra in extras:
    directories.append(os.path.join(base_path, extra))

  config_file="/workspace/schuett_temporal_lattice/seq_config/lnn_eval_semantic_kitti.cfg"
  print(config_file)
  config_parser = cfgParser(config_file)
  dataset = SemanticKittiDataset(split = "valid", config_parser = config_parser, sequence_learning = True)
  valid_border=205#241#3872#0#30#998#240#2815
  valid_sampler = list(range(len(dataset)))[valid_border:] if valid_border is not None else None
  dataloader = torch.utils.data.DataLoader(dataset, num_workers = 8, batch_size=1, shuffle = False,  sampler = valid_sampler)

  myview.m_camera.from_string("9.24242  6.11435 -7.09283 0.0925219  0.890751  0.210117 -0.392239  9.26866  6.09659 -7.06882 60 0.3 6013.13")
  #-31.4014  20.7072  11.7785 -0.208461 -0.533014 -0.137447 0.80842   3.81318 -0.459874   -3.3169 60 0.3 6013.13
  #-18.2988  11.8501  6.87214 -0.206855 -0.552364 -0.143793 0.794622  4.05164  -1.4669 -1.43582 60 0.3 6013.13
  # 9.24242  6.11435 -7.09283 0.0925219  0.890751  0.210117 -0.392239  9.26866  6.09659 -7.06882 60 0.3 6013.13

  loader_iter = dataloader.__iter__()

  mymesh = Mesh()
  mymesh.m_label_mngr=label_mngr

  pbar = tqdm(total=len(dataloader.dataset))

  differences = []   
  max_diff_idx, max_diff = 0, 0
  max_diff_idx_1, max_diff_1 = 0, 0
  max_diff_idx_2, max_diff_2 = 0, 0
  cyclist = []

  for batch_idx, (positions_seq, values_seq, target_seq, path_seq,_) in enumerate(loader_iter):
    pbar.update(1)

    path = path_seq[-1][0].split("/")
    #pred_label = torch.tensor(load_pred_label(int(path[5]), int(batch_idx)))
    #print(path[-1][0:-4])
    #if int(path[-1][0:-4]) > 2000 and int(path[-1][0:-4]) < 3000:
    #  continue

    gru_gru_cli_gru = load_pred_label(int(path[5]), int(path[-1][0:-4]), directories[0])
    # gru_gru_cli_cli = load_pred_label(int(path[5]), int(path[-1][0:-4]), directories[1])
    # lstm_lstm_cli_lstm = load_pred_label(int(path[5]), int(path[-1][0:-4]), directories[2])
    spsequencenet = load_pred_label_SP(int(path[5]), int(path[-1][0:-4]))
    gT = target_seq[-1].squeeze(0).clone().numpy() 


    #kpconv = load_kpconv_label(int(path[5]), int(path[-1][0:-4]))

    #tmp = np.array[np.sum(gT == gru_gru_cli_gru), np.sum(gT == gru_gru_cli_cli), np.sum(gT == lstm_lstm_cli_lstm)]
    # diff = np.sum(gT != gru_gru_cli_gru)
    # if diff > max_diff:
    #   max_diff = diff
    #   max_diff_idx = int(path[-1][0:-4])
    # diff = np.sum(gT != gru_gru_cli_cli)
    # if diff > max_diff_1:
    #   max_diff_1 = diff
    #   max_diff_idx_1 = int(path[-1][0:-4])
    # diff = np.sum(gT != lstm_lstm_cli_lstm)
    # if diff > max_diff_2:
    #   max_diff_2 = diff
    #   max_diff_idx_2 = int(path[-1][0:-4])
    #differences.append( np.array[np.sum(gT == gru_gru_cli_gru), np.sum(gT == gru_gru_cli_cli), np.sum(gT == lstm_lstm_cli_lstm)])
    #print(tmp)

    mymesh = mymesh.clone()
    mymesh.m_label_mngr=label_mngr
    mymesh.V = positions_seq[-1].squeeze(0).clone()#[gT == 21]
    mymesh.L_pred = torch.tensor(gru_gru_cli_gru)#[gT == 21]
    mymesh.L_gt = target_seq[-1].squeeze(0).clone()#[gT == 21]    
    mymesh.m_vis.m_point_size=14.0#8.0        
    mymesh.m_vis.m_show_points=False
    mymesh.m_vis.set_color_semanticpred()
    Scene.show(mymesh,"gru_gru_cli_gru")

    # mymesh1 = mymesh.clone()
    # mymesh1.m_label_mngr=label_mngr     
    # mymesh1.m_vis.m_show_points=True
    # mymesh1.L_pred = torch.tensor(gru_gru_cli_cli)
    # Scene.show(mymesh1,"gru_gru_cli_cli")

    # mymesh2 = mymesh.clone()
    # mymesh2.m_label_mngr=label_mngr     
    # mymesh2.m_vis.m_show_points=True
    # mymesh2.L_pred = torch.tensor(lstm_lstm_cli_lstm)
    # Scene.show(mymesh2,"lstm_lstm_cli_lstm")

    # mymesh4 = mymesh.clone()
    # mymesh4.m_label_mngr=label_mngr     
    # mymesh4.m_vis.m_show_points=True
    # mymesh4.L_pred = torch.tensor(kpconv)
    # Scene.show(mymesh4,"kpconv")

    mymesh5 = mymesh.clone()
    mymesh5.m_label_mngr=label_mngr     
    mymesh5.m_vis.m_show_points=True
    mymesh5.L_pred = torch.tensor(spsequencenet)#[gT == 21]
    Scene.show(mymesh5,"spsequencenet")

    # moving_motorcyclist = 23
    # found = (target_seq[-1].squeeze(0) == 23)
    # if torch.sum(found) > 1:
    #   cyclist.append(str(int(path[5]))+ " "+ str(int(path[-1][0:-4])))

    # if( batch_idx %100) == 0:
    #   print(cyclist)    

    mymesh3 = mymesh.clone()
    mymesh3.m_label_mngr=label_mngr     
    mymesh3.m_vis.m_show_points=True
    mymesh3.m_vis.set_color_semanticgt()
    Scene.show(mymesh3,"gT")
    

    #myview.update()
    #recorder.record(str(batch_idx)+".png", "./recordings/spsequencenet/")

    while True:
      myview.update()

      if myview.m_new_cloud:
        #recorder.record(str(batch_idx)+".png", "./recordings/")
        myview.m_new_cloud = False
        break
  # print(max_diff_idx, " ",max_diff)
  # print(max_diff_idx_1, " ",max_diff_1)
  # print(max_diff_idx_2, " ",max_diff_2)
  # print("Movon motorcyclist here: ", cyclist)

# def findDifference():
#   base_path = "/workspace/schuett_temporal_lattice/predictions"
#   extras = ["gru_gru_cli_gru", "gru_gru_cli_cli", "lstm_lstm_cli_lstm"]
#   tmp = "sequences/08/predictions/"

#   meshes_path = os.path.join(base_path, extras[0], tmp)
#   #print(meshes_path)
#   #print(listdir(meshes_path))
#   files = [f for f in listdir(meshes_path) if isfile(join(meshes_path, f)) and "label" in f  ]
  
#   num_files = len(files)

#   for file in files:
#     load_pred_label(sequence, index, dir = "/workspace/schuett_temporal_lattice/predictions/gru_gru_cli_gru"):

#   return



if __name__ == "__main__":
  # findDifference()
  # exit()
  #config_file="/workspace/schuett_temporal_lattice/seq_config/lnn_eval_semantic_kitti.cfg"
  config_file="/workspace/schuett_temporal_lattice/seq_config/lnn_eval_semantic_kitti.cfg"
  config_parser = cfgParser(config_file)

  #eval_params=EvalParams.create(config_file)    
  #model_params=ModelParams.create(config_file)    

  # meshes_path="/workspace/schuett_temporal_lattice/predictions/tests/sequences/08/predictions"
  # files = [f for f in listdir(meshes_path) if isfile(join(meshes_path, f)) and "pred" in f  ]
  # files.sort()
  # files=natsort.natsorted(files,reverse=False)
  # print(files[-1])

  '''car=Mesh("/workspace/schuett_temporal_lattice/visualization_assets/car_passat_big.ply")
  car.V=car.V*0.5
  car.model_matrix.rotate_axis_angle_local([1,0,0],0)
  car.model_matrix.rotate_axis_angle_local([0,1,0],-90)
  car.translate_model_matrix([0, -1.5, 0])
  car.translate_model_matrix([0, 0, -11.1])
  Scene.show(car,"car")'''

  label_mngr_params = config_parser.get_label_mngr_vars()
  m_ignore_index = label_mngr_params["unlabeled_idx"]
  labels_file=str(label_mngr_params["labels_file"])
  colorscheme_file=str(label_mngr_params["color_scheme_file"])
  frequency_file=str(label_mngr_params["frequency_file"])
  label_mngr=LabelMngr(labels_file, colorscheme_file, frequency_file, m_ignore_index )


  myview=Viewer.create(config_file) #first because it needs to init context
  #myview.m_camera.from_string("56.1016 31.3023 43.6047 -0.185032  0.430075 0.0905343 0.878978 0 0 0 40 0.2 6004.45")
  #myview.m_camera.from_string("48.057  18.2299 -14.1026 -0.103289  0.793311  0.140475 0.58331   0.283752 -0.0689163   0.820265 40 0.2 6004.45")
  #myview.m_camera.from_string("48.057  18.2299 -14.1026 -0.103289  0.793311  0.140475 0.58331   0 0 0 40 0.2 6004.45")
  
  recorder=myview.m_recorder
  Scene.set_floor_visible(False)

  ## TODO: I can not change the color that is assigned to the predictions, because the color is in the ply files. I would have to create a new file format for it. 

  visMultipleDatasetPred(myview, label_mngr)
  #visCertainSequence(meshes_path, myview, label_mngr, None)