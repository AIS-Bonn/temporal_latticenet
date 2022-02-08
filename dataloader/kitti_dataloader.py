# Loader for the SemanticKITTI dataset 
# http://www.semantic-kitti.org/

import torch
import numpy as np
from numpy.linalg import inv
import os,sys,inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

#from datetime import datetime
from DataTransformer import *

# to read kitti data: https://github.com/utiasSTARS/pykitti
import yaml
from easypbr import *


class SemanticKittiDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            split,
            config_parser,
            sequence_learning):

        loader_config = config_parser.get_loader_semantic_kitti_vars()
        model_config = config_parser.get_model_vars()

        self.split = split
        self.data_dir = loader_config['dataset_path']
        if loader_config["include_moving_classes"]:
            yaml_config = loader_config["yaml_config_all"]            
        else:
            yaml_config = loader_config["yaml_config"]

        DATA = yaml.safe_load(open(yaml_config, 'r'))        
        self.split_seqs = DATA["split"]
        self.split_lengths = DATA["split_lengths"]
        self.synthkitti = loader_config["synthkitti"]

        class_remap = DATA["learning_map"]
        # make lookup table for mapping
        maxkey = max(class_remap.keys())
        # +100 hack making lut bigger just in case there are unknown labels
        self.remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        self.remap_lut[list(class_remap.keys())] = list(class_remap.values())

        self.dataset_lengths = self.split_lengths[self.split] # we need to know how many scans each sequence has
        self.sequences = loader_config['sequences']

        self.nr_clouds_to_read = loader_config['nr_clouds_to_read']
        self.nr_clouds_to_skip = loader_config['nr_clouds_to_skip']
        self.do_overfit = loader_config['do_overfit'] 
        self.overfit_num_clouds = loader_config['overfit_num_clouds']
        self.debug_loader = loader_config['debug_loader'] 
        
        # parameters for get_item
        self.T_velo_to_cam = load_velo_to_cam_transform()
        self.frame_num = loader_config['frames_per_seq'] if sequence_learning else 1
        self.feature_mode = model_config['values_mode']
        self.world_frame = loader_config['do_pose']
        self.cloud_scope = loader_config['cloud_scope']
        self.shuffle_points = loader_config['shuffle_points'] 
        self.accumulate_clouds = loader_config['accumulate_clouds'] 
        self.cap_distance = loader_config['cap_distance']
        self.min_distance = loader_config['min_distance']
        self.seq_same_cloud = loader_config['seq_same_cloud']

        self.transformer = DataTransformer(config_parser = config_parser, split = self.split)
                
        if self.nr_clouds_to_read == -1: # -1 means all
            self.dataset_size = np.sum(self.dataset_lengths) - self.nr_clouds_to_skip
        else:
            self.dataset_size = self.nr_clouds_to_read
        
        if self.do_overfit:
            if self.split == "train":
                print("------------------- OVERFITTING -------------------")
            self.dataset_size = self.overfit_num_clouds
        elif self.debug_loader:
            if self.split == "train":
                print("------------------- DEBUGGING DATALOADER -------------------")
            self.dataset_size = 1
        elif self.synthkitti:
            if self.split == "train":
                print("------------------- SYNTHETIC SET -------------------")
            self.dataset_size = 1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        if self.synthkitti:
            return self.__getSynthitem__(index)

        if self.debug_loader:
            index = 4541 + 1101+4661+801+200 # needs to be rather small for the val set to work
            assert index < np.sum(self.dataset_lengths), "The debug index you want to use is too big for this dataset"
        
        is_training = True if self.split == "train" else False
        index += self.nr_clouds_to_skip # nr_clouds_to_skip is the offset. In most cases it is 0

        scan_seq, feature_seq, label_seq, path_seq,len_seq = [],[],[],[],[]
        # for cloud_scope = 3 it would be [-6,-3,0]
        indeces = (np.arange(self.frame_num)-(self.frame_num-1))*self.cloud_scope
        if self.seq_same_cloud:
            indeces = np.zeros(self.frame_num, dtype = np.int)

        real_indeces, dataset_idx, seq = None, None, None
        cum_lengths = np.cumsum(self.dataset_lengths)
        last_cumsum = 0

        for i, cumsum in enumerate(cum_lengths):
            if index < cumsum:
                seq = int(self.split_seqs[self.split][i])
                real_idx = index - last_cumsum # if last_cumsum != 0 else index

                dataset_idx = i
                real_indeces = np.maximum(indeces + real_idx, 0)
                break
            last_cumsum = cumsum

        dataset_idx = self.split_seqs[self.split][dataset_idx]
        #print(dataset_idx)
        #print("Indices are: ", real_indeces)

        cam_to_world_first_scan = get_velo_to_world_pose_scan(self.data_dir, dataset_idx, real_indeces[0])
        for i in range(0, real_indeces.shape[0]):
            idx = real_indeces[i]
            scan, label, feature, reflectance = None, None, None, None

            filename = os.path.join(self.data_dir,'sequences', '{:02d}'.format(seq), 'velodyne', '{:06d}.bin'.format(idx))
            assert(os.path.isfile(filename)), str("Filename not found: ", filename)
            scan_xyzref = np.fromfile(filename, dtype=np.float32).reshape(-1,4).transpose()
            reflectance = scan_xyzref[3,:]
            scan_xyz = scan_xyzref[0:3,:]

            # for semantic kitti we do not have the labels for the test split
            if self.split == "test":
                label = np.zeros((scan_xyz.shape[1])) 
            elif self.debug_loader:
                label = np.ones((scan_xyz.shape[1]))*i # color to distinguish between the clouds of the sequence
            else:
                label = load_label(seq, idx, self.remap_lut, dir = self.data_dir)
                label = np.squeeze(label, axis = 1)   

            if (self.cap_distance >= 0) and is_training:
                length = np.linalg.norm(scan_xyz, axis = 0)
                mask = length < self.cap_distance # remove all points that are too far away
                scan_xyz = scan_xyz[:,mask]
                label = label[mask]
                reflectance = reflectance[mask]

            if (self.min_distance >= 0) and is_training:
                length = np.linalg.norm(scan_xyz, axis = 0)
                mask = length > self.min_distance # remove all points that are too far away
                scan_xyz = scan_xyz[:,mask]
                label = label[mask]
                reflectance = reflectance[mask]

            scan_xyz_homogenous = np.ones((4, scan_xyz.shape[1]))
            scan_xyz_homogenous[0:3,:] = scan_xyz       

            if self.world_frame:
                scan_world = np.matmul(get_velo_to_world_pose_scan(self.data_dir, dataset_idx, idx), scan_xyz_homogenous) # velo to world coordinate frame -> get_pose_scan returns the velo to world difference
                
                #get the scans from the world coord frame of the last scan in the sequence 
                world_to_cam_last_scan = np.linalg.inv( cam_to_world_first_scan )
                scan_world = np.matmul(world_to_cam_last_scan, scan_world) #goes from world to the coord system of the last scan in the sequence 
                scan_ros = scan_world
                scan_ros = np.matmul(rotation_matrix(-90, "x"), scan_world)
                scan = scan_ros[0:3,:] / scan_ros[3,:] # divide by w (homogenous coordinates are x,y,z,w)
            else:
                scan_ros = np.matmul(rotation_matrix(-90, "x"), scan_xyz_homogenous)# angle 1.0
                scan = scan_ros

            scan = scan[0:3,:].transpose()  # shape has to be [i,3]

            # same shuffle for all arrays
            if (self.shuffle_points) and is_training:
                randomize = np.arange(scan.shape[0])
                np.random.shuffle(randomize)
                scan = scan[randomize,:]
                label = label[randomize]
                reflectance = reflectance[randomize]

            # append xyz, features, label
            if self.feature_mode == "reflectance":
                feature = np.expand_dims(reflectance, axis=1)
            else:   # if we do not want to use any features we just pass a 1 vector to the network
                feature = np.ones((reflectance.shape[0],1), dtype = float)

            label = torch.tensor(label, dtype = torch.long)
            feature = torch.tensor(feature, dtype = torch.float)
            
            scan_seq.append(scan)
            label_seq.append(label)
            path_seq.append(filename)
            feature_seq.append(feature)
            len_seq.append(scan.shape[0])

        scan_seq = self.transformer.transform(scan_seq)
        if not self.accumulate_clouds:
            return scan_seq, feature_seq, label_seq, path_seq, len_seq
        else:
            return torch.cat(scan_seq), torch.cat(feature_seq), torch.cat(label_seq), path_seq, len_seq


    # return a sequence of length self.frame_num that uses the same cloud and adds a moving car to the scene   
    def __getSynthitem__(self, index):
        scan_seq, feature_seq, label_seq, path_seq = [],[],[],[]
        index = 1
        indeces = np.zeros(self.frame_num, dtype = np.int)

        real_indeces = None
        dataset_idx = None
        seq = None

        cum_lengths = np.cumsum(self.dataset_lengths)
        last_cumsum = 0

        for i, cumsum in enumerate(cum_lengths):
            if index < cumsum:
                seq = int(self.split_seqs[self.split][i])
                real_idx = index - last_cumsum # if last_cumsum != 0 else index

                dataset_idx = i
                real_indeces = np.maximum(indeces + real_idx, 0)
                break
            last_cumsum = cumsum

        for i in range(0, real_indeces.shape[0]):
            idx = real_indeces[i]
            scan, label, feature = None, None, None

            filename = os.path.join(self.data_dir,'sequences', '{:02d}'.format(seq), 'velodyne', '{:06d}.bin'.format(idx))
            assert(os.path.isfile(filename)), str("Filename not found: ", filename)
            
            scan_xyzref = np.fromfile(filename, dtype=np.float32).reshape(-1,4).transpose()
            label = load_label(seq, idx, self.remap_lut, dir = self.data_dir)
            label[label == 20] = 1 # we have only this static cloud, therefore all possible other "moving cars" are static

            scan_xyz = scan_xyzref[0:3,:]
            label = np.squeeze(label, axis = 1)  

            scan_xyz_homogenous = np.ones((4, scan_xyz.shape[1]))
            scan_xyz_homogenous[0:3,:] = scan_xyz    

            scan = np.matmul(rotation_matrix(-90, "x"), scan_xyz_homogenous)# angle 1.0
            scan = scan[0:3,:].transpose()  # shape has to be [i,3]

            # add the car to the scene
            car=Mesh("/workspace/schuett_temporal_lattice/visualization_assets/car_passat.ply")
            car_vertices = random_subsample(cloud = car.V, percentage_removal = 0.4)
            r = R.from_euler('z', 90, degrees=True)
            car_vertices = r.apply(car_vertices)
            r = R.from_euler('x', -90, degrees=True)
            car_vertices = r.apply(car_vertices)
            car_vertices += np.array([0,-0.8,0]) # shift car onto road
            car_vertices += np.array([18 - i*3.5,0,0]) # to create a "driving car" scenario the car model has to move

            scan = np.append(scan, car_vertices, axis = 0)
            label = np.append(label, np.ones((car_vertices.shape[0],))*20, axis = 0) #moving cars have label 20
            feature = np.ones((label.shape[0],1), dtype = float)

            if self.shuffle_points:
                randomize = np.arange(scan.shape[0])
                np.random.shuffle(randomize)
                scan = scan[randomize,:]
                label = label[randomize]

            #print(label)
            label = torch.tensor(label, dtype = torch.long)
            feature = torch.tensor(feature, dtype = torch.float)

            scan_seq.append(scan)
            label_seq.append(label)
            path_seq.append(filename)
            feature_seq.append(feature)

        scan_seq = self.transformer.transform(scan_seq)
        return scan_seq, feature_seq, label_seq, path_seq, _



def parse_calibration(filename):
  """ read calibration file with given filename
      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
  """
  calib = {}

  calib_file = open(filename)
  for line in calib_file:
    key, content = line.strip().split(":")
    values = [float(v) for v in content.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    calib[key] = pose

  calib_file.close()

  return calib

def parse_poses(filename, calibration):
  """ read poses file with per-scan poses from given filename
      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
  file = open(filename)

  poses = []

  Tr = calibration["Tr"]
  Tr_inv = inv(Tr)

  for line in file:
    values = [float(v) for v in line.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0
    #print("Pose: ", np.matmul(pose, Tr))
    #print("Changed: ", np.matmul(Tr_inv, np.matmul(pose, Tr)))

    poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

  return poses

def get_velo_to_world_pose_scan(data_dir, sequence_id, scan_number):
    calibration = parse_calibration(os.path.join(data_dir,"sequences","{:02d}".format(sequence_id), "calib.txt"))
    poses = parse_poses(os.path.join(data_dir,"sequences","{:02d}".format(sequence_id), "poses.txt"), calibration)
    return poses[scan_number]

def get_cam_to_world_pose_scan(data_dir, sequence_id, scan_number):
    filename = os.path.join(data_dir,"sequences","{:02d}".format(sequence_id), "poses.txt")
    file = open(filename)
    poses = []

    for line in file:
        values = [float(v) for v in line.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(pose)

    return poses[scan_number]

def load_label(sequence, index, remap_lut, dir):
    filename = os.path.join(dir,'sequences', '{:02d}'.format(sequence), 'labels', '{:06d}.label'.format(index))
    assert(os.path.isfile(filename)), str("Filename not found: ", filename)
    npz = np.fromfile(filename, dtype=np.uint16)
    
    # Lower 16 bits: Label. 
    # Upper 16 bits: Instance id
    labels = (npz[0::2].reshape(len(npz)//2, 1)).astype(np.int16)
    labels = remap_lut[labels] # the labels in the normal dataset are not in the range 0-19 or 0-25 (see website for more information)
    
    return labels

# vis_target is only relevant for debugging
def create_cloud(positions, target, cloud_path, label_mngr, pred_softmax = None, vis_target = None):
    cloud = Mesh()
    cloud.V = positions.clone().detach().cpu().numpy()
    cloud.L_gt = target.clone().detach().cpu().numpy() if target is not None else None
    cloud.L_pred = pred_softmax.detach().argmax(axis=1).cpu().numpy() if pred_softmax is not None else None
    # cloud.L_pred = vis_target if vis_target is not None # only debug
    
    cloud.m_vis.m_point_size=4
    cloud.m_vis.set_color_semanticpred()

    #some sensible visualization options
    cloud.m_vis.m_show_mesh=False
    cloud.m_vis.m_show_points=True

    #set the labelmngr which will be used by the viewer to put correct colors for the semantics
    cloud.m_label_mngr=label_mngr
    cloud.m_disk_path= cloud_path

    return cloud


# the main function visualizes the Dataset
if __name__ == "__main__":
    config_file="/workspace/schuett_temporal_lattice/seq_config/lnn_train_semantic_kitti.cfg"
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
    config_parser = cfgParser(config_file)

    view=Viewer.create(config_file) #first because it needs to init context
    view.m_camera.from_string("56.1016 31.3023 43.6047 -0.185032  0.430075 0.0905343 0.878978 0 0 0 40 0.2 6004.45")
    recorder=view.m_recorder

    label_mngr_params = config_parser.get_label_mngr_vars()
    m_ignore_index = label_mngr_params["unlabeled_idx"]
    labels_file=str(label_mngr_params["labels_file"])
    colorscheme_file=str(label_mngr_params["color_scheme_file"])
    frequency_file=str(label_mngr_params["frequency_file"])
    label_mngr=LabelMngr(labels_file, colorscheme_file, frequency_file, m_ignore_index )


    train_dataset = SemanticKittiDataset(split = "train", config_parser = config_parser, sequence_learning = True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers = 1, batch_size=1, shuffle = False)

    loader_iter = train_dataloader.__iter__()
    for batch_idx, (positions_seq, values_seq, target_seq, path_seq) in enumerate(loader_iter):
        
        for i in range(0,len(positions_seq)):
            #print(path_seq)
            positions = positions_seq[i].squeeze(0)
            values = values_seq[i].squeeze(0)
            target = target_seq[i].squeeze(0)

            cloud = create_cloud(positions, None, path_seq[i][0], label_mngr, vis_target = target)

            Scene.show(cloud,str("mesh_{0}").format(i))
            view.update()
        
    while True:
        view.update()
