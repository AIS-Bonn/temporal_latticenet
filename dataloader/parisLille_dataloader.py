# Loader for the Paris-Lille Dataset 
# https://npm3d.fr/paris-lille-3d

# Based on the loader from: https://github.com/edwardzhou130/PolarSeg

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from plyfile import PlyData

from scipy.spatial.transform import Rotation as R
from xml.dom import minidom
from DataTransformer import *
from easypbr import *
import os,sys,inspect
import yaml

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

class ParisLille3DDataset(torch.utils.data.Dataset):
    def __init__(self,
            split,
            config_parser,
            sequence_learning):

        loader_config = config_parser.get_loader_paris_lille_vars()
        model_config = config_parser.get_model_vars()

        # generel information for the loader 
        self.split = split
        self.data_dir = loader_config['dataset_path']
        self.validation_cloud = loader_config['validation_cloud']
        self.nr_clouds_to_read = loader_config['nr_clouds_to_read']
        self.nr_clouds_to_skip = loader_config['nr_clouds_to_skip']
        self.do_overfit = loader_config['do_overfit'] 
        self.overfit_num_clouds = loader_config['overfit_num_clouds']
        self.debug_loader = loader_config['debug_loader'] 
        
        # stuff for get_item
        self.frame_num = loader_config['frames_per_seq'] if sequence_learning else 1
        self.feature_mode = model_config['values_mode']
        self.world_frame = loader_config['do_pose']
        self.cloud_scope = loader_config['cloud_scope']
        self.shuffle_points = loader_config['shuffle_points'] 
        self.accumulate_clouds = loader_config['accumulate_clouds'] 
        self.cap_distance = loader_config['cap_distance']
        self.seq_same_cloud = loader_config['seq_same_cloud']
        self.subsample_percentage = loader_config['subsample_percentage']
        self.transformer = DataTransformer(config_parser = config_parser, split = self.split)

        if self.split == "train" or self.split == "test": 
            print("Frame num: ",  self.frame_num)
            print("Cloud scope: ", self.cloud_scope)

        sample_interval = 2 # how many seconds of measurements one cloud should accumulate
        time_step = 1       # how many clouds one sample cloud should result in (e.g. time_step = 2 results in 2 clouds from one sample_interval)

        # class id mappings from the originals to the 10 (12) we are using
        if loader_config["include_moving_classes"] and not self.split == "test":
            xmldoc = minidom.parse(loader_config['xml_config_all'])
        else:
            print("Using loader: ", loader_config['xml_config'])
            xmldoc = minidom.parse(loader_config['xml_config'])
        itemlist = xmldoc.getElementsByTagName('class')
        self.class2coarse = np.array([[int(i.attributes['id'].value),int(i.attributes['coarse'].value)] for i in itemlist]).astype(np.uint32)
        #coarseID_name={int(i.attributes['coarse'].value):i.attributes['coarse_name'].value for i in itemlist}
        #for i in range(len(coarseID_name)-1): coarseID_name[i] = coarseID_name.pop(i+1)        
        yaml_config = loader_config["yaml_config"]    # ParisLille3D only does tests on 10 coarse classes - we train with 12 (moving-cars, moving-persons added) and therefore need to remap these 12 to the corresponding 10 classes
        DATA = yaml.safe_load(open(yaml_config, 'r'))
        class_remap = DATA["learning_map_inv"]
        maxkey = max(class_remap.keys())
        self.remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        self.remap_lut[list(class_remap.keys())] = list(class_remap.values()) #usage: labels = remap_lut[labels]

        # find all ply files that are available
        ply_files = []
        for file in os.listdir(self.data_dir):
            if file.endswith(".ply"):
                ply_files.append(os.path.join(self.data_dir, file))

        self.xyz_list, self.reflectance_list, self.class_list, self.start_end_list, self.dataset_lengths, self.dataset_names = [], [], [], [], [], []

        # we save the data as npy files to be loaded easier and faster
        npy_folder =  os.path.join(self.data_dir, "npys")
        npy_folder += "_{training}{moving}".format(training = self.split, moving = "_moving" if loader_config["include_moving_classes"] else "")

        if os.path.exists(npy_folder):
            print("##### Loading Dataset from npy files under {}#####".format(npy_folder))
            self.xyz_list = np.load( os.path.join(npy_folder, "xyz_list.npy"), allow_pickle = True)
            self.reflectance_list = np.load( os.path.join(npy_folder, "reflectance_list.npy"), allow_pickle = True)
            self.class_list = np.load( os.path.join(npy_folder, "class_list.npy"), allow_pickle = True)
            self.start_end_list = np.load( os.path.join(npy_folder, "start_end_list.npy"), allow_pickle = True)
            self.dataset_lengths = np.load( os.path.join(npy_folder, "dataset_lengths.npy"), allow_pickle = True)
            self.dataset_names = np.load( os.path.join(npy_folder, "dataset_names.npy"), allow_pickle = True)
        else:
            if self.split == "train": 
                print("##### Preparing Dataset ####### \n This may take a while, but only needs to be done once ...")
            os.mkdir(npy_folder)
            for ply_file in ply_files:

                if (self.split == "train") and (ply_file.endswith(self.validation_cloud)):
                    continue
                
                if (self.split == "valid") and (not ply_file.endswith(self.validation_cloud)):
                    continue

                print(self.split, " ", ply_file)

                # Load point cloud and labels
                plydata = PlyData.read(ply_file)
                self.dataset_names.append(ply_file)
                origins = np.array(np.transpose(np.stack((plydata['vertex']['x_origin'],plydata['vertex']['y_origin'],plydata['vertex']['z_origin'])))).astype(np.float32) 

                # we center the cloud around the first point
                self.xyz_list.append( np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'])))).astype(np.float32)  - origins[0,:]  )  
                
                if self.feature_mode == "reflectance":
                    self.reflectance_list.append( np.array(plydata['vertex']['reflectance']).astype(np.float32) )
                
                try:
                    class_id = np.array(plydata['vertex']['class'])
                except:
                    class_id = np.zeros_like(plydata['vertex']['x'],dtype=int)
                
                if loader_config['fifty_classes'] == False and not self.split == "test":  
                    class_id = self.__PLfine2coarse__(class_id)
                #print("Class id uniques: ", np.unique(class_id))
                self.class_list.append(class_id)
                
                # parse data by timestamp
                GPS_time = plydata['vertex']['GPS_time']
                sample_start_times = np.arange(GPS_time[0]+sample_interval,GPS_time[-1]-sample_interval,time_step)
                start_ind = np.searchsorted(GPS_time,sample_start_times - sample_interval)
                end_ind = np.searchsorted(GPS_time,sample_start_times + sample_interval)
                end_ind[-1] = np.size(GPS_time)

                start_end = np.transpose(np.stack((start_ind,end_ind)))
                start_end = np.unique(start_end,axis=0)
                #remain_ind = (start_end[:,1] - start_end[:,0]) > 1000
                #start_end = start_end[remain_ind,:]

                if self.cap_distance > 0:
                    remain_ind = (start_end[:,1] - start_end[:,0]) > self.cap_distance
                    start_end = start_end[remain_ind,:]
                self.start_end_list.append(start_end)
                self.dataset_lengths.append(start_end.shape[0])       

            self.xyz_list, self.reflectance_list, self.class_list, self.start_end_list, self.dataset_lengths, self.dataset_names = np.asarray(self.xyz_list), np.asarray(self.reflectance_list), np.asarray(self.class_list), np.asarray(self.start_end_list), np.asarray(self.dataset_lengths), np.asarray(self.dataset_names)
            np.save( os.path.join(npy_folder, "xyz_list"), self.xyz_list, allow_pickle = True)
            np.save( os.path.join(npy_folder, "reflectance_list"), self.reflectance_list, allow_pickle = True)
            np.save( os.path.join(npy_folder, "class_list"), self.class_list, allow_pickle = True)
            np.save( os.path.join(npy_folder, "start_end_list"), self.start_end_list, allow_pickle = True)
            np.save( os.path.join(npy_folder, "dataset_lengths"), self.dataset_lengths, allow_pickle = True)
            np.save( os.path.join(npy_folder, "dataset_names"), self.dataset_names, allow_pickle = True)


        self.dataset_lengths = np.asarray(self.dataset_lengths)
        self.dataset_size = np.sum(self.dataset_lengths)

        if self.nr_clouds_to_read == -1: # -1 means all
            self.dataset_size = np.sum(self.dataset_lengths) - self.nr_clouds_to_skip
        else:
            self.dataset_size = self.nr_clouds_to_read
        
        if self.do_overfit:
            self.dataset_size = self.overfit_num_clouds

        # this only loads one specific cloud sequence
        if self.debug_loader:
            print("------------------- DEBUGGING DATALOADER -------------------")
            self.dataset_size = 1


    def __remapTwelveToTenClasses__(self, labels):
        'Remap the label array from 12 moving classes to the 10 coarse classes'
        return self.remap_lut[labels]
       
    
    def __PLfine2coarse__(self, label_array):
        new_label = label_array.copy()
        for i in range(self.class2coarse.shape[0]):
            new_label[label_array == self.class2coarse[i,0]] = self.class2coarse[i,1]
        return np.uint8(new_label)
  

    def __len__(self):
        'Denotes the total number of samples'
        return self.dataset_size


    def __getitem__(self,index):
        cloud_seq, scan_seq, feature_seq, label_seq, path_seq, len_seq = [],[],[],[],[],[]

        real_idx, dataset_idx = index, None

        for idx, i in enumerate(self.dataset_lengths):
            if real_idx < i:
                dataset_idx = idx
                break
            real_idx -= i

        indeces = (np.arange(self.frame_num)-(self.frame_num-1))*self.cloud_scope
        real_indeces = np.maximum(indeces + real_idx, 0)
        #print(real_indeces)
        for idx in real_indeces:
            scan, reflectance, label, path = self.__getSingleItem__(dataset_idx, idx)
            scan_seq.append(scan)
            feature_seq.append(reflectance)
            label_seq.append(label)
            path_seq.append(path)

        scan_seq = self.transformer.transform(scan_seq)
        if not self.accumulate_clouds:
            return scan_seq, feature_seq, label_seq, path_seq, len_seq
        else:
            return torch.cat(scan_seq), torch.cat(feature_seq), torch.cat(label_seq), path_seq, len_seq


    def __getSingleItem__(self,dataset_idx, index):
        'Generates one sample of data'
        # Select sample
        start_ind,end_ind = self.start_end_list[dataset_idx][index,:]
        data_indexes = range(start_ind,end_ind)
                
        scan = np.float32(self.xyz_list[dataset_idx][data_indexes,...]).copy()
        #scan = scan - np.mean(scan,keepdims = True,axis = 0)
        
        r = R.from_euler('x', -90, degrees=True) # z should be forward
        scan = r.apply(scan) 

        if self.split != "test":
            label = np.int32(self.class_list[dataset_idx][data_indexes]).copy()
        else: # for test split
            label = np.zeros((scan.shape[0]))

        path = self.dataset_names[dataset_idx]
        if self.feature_mode == "reflectance":
            reflectance = self.reflectance_list[dataset_idx][data_indexes,np.newaxis]
        else:
            reflectance = np.ones((scan.shape[0], 1))

        if self.shuffle_points:
            randomize = np.arange(scan.shape[0])
            np.random.shuffle(randomize)
            scan = scan[randomize,:]
            label = label[randomize]
            reflectance = reflectance[randomize]
        
        if self.subsample_percentage > 0 and scan.shape[0] > 100000:
            scan, reflectance, label = random_subsample(scan, reflectance, label, self.subsample_percentage)

        # scan gets converted to tensor by the DataTransformer
        return scan, torch.tensor(reflectance, dtype = torch.float), torch.tensor(label, dtype = torch.long), path



if __name__ == "__main__":
    config_file="/workspace/schuett_temporal_lattice/seq_config/lnn_train_paris_lille.cfg"
    config_parser = cfgParser(config_file)

    view=Viewer.create(config_file)
    label_mngr_params = config_parser.get_label_mngr_vars()


    dataset = ParisLille3DDataset("train", config_parser, True)

    dataloader = torch.utils.data.DataLoader(dataset,num_workers = 1, batch_size=1, shuffle = True)

    iterator = dataloader.__iter__()

    for batch_idx, (positions_seq, feature_seq, labels_seq, path_seq) in enumerate(iterator):

        for i in range(0,len(positions_seq)):

            positions = positions_seq[i].squeeze(0).numpy()
            feature = feature_seq[i].squeeze(0).numpy()
            labels = labels_seq[i].squeeze(0).numpy()

            cloud = Mesh()
            cloud.V = positions
            cloud.m_vis.m_show_points=True
            color = [1.0, 0.0, 0.0]
            #color[i] = 1.0
            cloud.m_vis.m_point_color= color
            Scene.show(cloud, "Peer_Cloud_"+str(i))
            view.update()

            continue
        
        while True:
            view.update()

        
