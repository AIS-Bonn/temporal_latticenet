import torch 
import torchvision.transforms as transforms
import configparser
from cfgParser import *
import numpy as np
import random
from scipy.spatial.transform import Rotation as R

# this data is taken from the file "calib_velo_to_cam.txt"
def load_velo_to_cam_transform():
    R_velo_to_cam = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04], [1.480249e-02, 7.280733e-04, -9.998902e-01], [9.998621e-01, 7.523790e-03, 1.480755e-02]])
    trans_velo_to_cam = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])
    T_velo_to_cam = np.identity(4)
    T_velo_to_cam[:3,:3] = R_velo_to_cam
    T_velo_to_cam[:3,3] = trans_velo_to_cam
    return T_velo_to_cam

# angle in degrees, axis is either x,y or z
def rotation_matrix(angle, axis):
    r=None
    if axis == "x":
        r = R.from_euler('X', angle, degrees=True).as_matrix()
    elif axis == "y":
        r = R.from_euler('Y', angle, degrees=True).as_matrix()
    elif axis == "z":
        r = R.from_euler('Z', angle, degrees=True).as_matrix()
    else:
        print("Axis has to be either x,y or z")
    T = np.identity(4)
    T[:3,:3] = r
    return T

# subsamples the point cloud a certain nr of times by randomly dropping points. If percentage_removal is 1 then we remove all the points, if it's 0 then we keep all points
def random_subsample(cloud, reflectance = None, label = None, percentage_removal = 0.0):
    
    prob_of_death=1.0-percentage_removal
    vertices_marked_for_removal=0
    is_vertex_to_be_removed = np.zeros((cloud.shape[0]), dtype = np.int) #(V.rows(), false);
    for i in range(0, cloud.shape[0]): #)for(int i = 0; i < V.rows(); i++){
        rand = random.uniform(0, 1)
        if(rand < prob_of_death):
            is_vertex_to_be_removed[i] = 1
            vertices_marked_for_removal += 1

    if reflectance is not None and label is not None:
        return cloud[is_vertex_to_be_removed == 1], reflectance[is_vertex_to_be_removed == 1], label[is_vertex_to_be_removed == 1]
    elif reflectance is not None:
        return cloud[is_vertex_to_be_removed == 1], reflectance[is_vertex_to_be_removed == 1]
    elif label is not None:
        return cloud[is_vertex_to_be_removed == 1], label[is_vertex_to_be_removed == 1]
    else:
        return cloud[is_vertex_to_be_removed == 1]

# This class is used for data augmentation
class DataTransformer():

    def __init__(self, config_parser, split = "train"):
        transformer_config = config_parser.get_transformer_vars()
        # train_config = config_parser.get_train_vars()

        self.m_random_translation_xyz_magnitude=transformer_config["random_translation_xyz_magnitude"]
        self.m_random_translation_xz_magnitude=transformer_config["random_translation_xz_magnitude"]
        self.m_rotation_y_max_angle=transformer_config["rotation_y_max_angle"]
        self.m_random_stretch_xyz_magnitude=transformer_config["random_stretch_xyz_magnitude"]
        self.m_adaptive_subsampling_falloff_start=transformer_config["adaptive_subsampling_falloff_start"]
        self.m_adaptive_subsampling_falloff_end=transformer_config["adaptive_subsampling_falloff_end"]
        self.m_random_subsample_percentage=transformer_config["random_subsample_percentage"]
        self.m_random_mirror_x=transformer_config["random_mirror_x"]
        self.m_random_mirror_z=transformer_config["random_mirror_z"]
        self.m_random_rotation_90_degrees_y=transformer_config["random_rotation_90_degrees_y"]

        self.m_hsv_jitter=transformer_config["hsv_jitter"]
        self.m_chance_of_xyz_noise = transformer_config["chance_of_xyz_noise"]
        self.m_xyz_noise_stddev=transformer_config["xyz_noise_stddev"]

        self.split = split

    # transforms the points in the cloud randomly (data augmentation)
    # Input: scan_seq
    # each seperate array has shape [i,3]
    # Outputs: Augmented clouds as pytorch tensors
    def transform(self, clouds):
        
        # only do transformation for the training examples
        if self.split != "train": 
            for i in range(0,len(clouds)):
                clouds[i] = torch.tensor(clouds[i], dtype = torch.float) 
            return clouds
 
        if(self.m_adaptive_subsampling_falloff_end!=0.0):
            assert self.m_adaptive_subsampling_falloff_start<self.m_adaptive_subsampling_falloff_end , str(" The falloff for the adaptive subsampling start should be lower than the end. For example we start at 0 meters and we end at 60m. The start is " + self.m_adaptive_subsampling_falloff_start + " and the end is " + self.m_adaptive_subsampling_falloff_end)
            pass
        
        if(self.m_random_subsample_percentage!=0.0):
            for i in range(0, len(clouds)):                
                subsample_mask = np.random.choice(a = [False, True], size = (clouds[i].shape[0]), p = [self.m_random_subsample_percentage, 1-self.m_random_subsample_percentage])
                clouds[i] = clouds[i][subsample_mask]
            
        if(self.m_random_translation_xyz_magnitude!=0.0):
            translation = np.random.rand(3)* self.m_random_translation_xyz_magnitude
            for i in range(0, len(clouds)):
                    clouds[i][:] = clouds[i][:] + translation

        if(self.m_random_translation_xz_magnitude!=0.0):
            translation = np.random.rand(3)* self.m_random_translation_xz_magnitude
            translation[1] = 0
            for i in range(0, len(clouds)):
                    clouds[i][:] = clouds[i][:] + translation

        if(self.m_random_stretch_xyz_magnitude!=0.0):
            s = stretch_factor_x = 1.0 + random.uniform(-self.m_random_stretch_xyz_magnitude, self.m_random_stretch_xyz_magnitude)
            stretch_factor_x = 1.0 + random.uniform(-s, s)
            stretch_factor_y = 1.0 + random.uniform(-s, s)
            stretch_factor_z = 1.0 + random.uniform(-s, s)

            for i in range(0, len(clouds)):
                clouds[i][:,0] *= stretch_factor_x 
                clouds[i][:,1] *= stretch_factor_y
                clouds[i][:,2] *= stretch_factor_z


        if(self.m_rotation_y_max_angle!=0):
            rand_angle_degrees = random.uniform(-self.m_rotation_y_max_angle/2.0, self.m_rotation_y_max_angle/2.0)
            r = R.from_euler('Y', rand_angle_degrees, degrees=True).as_matrix()
            for i in range(0, len(clouds)):
                clouds[i] = (r @ clouds[i].transpose()).transpose()

        if(self.m_random_mirror_x):
            do_flip = random.random() < 0.5
            if do_flip:
                for i in range(0, len(clouds)):
                    clouds[i][:,0] = - clouds[i][:,0]    

        if(self.m_random_mirror_z):
            do_flip = random.random() < 0.5
            if do_flip:
                for i in range(0, len(clouds)):
                    clouds[i][:,2] = - clouds[i][:,2]    

        if(self.m_random_rotation_90_degrees_y):
            nr_times = random.randint(0,3)
            r = R.from_euler('Y', 90*nr_times, degrees=True).as_matrix()
            for i in range(0, len(clouds)):
                    clouds[i] = (r @ clouds[i].transpose()).transpose() 

        if (self.m_hsv_jitter==0):
            pass

        do_xyz_noise = random.random() < self.m_chance_of_xyz_noise
        if(do_xyz_noise):
            if (self.m_xyz_noise_stddev != 0):
                pass

        for i in range(0,len(clouds)):
            clouds[i] = torch.tensor(clouds[i], dtype = torch.float)

        return clouds



if __name__ == "__main__":
    config_file="/workspace/schuett_temporal_lattice/config/lnn_train_semantic_kitti.cfg"
    config_parser = cfgParser(config_file)
    dt = DataTransformer(config_parser)

    clouds = []
    cloud1 = np.array([[1,2,3], [1,2,3]])
    cloud2 = np.array([[3,2,3], [3,2,3]])
    clouds.append(cloud1)
    clouds.append(cloud2)
    print(cloud1)

    clouds = dt.transform(clouds)
