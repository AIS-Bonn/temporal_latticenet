import torch
from torch.autograd import Function
from torch import Tensor
from torch.nn import functional as F

import sys
from latticenet  import HashTable
from latticenet  import Lattice
import numpy as np
import time
import math
import torch_scatter
# from latticenet_py.lattice.lattice_py import LatticePy
from latticenet_py.lattice.lattice_funcs import *
from latticenet_py.lattice.lattice_modules import *

class LSTMModule(torch.nn.Module):
    def __init__(self, nr_output_channels):
        super(LSTMModule, self).__init__()
        self.lstm = torch.nn.LSTMCell(input_size = nr_output_channels, hidden_size = nr_output_channels, bias = True) # https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html

        self.hidden_linear = torch.nn.Linear(nr_output_channels, nr_output_channels)
        self.h_lv = None

    def reset_sequence(self):
        self.h_lv = None

    def forward(self, lv, ls):
        if (self.h_lv == None):
            self.h_lv = lv.clone()
        else:
            self.h_lv = self.hidden_linear(self.h_lv)
            h_lv_padded = torch.nn.utils.rnn.pad_sequence([self.h_lv, lv], padding_value = 0.0)
            h_lv = h_lv_padded[:,0,:].squeeze()

            lv, _ = self.lstm(lv, (h_lv, torch.zeros_like(h_lv))) # we dont use the next cell state, therefore we set it to zero
            self.h_lv = lv.clone()
            ls.set_values(lv) 

        return lv, ls

class GRUModule(torch.nn.Module):
    def __init__(self, nr_output_channels):
        super(GRUModule, self).__init__()
        self.GRU = torch.nn.GRUCell(input_size = nr_output_channels, hidden_size = nr_output_channels, bias = True) # https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html

        self.hidden_linear = torch.nn.Linear(nr_output_channels, nr_output_channels)
        self.h_lv = None

    def reset_sequence(self):
        self.h_lv = None

    def forward(self, lv, ls):
        if (self.h_lv == None):
            new_lv = lv.clone()
            self.h_lv = lv.clone()
        else:
            self.h_lv = self.hidden_linear(self.h_lv)
            h_lv_padded = torch.nn.utils.rnn.pad_sequence([self.h_lv, lv], padding_value = 0.0)
            h_lv = h_lv_padded[:,0,:].squeeze()

            new_lv = self.GRU(lv, h_lv) # we dont use the next cell state, therefore we set it to zero
            self.h_lv = new_lv.clone()
            ls.set_values(new_lv) 

        return new_lv, ls


# SpSequenceNet: https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_SpSequenceNet_Semantic_Segmentation_Network_on_4D_Point_Clouds_CVPR_2020_paper.pdf
class CrossframeGlobalAttentionModule(torch.nn.Module):
    def __init__(self, nr_output_channels):
        super(CrossframeGlobalAttentionModule, self).__init__()
        self.relu=torch.nn.ReLU(inplace=False)
        self.sigmoid=torch.nn.Sigmoid()
        self.groupnorm= Gn()
        self.conv=Conv1x1(out_channels=nr_output_channels, bias=False) 

        self.hidden_linear = torch.nn.Linear(nr_output_channels, nr_output_channels)
        self.h_lv = None

    def reset_sequence(self):
        self.h_lv = None

    def forward(self, lv, ls):

        if (self.h_lv == None):
            self.h_lv = lv.clone()
        else:
            self.h_lv = self.hidden_linear(self.h_lv)
            h_lv_padded = torch.nn.utils.rnn.pad_sequence([self.h_lv, lv], padding_value = 0.0)
            h_lv = h_lv_padded[:,0,:].squeeze()
            
            # h_lv is used to guide lv
            # 3D conv
            h_lv = self.conv(h_lv)
            #print(h_lv.shape, " ", lv.shape)
            # 3D relu
            h_lv = self.relu(h_lv)
            # 3D batchnorm
            h_lv, ls_clone = self.groupnorm(h_lv, ls)
            # 3D Conv
            h_lv = self.conv(h_lv)
            # global avg pooling
            h_lv = h_lv * torch.tensor(1/(h_lv.shape[0]+h_lv.shape[1]))
            # sigmoid
            h_lv = self.sigmoid(h_lv)

            # we have to do a one-padding, because multiplication with zeros would erase values in lv
            index = torch.arange(start=self.h_lv.shape[0], end=lv.shape[0], step=1).to("cuda")
            h_lv=torch.index_fill(h_lv, dim=0, index=index, value=1.0)

            lv = h_lv * lv # elementwise multiplication
            self.h_lv = lv.clone()
            ls.set_values(lv)        
        
        return lv, ls


class TemporalMaxPoolModule(torch.nn.Module):
    def __init__(self ):
        super(TemporalMaxPoolModule, self).__init__()
        self.h_lv = None

    def reset_sequence(self):
        self.h_lv = None

    def forward(self, lv, ls):

        alpha = 0.0
        if (self.h_lv == None):
            #print("First lv: ", lv.shape)
            self.h_lv = lv.clone()
        else:
            h_lv = self.h_lv

            #print("H_lv: ", h_lv.shape, " Lv: ", lv.shape)
            # use -9999, because some values can be smaller than zero
            h_lv_padded = torch.nn.utils.rnn.pad_sequence([h_lv, lv], padding_value = -9999.0) # returns a i x 2 x 64 tensor, where both tensors are padded to the same length
            h_lv = h_lv_padded[:,0,:].squeeze()
            
            lv, _ = torch.max(h_lv_padded, dim = 1)
            self.h_lv = alpha*h_lv + (1-alpha)*lv.clone()

        ls.set_values(lv)
        return lv, ls



class TemporalLinearModule(torch.nn.Module):
    def __init__(self, nr_output_channels ):
        super(TemporalLinearModule, self).__init__()
        #print("Making TemporalLinear with nr_output_channels ", nr_output_channels)
        self.nr_output_channels = nr_output_channels
        self.relu=torch.nn.ReLU(inplace=False)
        self.linear = torch.nn.Linear(self.nr_output_channels*2,self.nr_output_channels)

        self.hidden_linear = torch.nn.Linear(nr_output_channels, nr_output_channels)
        self.h_lv = None

    def reset_sequence(self):
        self.h_lv = None

    def forward(self, lv, ls):
        if (self.h_lv == None): 
            # assert(lv.shape[1] == self.nr_output_channels), "The second dimension of lv and the MLP do not match! Lv is ", lv.shape[1] , "output channels is ", self.nr_output_channels
            if ( lv.shape[1] != self.nr_output_channels ):
                print("The second dimension of lv and the MLP do not match! Lv is ", lv.shape[1] , "output channels is ", self.nr_output_channels)
                exit(1)
            self.h_lv = lv.clone()
            
        elif (self.h_lv != None):
            self.h_lv = self.hidden_linear(self.h_lv)
            alpha = 0.0
            padding_amount = lv.shape[0] - self.h_lv.shape[0]
            h_lv_padded = torch.nn.functional.pad(self.h_lv, (0,0,0,padding_amount), value=0 )
            cat_tensors = torch.cat([h_lv_padded,lv],dim=1)

            cat_tensors = self.linear(cat_tensors)
            cat_tensors = self.relu(cat_tensors)

            lv = alpha*h_lv_padded + (1.0-alpha)*cat_tensors 
            self.h_lv = lv.clone()

        ls.set_values(lv)
        return lv, ls

# SpSequenceNet: https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_SpSequenceNet_Semantic_Segmentation_Network_on_4D_Point_Clouds_CVPR_2020_paper.pdf
class CrossframeLocalInterpolationModule(torch.nn.Module):
    def __init__(self, nr_output_channels, train_alpha_beta = True, use_center = True):
        super(CrossframeLocalInterpolationModule, self).__init__()
        self.h_lv = None
        
        self.nr_output_channels = nr_output_channels
        self.AFLOW = CustomKernelConvLatticeIm2RowModule(nr_filters=nr_output_channels, train_alpha_beta=train_alpha_beta, use_center= use_center)
        self.relu=torch.nn.ReLU(inplace=False)
        self.linear = torch.nn.Linear(self.nr_output_channels*2,self.nr_output_channels)

        self.h_lv_vis, self.weights_vis, self.lattice_neighbors_previous = None, None, None

    def reset_sequence(self):
        self.h_lv = None
        self.h_lv_vis, self.weights_vis, self.lattice_neighbors_previous = None, None, None

    def return_for_vis(self):
        return self.h_lv_vis, self.weights_vis, self.lattice_neighbors_previous

    def forward(self, lv, ls):
        if (self.h_lv == None):
            self.h_lv = lv.clone()
            
        elif (self.h_lv != None):
            alpha = 0.
            padding_amount = lv.shape[0] - self.h_lv.shape[0]
            # we pad with -999999, because the difference between vectors in feature space is calculated and -999999 gives a quite high distance (therefore a small weight)
            h_lv_padded = torch.nn.functional.pad(self.h_lv, (0,0,0,padding_amount), value=-999999 ) 

            #print("Hidden AFLOW ---: ", h_lv_padded)
            AFLOW_feature_vec, weights, lattice_neighbors_previous = self.AFLOW(lv, h_lv_padded, ls)
            self.h_lv_vis, self.weights_vis, self.lattice_neighbors_previous = h_lv_padded.clone().detach(), weights.clone().detach(), lattice_neighbors_previous.clone().detach()

            # concat new feature with feature from lv
            #print(torch.sum(AFLOW_feature_vec), " ", torch.sum(lv))
            cat_tensors = torch.cat([AFLOW_feature_vec,lv],dim=1)

            # residual block to get a feature vector
            cat_tensors = self.linear(cat_tensors)
            cat_tensors = self.relu(cat_tensors)

            lv = alpha*h_lv_padded + (1.-alpha)*cat_tensors 
            #print(torch.sum(lv))
            self.h_lv = lv.clone()
            #print("New h_lv: ", self.h_lv.shape, "\n", self.h_lv[0,:])

        ls.set_values(lv)
        return lv, ls


class CustomKernelConvLatticeIm2RowModule(torch.nn.Module):
    def __init__(self, nr_filters, neighbourhood_size=1, dilation=1, bias=True,  use_center= True, train_alpha_beta=True):
        super(CustomKernelConvLatticeIm2RowModule, self).__init__()
        self.first_time=True
        self.weight=None
        self.bias=None
        self.neighbourhood_size=neighbourhood_size #1-hop neighbor
        self.nr_filters=nr_filters
        self.dilation=dilation
        self.use_bias=bias
        self.use_center=use_center

        if train_alpha_beta == True:
            print("AFLOW: Training alpha and beta values")
            self.alpha = torch.nn.Parameter(data=torch.tensor(0.1), requires_grad=True)
            self.beta = torch.nn.Parameter(data=torch.tensor(0.1), requires_grad=True)
        else:
            print("AFLOW: alpha and beta are set to 0.1 (constant)")
            self.alpha = torch.tensor(0.1)
            self.beta = torch.tensor(0.1)
        #self.alpha = 0.18 #0.4 # values from SpSequenceNet
        #self.beta = 0.5    # values from SpSequenceNet
        self.weights = None
        self.counter = 0

    #as per https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L49
    def reset_parameters(self, filter_extent):
        # torch.nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu') #pytorch uses default leaky relu but we use relu as here https://github.com/szagoruyko/binary-wide-resnet/blob/master/wrn_mcdonnell.py and as in here https://github.com/pytorch/vision/blob/19315e313511fead3597e23075552255d07fcb2a/torchvision/models/resnet.py#L156

        fan = torch.nn.init._calculate_correct_fan(self.weight, "fan_out")
        gain = torch.nn.init.calculate_gain("relu", 1)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        # print("reset params, self use_bias is", self.use_bias)
        if self.bias is not None:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            torch.nn.init.uniform_(self.bias, -bound, bound)


    # hidden state is the already padded h^(t-1)
    def forward(self, lattice_values, hidden_state, lattice_structure):

        lattice_structure.set_values(lattice_values)
        filter_extent=lattice_structure.get_filter_extent(self.neighbourhood_size)
        dilation=1

        if(self.first_time):
            self.first_time=False
            val_dim=lattice_structure.val_dim()
            self.weight = torch.nn.Parameter( torch.empty( filter_extent * val_dim, self.nr_filters ).to("cuda") ) #works for ConvIm2RowLattice
            if self.use_bias:
                self.bias = torch.nn.Parameter( torch.empty( self.nr_filters ).to("cuda") )
            with torch.no_grad():
                self.reset_parameters(filter_extent)


        # 1) get the neighbors in h^(t-1)
        # Lv:  torch.Size([7213, 32]),  lattice_rowified:  torch.Size([7213, 288]), neighborhood of 1 (in 3D this are 2*(3+1) neighbors), nr_filters: 32, 288 = 8*32 + 32 
        lattice_structure.set_values(hidden_state)
        lattice_neighbors_previous=Im2RowLattice.apply(hidden_state, lattice_structure, filter_extent, dilation, self.nr_filters) 
        with torch.no_grad():
            lattice_structure.set_values(hidden_state)
            lattice_neighbors_previous_index=Im2RowIndicesLattice.apply(hidden_state, lattice_structure, filter_extent, dilation, self.nr_filters)     # important: the last index (index center vertex) is wrong here   
            
        # x += 1 -> in-place https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308
        # x = x+1 -> not in-place
        
        AFLOW_lv = torch.zeros_like(lattice_values).to("cuda") 
        distances = torch.zeros((lattice_values.shape[0],9)).to("cuda")
        weights = torch.zeros((lattice_values.shape[0],9)).to("cuda")
        alpha_tensor = torch.ones_like(weights)*self.alpha

        # 2) Calc the distances between the neighbors in h^{t-1} and our center vertex in x^t
        # cdist calcs the rowwise distance between x1 and x2 (each row in x1 gets compared with each row in x2) -> x1.shape: (700,9,self.nr_filters), x2.shape: (700,1,self.nr_filters)
        distances[:,:] = distances[:,:] + torch.cdist(x1 = lattice_neighbors_previous[:,:].reshape(lattice_values.shape[0],-1,self.nr_filters), x2=lattice_values[:,:].unsqueeze(1), p=2.0).squeeze(2)
        #distances[:,:] = distances[:,:] + torch.cdist(x1=lattice_values[:,:].unsqueeze(1), x2 = lattice_neighbors_previous[:,:].reshape(lattice_values.shape[0],-1,self.nr_filters)).squeeze(2)
        distances[:,:] = distances[:,:]* (lattice_neighbors_previous_index[:,::self.nr_filters] != -1)  # we dont want the feature vector of the <not found> neighbors. These are often written down as -1
        if self.use_center == False:
            distances[:,-1] = distances[:,-1]*0. # last element is the center vertex
        distances[:,:] = distances[:,:] * 1/(torch.sum(distances[:,:], dim = 1).unsqueeze(1).repeat_interleave(9,dim=1).detach()) # normalization
        
        # 3) Calc weights from distances
        weights[:,:] = weights[:,:] + (alpha_tensor - torch.min(distances[:,:], alpha_tensor))*self.beta 
        weights[:,:] = weights[:,:] * (lattice_neighbors_previous_index[:,::self.nr_filters] != -1)  # we dont want the feature vector of the <not found> neighbors
        if self.use_center == False:
            weights[:,-1] = weights[:,-1]*0. # last element is the center vertex
        #print(weights)

        # 4) Weight all neighbors with their respective weights
        AFLOW_lv[:,:] = AFLOW_lv[:,:] + torch.sum((lattice_neighbors_previous[:,:].reshape(lattice_values.shape[0],-1,self.nr_filters).permute(0,2,1))*(weights[:,:].unsqueeze(1).repeat_interleave(self.nr_filters,dim=1)), axis = 2)
        
        if self.use_bias:
            AFLOW_lv+=self.bias
        #print("Alpha: ", self.alpha, " Beta: ", self.beta)

        # I do not execute ls.set_values(AFLOW_lv), because I am only interested in the feature vector
        lattice_structure.set_values(lattice_values)
        return AFLOW_lv, weights, lattice_neighbors_previous_index[:,::self.nr_filters]



class PointNetSeqModule(torch.nn.Module):
    def __init__(self, nr_output_channels_per_layer, nr_outputs_last_layer, experiment, rnn_modules, sequence_learning = False, multiplier_hidden_activations=1.0):
        super(PointNetSeqModule, self).__init__()
        self.first_time=True
        self.nr_output_channels_per_layer=nr_output_channels_per_layer
        self.nr_outputs_last_layer=nr_outputs_last_layer
        self.nr_linear_layers=len(self.nr_output_channels_per_layer)
        self.layers=torch.nn.ModuleList([])
        self.norm_layers=torch.nn.ModuleList([])
        self.relu=torch.nn.ReLU(inplace=False)
        self.tanh=torch.nn.Tanh()
        self.leaky=torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.experiment=experiment
        
        # Sequence learning
        self.sequence_learning = sequence_learning
        self.h_lv = None
        self.rnn_modules = rnn_modules
        self.multiplier_hidden_activations=multiplier_hidden_activations
        self.fusion_module = None

        if (self.sequence_learning) and (rnn_modules[0]=="linear" ):
            print("adding Early_Linear fusion with nr_output_channels ", self.nr_outputs_last_layer)
            self.fusion_module = TemporalLinearModule( self.nr_output_channels_per_layer[-1]*2 )
        #self.early_fusion_linear = TemporalLinearModule( self.nr_output_channels_per_layer[-1]*2 )
        if (self.sequence_learning) and (rnn_modules[0]=="cga" ):
            print("adding Early_CGA with nr_output_channels ", self.nr_outputs_last_layer)
            self.fusion_module = CrossframeGlobalAttentionModule(self.nr_output_channels_per_layer[-1]*2)
        #self.CGA = CrossframeGlobalAttentionModule(self.nr_output_channels_per_layer[-1]*2)
        if (self.sequence_learning) and (rnn_modules[0]=="aflow" ):
            print("adding Early_AFLOW with nr_output_channels ", self.nr_outputs_last_layer)
            self.fusion_module = CrossframeLocalInterpolationModule( self.nr_output_channels_per_layer[-1]*2)
        #self.AFLOW = CrossframeLocalInterpolationModule( self.nr_output_channels_per_layer[-1]*2)
        if (self.sequence_learning) and (rnn_modules[0]=="lstm" ):
            print("adding Early_LSTM with nr_output_channels ", self.nr_outputs_last_layer)
            self.fusion_module = LSTMModule(self.nr_output_channels_per_layer[-1]*2)
        #self.LSTM = LSTMModule(self.nr_output_channels_per_layer[-1]*2)
        if (self.sequence_learning) and (rnn_modules[0]=="gru" ):
            print("adding Early_GRU with nr_output_channels ", self.nr_outputs_last_layer)
            self.fusion_module = GRUModule(self.nr_output_channels_per_layer[-1]*2)
        #self.GRU = GRUModule(self.nr_output_channels_per_layer[-1]*2)
        if (self.sequence_learning) and (rnn_modules[0]=="maxpool" ):
            print("adding Early_MaxPool with nr_output_channels ", self.nr_outputs_last_layer)
            self.fusion_module = TemporalMaxPoolModule()
        #self.fusion_maxpool = TemporalMaxPoolModule()
        self.is_early_maxpool_fusion =  ( rnn_modules[0]=="maxpool" and sequence_learning)  

        self.diff_matrix = None

        self.sum = 0.0
        self.counter_sum = 0

        self.nr_iters=0

    def reset_sequence(self):
        self.fusion_module.reset_sequence()
        # self.early_fusion_linear.reset_sequence()
        # self.fusion_maxpool.reset_sequence()
        # self.CGA.reset_sequence()
        # self.LSTM.reset_sequence()
        # self.GRU.reset_sequence()
        # self.AFLOW.reset_sequence()
        self.h_lv = None

    def forward(self, lattice_py, distributed, indices):
        self.nr_iters+=1
        
        if (self.first_time):
            with torch.no_grad():
                self.first_time=False

                #get the nr of channels of the distributed tensor
                nr_input_channels=distributed.shape[1] - 1
                if (self.experiment=="attention_pool"):
                    nr_input_channels=distributed.shape[1] 
                # initial_nr_channels=distributed.shape[1]

                nr_layers=0
                for i in range(len(self.nr_output_channels_per_layer)):
                    nr_output_channels=self.nr_output_channels_per_layer[i]
                    is_last_layer=i==len(self.nr_output_channels_per_layer)-1 #the last layer is folowed by scatter max and not a batch norm therefore it needs a bias
                    self.layers.append( torch.nn.Linear(nr_input_channels, nr_output_channels, bias=True).to("cuda")  )
                    with torch.no_grad():
                        torch.nn.init.kaiming_normal_(self.layers[-1].weight, mode='fan_in', nonlinearity='relu')
                    # self.norm_layers.append( GroupNormLatticeModule(nr_params=nr_output_channels, affine=True)  )  #we disable the affine because it will be slow for semantic kitti
                    nr_input_channels=nr_output_channels
                    nr_layers=nr_layers+1

                if (self.experiment=="attention_pool"):
                    self.pre_conv=torch.nn.Linear(nr_input_channels, nr_input_channels, bias=False).to("cuda") #the last distributed is the result of relu, so we want to start this paralel branch with a conv now
                    self.gamma  = torch.nn.Parameter( torch.ones( nr_input_channels ).to("cuda") ) 
                    with torch.no_grad():
                        torch.nn.init.kaiming_normal_(self.pre_conv.weight, mode='fan_in', nonlinearity='relu')
                    self.att_activ=GnRelu1x1(nr_input_channels, False)
                    self.att_scores=GnRelu1x1(nr_input_channels, True)


                self.last_conv=ConvLatticeModule(nr_filters=self.nr_outputs_last_layer, neighbourhood_size=1, dilation=1, bias=False) #disable the bias becuse it is followed by a gn

                # self.c1=GnReluConv(self.nr_outputs_last_layer, dilation=1, bias=False, with_dropout=False)
                # self.c2=GnReluConv(self.nr_outputs_last_layer, dilation=1, bias=False, with_dropout=False)
                # self.c3=GnReluConv(self.nr_outputs_last_layer, dilation=1, bias=False, with_dropout=False)



        barycentric_weights=distributed[:,-1]
        if ( self.experiment=="attention_pool"):
            distributed=distributed #when we use attention pool we use the distributed tensor that contains the barycentric weights
        else:
            distributed=distributed[:, :distributed.shape[1]-1] #IGNORE the barycentric weights for the moment and lift the coordinates of only the xyz and values

        # #run the distributed through all the layers
        experiment_that_imply_no_elevation=["pointnet_no_elevate", "pointnet_no_elevate_no_local_mean", "splat"]
        if self.experiment in experiment_that_imply_no_elevation:
            # print("not performing elevation by pointnet as the experiment is", self.experiment)
            pass
        else:
            for i in range(len(self.layers)): 

                if (self.experiment=="attention_pool"):
                    distributed=self.layers[i] (distributed)
                    # distributed, lattice_py=self.norm_layers[i] (distributed, lattice_py) 
                    if( i < len(self.layers)-1): 
                        distributed=self.relu(distributed) 
                else:
                    distributed=self.layers[i] (distributed)
                    # if( i < len(self.layers)-1): #last tanh before the maxing need not be applied because it actually hurts the performance, also it's not used in the original pointnet https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
                        #last bn need not be applied because we will max over the lattices either way and then to a bn afterwards
                        # distributed, lattice_py=self.norm_layers[i] (distributed, lattice_py) 
                    if( i < len(self.layers)-1): 
                        distributed=self.relu(distributed) 



        indices_long=indices.long()

        #some indices may be -1 because they were not inserted into the hashmap, this will cause an error for scatter_max so we just set them to 0
        indices_long[indices_long<0]=0
        #print("dist: ", distributed.shape)


        if self.experiment=="splat":
            distributed_reduced = torch_scatter.scatter_mean(distributed, indices_long, dim=0)
        if self.experiment=="attention_pool":
            #attention pooling ##################################################
            #concat for each vertex the max over the simplex
            max_reduced, argmax = torch_scatter.scatter_max(distributed, indices_long, dim=0)
            max_per_vertex=torch.index_select(max_reduced, 0, indices_long)
            distributed_with_max=distributed+self.gamma*max_per_vertex

            pre_conv=self.pre_conv(distributed_with_max) 
            att_activ, lattice_py=self.att_activ(pre_conv, lattice_py)
            att_scores, lattice_py=self.att_scores(att_activ, lattice_py)
            att_scores=torch.exp(att_scores)
            att_scores_sum_reduced = torch_scatter.scatter_add(att_scores, indices_long, dim=0)
            att_scores_sum=torch.index_select(att_scores_sum_reduced, 0, indices_long)
            att_scores=att_scores/att_scores_sum
            #softmax them somehow
            distributed=distributed*att_scores
            distributed_reduced = torch_scatter.scatter_add(distributed, indices_long, dim=0)

            #get also the nr of points in the lattice so the max pooled features can be different if there is 1 point then if there are 100
            ones=torch.cuda.FloatTensor( indices_long.shape[0] ).fill_(1.0)
            nr_points_per_simplex = torch_scatter.scatter_add(ones, indices_long)
            nr_points_per_simplex=nr_points_per_simplex.unsqueeze(1)
            minimum_points_per_simplex=4
            simplexes_with_few_points=nr_points_per_simplex<minimum_points_per_simplex
            distributed_reduced=distributed_reduced.masked_fill(simplexes_with_few_points, 0)
        else:
            distributed_reduced, argmax = torch_scatter.scatter_max(distributed, indices_long, dim=0)
            argmax_clone = argmax.clone() # we have to do this, because scatter_max initialises with max_index+1, which leads to index out of bounds problems, when not all indices have values
            argmax_clone[argmax > argmax.shape[0]] = 0
            # assert torch.max(argmax_clone) < distributed.shape[0], "The max index of argmax is out of bounds!"

            #print(distributed_reduced.shape)
            #get also the nr of points in the lattice so the max pooled features can be different if there is 1 point then if there are 100
            ones=torch.cuda.FloatTensor( indices_long.shape[0] ).fill_(1.0)
            nr_points_per_simplex = torch_scatter.scatter_add(ones, indices_long)
            nr_points_per_simplex=nr_points_per_simplex.unsqueeze(1)
            barycentric_reduced=torch.index_select(barycentric_weights, 0, argmax_clone.flatten()) #we select for each vertex the 64 barycentric weights that got selected by the scatter max
            # barycentric_reduced=torch.index_select(barycentric_weights, 0, argmax_positive ) #we select for each vertex the 64 barycentric weights that got selected by the scatter max
            barycentric_reduced=barycentric_reduced.view(argmax.shape[0], argmax_clone.shape[1])
            distributed_reduced=torch.cat((distributed_reduced,barycentric_reduced),1)

            if not self.is_early_maxpool_fusion: # is we do maxpool  we would like to not have the vertices with low nr of points set to zero because at some other step maybe the features are [-1,-1-2] and then the zero vector will be choosed by the maxpool instead of the actual features with can be negative
                minimum_points_per_simplex=4
                simplexes_with_few_points=nr_points_per_simplex<minimum_points_per_simplex
                distributed_reduced=distributed_reduced.masked_fill(simplexes_with_few_points, 0) #Removed this because further in this module we select the vertices that have no values as the ones that have a zero, and this ones actually have a value but we just set to zero

        lattice_py.set_values(distributed_reduced)
        #print("Reduced: ", distributed_reduced.shape)

        # if(self.rnn_modules[0] == "linear"):
        #     distributed_reduced, lattice_py = self.early_fusion_linear(distributed_reduced, lattice_py)
        # if(self.rnn_modules[0] == "maxpool"):
        #     #the distributed reduced at second timestp has some zero rows which correspond to the rows (vertices) that are not touched by the current cloud, we have to set them to -999 so that the maxpool seelct the one from the previous timestep
        #     feat_size=distributed_reduced.shape[1]
        #     distributed_reduced_features=distributed_reduced[:, 0:int(feat_size/2) ]
        #     distributed_reduced_sumrowwise=distributed_reduced_features.abs().sum(dim=1)
        #     distributed_reduced_sumrowwise=distributed_reduced_sumrowwise.unsqueeze(1)
        #     mask_zeros=distributed_reduced_sumrowwise==0
        #     distributed_reduced=distributed_reduced.masked_fill(mask_zeros, -9900)
        #     distributed_reduced, lattice_py = self.fusion_maxpool(distributed_reduced, lattice_py)
        # if(self.rnn_modules[0] == "lstm"):
        #     distributed_reduced, lattice_py = self.LSTM(distributed_reduced, lattice_py)
        # if(self.rnn_modules[0] == "gru"):
        #     distributed_reduced, lattice_py = self.GRU(distributed_reduced, lattice_py)
        # if(self.rnn_modules[0] == "cga"):
        #     distributed_reduced, lattice_py = self.CGA(distributed_reduced, lattice_py)
        # if(self.rnn_modules[0] == "aflow"):
        #     distributed_reduced, lattice_py = self.AFLOW(distributed_reduced, lattice_py)
        
        if(self.rnn_modules[0] == "maxpool"):
            #the distributed reduced at second timestp has some zero rows which correspond to the rows (vertices) that are not touched by the current cloud, we have to set them to -999 so that the maxpool seelct the one from the previous timestep
            feat_size=distributed_reduced.shape[1]
            distributed_reduced_features=distributed_reduced[:, 0:int(feat_size/2) ]
            distributed_reduced_sumrowwise=distributed_reduced_features.abs().sum(dim=1)
            distributed_reduced_sumrowwise=distributed_reduced_sumrowwise.unsqueeze(1)
            mask_zeros=distributed_reduced_sumrowwise==0
            distributed_reduced=distributed_reduced.masked_fill(mask_zeros, -9900)
            distributed_reduced, lattice_py = self.fusion_module(distributed_reduced, lattice_py)
        elif self.sequence_learning == True:
            distributed_reduced, lattice_py = self.fusion_module(distributed_reduced, lattice_py)



        index = torch.tensor([0]).to("cuda")
        distributed_reduced=torch.index_fill(distributed_reduced, dim=0, index=index, value=0) #the first row corresponds to the invalid points, the ones that had an index of -1. We set it to 0 so it doesnt affect the prediction or the batchnorm
        lattice_py.set_values(distributed_reduced)
        
        distributed_reduced, lattice_py=self.last_conv(distributed_reduced, lattice_py)
        lattice_py.set_values(distributed_reduced)

        return distributed_reduced, lattice_py
