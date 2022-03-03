import torch

import sys, os
from termcolor import colored

from latticenet_py.lattice.lattice_funcs import *
from latticenet_py.lattice.lattice_modules import *

from functools import reduce
from torch.nn.modules.module import _addindent

from seq_lattice.lattice_modules import *


class LNN_SEQ(torch.nn.Module):
    def __init__(self, nr_classes, model_params, config_parser):
        print("\n-------- Model definition --------")
        super(LNN_SEQ, self).__init__()
        self.nr_classes=nr_classes

        model_config = config_parser.get_model_vars()
        loader_config = config_parser.get_loader_vars()

        self.frames_per_seq=loader_config['frames_per_seq']
        self.multiplier_hidden_activations=1.0/ self.frames_per_seq

        #a bit more control
        self.model_params=model_params
        self.nr_downsamples=model_params.nr_downsamples()
        self.nr_blocks_down_stage=model_params.nr_blocks_down_stage()
        self.nr_blocks_bottleneck=model_params.nr_blocks_bottleneck()
        self.nr_blocks_up_stage=model_params.nr_blocks_up_stage()
        self.nr_levels_down_with_normal_resnet=model_params.nr_levels_down_with_normal_resnet()
        self.nr_levels_up_with_normal_resnet=model_params.nr_levels_up_with_normal_resnet()
        compression_factor=model_params.compression_factor()
        dropout_last_layer=model_params.dropout_last_layer()
        experiment=model_params.experiment()
        #check that the experiment has a valid string
        valid_experiment=["none", "slice_no_deform", "pointnet_no_elevate", "pointnet_no_local_mean", "pointnet_no_elevate_no_local_mean", "splat", "attention_pool"]
        if experiment not in valid_experiment:
            err = "Experiment " + experiment + " is not valid"
            sys.exit(err)


        #####################
        # Sequence Learning #
        #####################        
        self.sequence_learning = model_config["sequence_learning"]   # true if we want to learn sequences
        self.h_lv = None
        self.first_sequence = True # in the first sequence we want to reset the hashmap and save the current state without any modifications as the hidden state
        self.rnn_modules = [x.lower() for x in model_config["rnn_modules"]]
        for i in range(0,len(self.rnn_modules)):
            if self.rnn_modules[i] not in ["linear","maxpool","cga","aflow","lstm","gru"]:
                self.rnn_modules[i] = "none" 
        print("Fusion Modules: ",self.rnn_modules) if not loader_config["accumulate_clouds"] else print("Accumulating all clouds!")
        assert(self.rnn_modules.count("none") < len(self.rnn_modules)), "If sequence_learning = True the rnn_modules can not all be none."


        ######################
        # Dist. and PointNet #
        ######################  
        self.distribute=DistributeLatticeModule(experiment) 
        self.pointnet_layers=model_params.pointnet_layers()
        self.start_nr_filters=model_params.pointnet_start_nr_channels()
        print("pointnet layers is ", self.pointnet_layers)

        self.point_net_seq=PointNetSeqModule( self.pointnet_layers, self.start_nr_filters, experiment, self.rnn_modules, self.sequence_learning,  self.multiplier_hidden_activations) 
         
        
        ######################
        # Recurrent Modules  #
        ###################### 
        
        if( self.sequence_learning ):

            if (self.sequence_learning) and (self.rnn_modules[1] == "linear"):
                print("adding Middle_Linear fusion with nr_output_channels ", model_params.pointnet_start_nr_channels())
            self.middle_fusion_linear = TemporalLinearModule(model_params.pointnet_start_nr_channels() )
            if (self.sequence_learning) and (self.rnn_modules[3] == "linear"):
                print("adding Late_Linear fusion with nr_output_channels ", model_params.pointnet_start_nr_channels()*3)
            self.late_fusion_linear = TemporalLinearModule(model_params.pointnet_start_nr_channels()*3)

            if (self.sequence_learning) and (self.rnn_modules[1] == "maxpool"):
                print("adding Middle_MaxPool fusion")
            self.middle_fusion_maxpool = TemporalMaxPoolModule()
            if (self.sequence_learning) and (self.rnn_modules[3] == "maxpool"):
                print("adding Late_MaxPool fusion")
            self.late_fusion_maxpool = TemporalMaxPoolModule()

            if (self.sequence_learning) and (self.rnn_modules[1] == "cga"):
                print("adding Middle_CGA fusion with nr_output_channels ", model_params.pointnet_start_nr_channels())
            self.middle_CGA = CrossframeGlobalAttentionModule(model_params.pointnet_start_nr_channels())
            if (self.sequence_learning) and (self.rnn_modules[3] == "cga"):
                print("adding Late_CGA fusion with nr_output_channels ", model_params.pointnet_start_nr_channels()*3)
            self.late_CGA = CrossframeGlobalAttentionModule(model_params.pointnet_start_nr_channels()*3)

            if (self.sequence_learning) and (self.rnn_modules[1] == "lstm"):
                print("adding Middle_LSTM fusion with nr_output_channels ", model_params.pointnet_start_nr_channels())
            self.middle_LSTM = LSTMModule(model_params.pointnet_start_nr_channels())
            if (self.sequence_learning) and (self.rnn_modules[3] == "lstm"):
                print("adding Late_LSTM fusion with nr_output_channels ", model_params.pointnet_start_nr_channels()*3)
            self.late_LSTM = LSTMModule(model_params.pointnet_start_nr_channels()*3)

            if (self.sequence_learning) and (self.rnn_modules[1] == "gru"):
                print("adding Middle_GRU fusion with nr_output_channels ", model_params.pointnet_start_nr_channels())
            self.middle_GRU = GRUModule(model_params.pointnet_start_nr_channels())
            if (self.sequence_learning) and (self.rnn_modules[3] == "gru"):
                print("adding Late_GRU fusion with nr_output_channels ", model_params.pointnet_start_nr_channels()*3)
            self.late_GRU = GRUModule(model_params.pointnet_start_nr_channels()*3)

            if (self.sequence_learning) and (self.rnn_modules[1] == "aflow"):
                print("adding Middle_AFLOW Module with nr_output_channels ", model_params.pointnet_start_nr_channels())
            self.middle_AFLOW = CrossframeLocalInterpolationModule(model_params.pointnet_start_nr_channels())
            if (self.sequence_learning) and (self.rnn_modules[2] == "aflow"):
                print("adding Bottleneck_AFLOW Module with nr_output_channels ", model_params.pointnet_start_nr_channels()*4)
            self.AFLOW = CrossframeLocalInterpolationModule(model_params.pointnet_start_nr_channels()*4)
            if (self.sequence_learning) and (self.rnn_modules[3] == "aflow"):
                print("adding LATE_AFLOW Module with nr_output_channels ", model_params.pointnet_start_nr_channels()*3)
            self.late_AFLOW = CrossframeLocalInterpolationModule(model_params.pointnet_start_nr_channels()*3)


        #####################
        # Downsampling path #
        #####################
        self.resnet_blocks_per_down_lvl_list=torch.nn.ModuleList([])
        self.coarsens_list=torch.nn.ModuleList([])
        self.maxpool_list=torch.nn.ModuleList([])
        corsenings_channel_counts = []
        skip_connection_channel_counts = []
        cur_channels_count=self.start_nr_filters
        for i in range(self.nr_downsamples):
            
            #create the resnet blocks
            self.resnet_blocks_per_down_lvl_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_down_stage[i]):
                if i<self.nr_levels_down_with_normal_resnet:
                    should_use_dropout=False
                    print("adding down_resnet_block with nr of filters", cur_channels_count , "and with dropout", should_use_dropout )
                    self.resnet_blocks_per_down_lvl_list[i].append( ResnetBlock(cur_channels_count, [1,1], [False,False], should_use_dropout) )
                else:
                    print("adding down_bottleneck_block with nr of filters", cur_channels_count )
                    self.resnet_blocks_per_down_lvl_list[i].append( BottleneckBlock(cur_channels_count, [False,False,False]) )
            skip_connection_channel_counts.append(cur_channels_count)
            nr_channels_after_coarsening=int(cur_channels_count*2*compression_factor)
            print("adding bnReluCorsen which outputs nr of channels ", nr_channels_after_coarsening )
            self.coarsens_list.append( GnReluCoarsen(nr_channels_after_coarsening)) #is still the best one because it can easily learn the versions of Avg and Blur. and the Max version is the worse for some reason
            cur_channels_count=nr_channels_after_coarsening
            corsenings_channel_counts.append(cur_channels_count)


        #####################
        #     Bottleneck    #
        #####################
        self.resnet_blocks_bottleneck=torch.nn.ModuleList([])
        for j in range(self.nr_blocks_bottleneck):
                print("adding bottleneck_resnet_block with nr of filters", cur_channels_count )
                self.resnet_blocks_bottleneck.append( BottleneckBlock(cur_channels_count, [False,False,False]) )

        self.do_concat_for_vertical_connection=True


        #######################
        #   Upsampling path   #
        #######################
        self.finefy_list=torch.nn.ModuleList([])
        self.up_activation_list=torch.nn.ModuleList([])
        self.up_match_dim_list=torch.nn.ModuleList([])
        self.up_bn_match_dim_list=torch.nn.ModuleList([])
        self.resnet_blocks_per_up_lvl_list=torch.nn.ModuleList([])
        for i in range(self.nr_downsamples):
            nr_chanels_skip_connection=skip_connection_channel_counts.pop()

            # if the finefy is the deepest one int the network then it just divides by 2 the nr of channels because we know it didnt get as input two concatet tensors
            nr_chanels_finefy=int(cur_channels_count/2)

            #do it with finefy
            print("adding bnReluFinefy which outputs nr of channels ", nr_chanels_finefy )
            self.finefy_list.append( GnReluFinefy(nr_chanels_finefy ))

            #after finefy we do a concat with the skip connection so the number of channels doubles
            if self.do_concat_for_vertical_connection:
                cur_channels_count=nr_chanels_skip_connection+nr_chanels_finefy
            else:
                cur_channels_count=nr_chanels_skip_connection

            self.resnet_blocks_per_up_lvl_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_up_stage[i]):
                is_last_conv=j==self.nr_blocks_up_stage[i]-1 and i==self.nr_downsamples-1 #the last conv of the last upsample is followed by a slice and not a bn, therefore we need a bias
                if i>=self.nr_downsamples-self.nr_levels_up_with_normal_resnet:
                    print("adding up_resnet_block with nr of filters", cur_channels_count ) 
                    self.resnet_blocks_per_up_lvl_list[i].append( ResnetBlock(cur_channels_count, [1,1], [False,is_last_conv], False) )
                else:
                    print("adding up_bottleneck_block with nr of filters", cur_channels_count ) 
                    self.resnet_blocks_per_up_lvl_list[i].append( BottleneckBlock(cur_channels_count, [False,False,is_last_conv] ) )

        self.slice_fast_cuda=SliceFastCUDALatticeModule(nr_classes=self.nr_classes, dropout_prob=dropout_last_layer, experiment=experiment)
        self.slice=SliceLatticeModule()
        self.splat=SplatLatticeModule()
        # self.classify=Conv1x1(out_channels=self.nr_classes, bias=True)

        self.start_time = None
       
        self.logsoftmax=torch.nn.LogSoftmax(dim=1)

        # some stuff for visualization
        self.lattice_neighbors_previous_index_list, self.avg_position_per_vertex_list, self.weight_vis_list = [],[],[]

        if experiment!="none":
            warn="USING EXPERIMENT " + experiment
            print(colored("-------------------------------", 'yellow'))
            print(colored(warn, 'yellow'))
            print(colored("-------------------------------", 'yellow'))



    def reset_sequence(self):
        self.h_lv = None
        self.first_sequence = True
        self.start_time = None
        self.lattice_neighbors_previous_index_list, self.avg_position_per_vertex_list, self.weight_vis_list = [],[],[]

        if self.sequence_learning:
            self.point_net_seq.reset_sequence()
            self.middle_fusion_linear.reset_sequence()
            self.late_fusion_linear.reset_sequence()
            self.middle_fusion_maxpool.reset_sequence()
            self.late_fusion_maxpool.reset_sequence()
            self.middle_CGA.reset_sequence()
            self.late_CGA.reset_sequence()
            self.middle_LSTM.reset_sequence()
            self.late_LSTM.reset_sequence()
            self.middle_GRU.reset_sequence()
            self.late_GRU.reset_sequence()
            self.AFLOW.reset_sequence()
            self.middle_AFLOW.reset_sequence()
            self.late_AFLOW.reset_sequence()


    def forward(self, ls, positions, values, early_return = False, with_gradient = True, vis_aflow = False):
        # ls_deepcopy = None
        # we want to clear the hashmap  before we do t=0, because we want to get rid of the previous sequences hashmap
        reset_hashmap = True
        if ((self.sequence_learning == True) and (self.first_sequence == False)):
            reset_hashmap = False
        # if self.first_sequence:
        #     torch.cuda.synchronize()
        #     self.start_time = time.time()
        

        #print(positions.shape)

        with torch.set_grad_enabled(False):
            ls, distributed, indices, weights=self.distribute(ls, positions, values, reset_hashmap)
        #print("Dist: ", distributed.shape)
        #print("Dist ended")

        # TIME_START("pointnet")
        lv, ls=self.point_net_seq(ls, distributed, indices)
        # TIME_END("pointnet")
        #print("LV: ", lv.shape)
        
        if (early_return) and (self.sequence_learning) and (self.rnn_modules[1] == "none") and (self.rnn_modules[2] == "none") and (self.rnn_modules[3] == "none"):
            self.first_sequence = False
            return lv, lv, ls 

        fine_structures_list=[]
        fine_values_list=[]
        # TIME_START("down_path")
        for i in range(self.nr_downsamples):

            #resnet blocks
            for j in range(self.nr_blocks_down_stage[i]):
                # print("start downsample stage ", i , " resnet block ", j, "lv has shape", lv.shape, " ls has val dim", ls.val_dim() )
                lv, ls = self.resnet_blocks_per_down_lvl_list[i][j] ( lv, ls) 

            #saving them for when we do finefy so we can concat them there
            fine_structures_list.append(ls) 
            fine_values_list.append(lv)

            if i == 0:
                # print("middle: ", lv.shape)
                if (self.sequence_learning):
                    if(self.rnn_modules[1] == "linear"):
                        lv, ls = self.middle_fusion_linear(lv,ls)
                    if(self.rnn_modules[1] == "maxpool"):
                        lv, ls = self.middle_fusion_maxpool(lv,ls)
                    if(self.rnn_modules[1] == "lstm"):
                        lv, ls = self.middle_LSTM(lv,ls)
                    if(self.rnn_modules[1] == "gru"):
                        lv, ls = self.middle_GRU(lv,ls)
                    if(self.rnn_modules[1] == "cga"):
                        lv, ls = self.middle_CGA(lv,ls)
                    if(self.rnn_modules[1] == "aflow"):
                        lv, ls = self.middle_AFLOW(lv,ls)

                #print("middle: ", lv.shape)

                if (early_return) and (self.sequence_learning) and (self.rnn_modules[2] == "none") and (self.rnn_modules[3] == "none"):
                    self.first_sequence = False
                    #print("Middle", flush = True)
                    return lv, lv, ls 

            #now we do a downsample
            # print("start coarsen stage ", i, "lv has shape", lv.shape, "ls has val_dim", ls.val_dim() )
            lv, ls = self.coarsens_list[i] ( lv, ls)
            #print(lv.shape)
            # print( "finished coarsen stage ", i, "lv has shape", lv.shape, "ls has val_dim", ls.val_dim() )


        # TIME_END("down_path")

        # #bottleneck
        for j in range(self.nr_blocks_bottleneck):
            # print("bottleneck stage", j,  "lv has shape", lv.shape, "ls has val_dim", ls.val_dim()  )
            lv, ls = self.resnet_blocks_bottleneck[j] ( lv, ls) 
            
        if (self.sequence_learning) and (self.rnn_modules[2] == "aflow"):
            lv,ls = self.AFLOW(lv,ls)
        #print("bottle: ", lv.shape)
        
        # we need to do this, because the ls has to be reset to the correct structure in the first dimension
        with torch.set_grad_enabled(not ((early_return) and (self.sequence_learning) and (self.rnn_modules[3] == "none")) and with_gradient):

            #upsample (we start from the bottom of the U-net, so the upsampling that is closest to the blottlenck)
            # TIME_START("up_path")
            for i in range(self.nr_downsamples):

                fine_values=fine_values_list.pop()
                fine_structure=fine_structures_list.pop()


                #finefy
                # print("start finefy stage", i,  "lv has shape", lv.shape, "ls has val_dim ", ls.val_dim(),  "fine strcture has val dim ", fine_structure.val_dim() )
                lv, ls = self.finefy_list[i] ( lv, ls, fine_structure  )
                #concat or adding for the vertical connection
                if self.do_concat_for_vertical_connection: 
                    lv=torch.cat((lv, fine_values ),1)
                else:
                    lv+=fine_values
                #print(lv.shape)


                if i == (self.nr_downsamples-1):
                    #print("late: ", lv.shape)
                    if (self.sequence_learning):
                        if(self.rnn_modules[3] == "linear"):
                            lv, ls = self.late_fusion_linear(lv,ls)
                        if(self.rnn_modules[3] == "maxpool"):
                            lv, ls = self.late_fusion_maxpool(lv,ls)
                        if(self.rnn_modules[3] == "lstm"):
                            lv, ls = self.late_LSTM(lv,ls)
                        if(self.rnn_modules[3] == "gru"):
                            lv, ls = self.late_GRU(lv,ls)
                        if(self.rnn_modules[3] == "cga"):
                            lv, ls = self.late_CGA(lv,ls)
                        if(self.rnn_modules[3] == "aflow"):
                            lv, ls = self.late_AFLOW(lv,ls)
                    #print("late: ", lv.shape)

                    if (early_return) and (self.sequence_learning):
                        self.first_sequence = False
                        #print(lv.shape, "\n\n", flush = True)
                        return lv, lv, ls 

            

            #resnet blocks
            for j in range(self.nr_blocks_up_stage[i]):
                # print("start resnet block in upstage", i, "lv has shape", lv.shape, "ls has val dim" , ls.val_dim() )
                lv, ls = self.resnet_blocks_per_up_lvl_list[i][j] ( lv, ls) 

        # TIME_END("up_path")


        if vis_aflow:
            #print("hi")
            h_lv_vis, weights_vis, lattice_neighbors_previous_index = self.late_AFLOW.return_for_vis()
            
            if weights_vis is not None:
                lattice_neighbors_previous_index = lattice_neighbors_previous_index
            else:
                weights_vis = torch.zeros((lv.shape[0],1), dtype = torch.long)
                lattice_neighbors_previous_index = torch.zeros((lv.shape[0],1), dtype = torch.long)
           
            pos_scatter = torch.repeat_interleave(positions, 4, dim=0)
            avg_position_per_vertex = torch.zeros((lv.shape[0],3)).to("cuda")
            avg_position_per_vertex =  torch_scatter.scatter_mean(pos_scatter, indices.clone().type(torch.int64), dim = 0, out = avg_position_per_vertex)
            
            self.avg_position_per_vertex_list.append(avg_position_per_vertex.clone())
            self.lattice_neighbors_previous_index_list.append(lattice_neighbors_previous_index.clone())
            self.weight_vis_list.append(weights_vis)

            ls.set_values(lv)
            self.first_sequence = False
            #return lv, lv, ls 

        #print(lv.shape)
        sv =self.slice_fast_cuda(lv, ls, positions, indices, weights)
        # sv =self.slice(lv, ls, positions, indices, weights)
        # sv =self.slice(lv, ls, positions)
        # sv=self.classify(sv)
        #print(sv.shape)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print("diff: ", end_time - self.start_time)

        logsoftmax=self.logsoftmax(sv)
        self.first_sequence = False
        return logsoftmax, sv, ls
        # return logsoftmax, s_final


    def visualize_the_aflow_module(self):
        return self.lattice_neighbors_previous_index_list, self.avg_position_per_vertex_list, self.weight_vis_list

    def prepare_cloud(self, cloud):
       

        with torch.set_grad_enabled(False):

            if self.model_params.positions_mode()=="xyz":
                positions_tensor=torch.from_numpy(cloud.V).float().to("cuda")
            elif self.model_params.positions_mode()=="xyz+rgb":
                xyz_tensor=torch.from_numpy(cloud.V).float().to("cuda")
                rgb_tensor=torch.from_numpy(cloud.C).float().to("cuda")
                positions_tensor=torch.cat((xyz_tensor,rgb_tensor),1)
            elif self.model_params.positions_mode()=="xyz+intensity":
                xyz_tensor=torch.from_numpy(cloud.V).float().to("cuda")
                intensity_tensor=torch.from_numpy(cloud.I).float().to("cuda")
                positions_tensor=torch.cat((xyz_tensor,intensity_tensor),1)
            else:
                err="positions mode of ", self.model_params.positions_mode() , " not implemented"
                sys.exit(err)


            if self.model_params.values_mode()=="none":
                values_tensor=torch.zeros(positions_tensor.shape[0], 1) #not really necessary but at the moment I have no way of passing an empty value array
            elif self.model_params.values_mode()=="intensity":
                values_tensor=torch.from_numpy(cloud.I).float().to("cuda")
            elif self.model_params.values_mode()=="rgb":
                values_tensor=torch.from_numpy(cloud.C).float().to("cuda")
            elif self.model_params.values_mode()=="rgb+height":
                rgb_tensor=torch.from_numpy(cloud.C).float().to("cuda")
                height_tensor=torch.from_numpy(cloud.V[:,1]).unsqueeze(1).float().to("cuda")
                values_tensor=torch.cat((rgb_tensor,height_tensor),1)
            elif self.model_params.values_mode()=="rgb+xyz":
                rgb_tensor=torch.from_numpy(cloud.C).float().to("cuda")
                xyz_tensor=torch.from_numpy(cloud.V).float().to("cuda")
                values_tensor=torch.cat((rgb_tensor,xyz_tensor),1)
            elif self.model_params.values_mode()=="height":
                height_tensor=torch.from_numpy(cloud.V[:,1]).unsqueeze(1).float().to("cuda")
                values_tensor=height_tensor
            elif self.model_params.values_mode()=="xyz":
                xyz_tensor=torch.from_numpy(cloud.V).float().to("cuda")
                values_tensor=xyz_tensor
            else:
                err="values mode of ", self.model_params.values_mode() , " not implemented"
                sys.exit(err)


            target=cloud.L_gt
            target_tensor=torch.from_numpy(target).long().squeeze(1).to("cuda").squeeze(0)

        return positions_tensor, values_tensor, target_tensor

    #like in here https://github.com/drethage/fully-convolutional-point-network/blob/60b36e76c3f0cc0512216e9a54ef869dbc8067ac/data.py 
    #also the Enet paper seems to have a similar weighting
    def compute_class_weights(self, class_frequencies, background_idx):
        """ Computes class weights based on the inverse logarithm of a normalized frequency of class occurences.
        Args:
        class_counts: np.array
        Returns: list[float]
        """

        #doing it my way but inspired by their approach of using the logarithm
        class_frequencies_tensor=torch.from_numpy(class_frequencies).float().to("cuda")
        class_weights = (1.0 / torch.log(1.05 + class_frequencies_tensor)) #the 1.2 says pretty much what is the maximum weight that we will assign to the least frequent class. Try plotting the 1/log(x) and you will see that I mean. The lower the value, the more weight we give to the least frequent classes. But don't go below the value of 1.0
        #1 / log(1.01+0.000001) = 100
        class_weights[background_idx]=0.00000001

        return class_weights

#https://github.com/pytorch/pytorch/issues/2001
def summary(self,file=sys.stderr):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if p is not None:
                total_params += reduce(lambda x, y: x * y, p.shape)
                # if(p.grad==None):
                #     print("p has no grad", name)
                # else:
                #     print("p has gradnorm ", name ,p.grad.norm() )

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stderr:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
            for name, p in model._parameters.items():
                if hasattr(p, 'grad'):
                    if(p.grad==None):
                        # print("p has no grad", name)
                        main_str+="p no grad"
                    else:
                        # print("p has gradnorm ", name ,p.grad.norm() )
                        main_str+= "\n" + name + " p has grad norm " + str(p.grad.norm())
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(self)
    if file is not None:
        print(string, file=file)
    return count

