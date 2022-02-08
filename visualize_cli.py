#!/usr/bin/env python3.6

from train_ln import *

import matplotlib.cm

torch.manual_seed(100)

with_viewer = True

config_file="/workspace/schuett_temporal_lattice/seq_config/lnn_train_semantic_kitti.cfg"
myview = None
if with_viewer:
    myview=Viewer.create(config_file) #first because it needs to init context
    recorder=myview.m_recorder
    myview.m_camera.from_string("-18.1639  11.1758  8.70597 -0.208463 -0.533018 -0.137448 0.808414   3.92244  -2.10007 -0.761759 60 0.3 6013.13")

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

# shapes are x ( n x d ) and y ( m x d)
def pairwiseDistance(x,y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, 2).sum(2)
    return dist


def vis_CLI(dataset_name = "semantickitti"):
    global key_pressed
    # DEFINE THE NETWORK AND LOAD ALL THE WEIGHTS 

    if dataset_name == "semantickitti":
        print("######## Using SemanticKitti Dataset ########")
        #config_file="/media/rosu/Data/phd/c_ws/src/schuett_temporal_lattice/lattice_net/config/lnn_train_semantic_kitti radu..cfg"
        config_file="/workspace/schuett_temporal_lattice/seq_config/lnn_train_semantic_kitti.cfg"
        print(config_file)
    elif(dataset_name=="parislille"):
        print("######## Using ParisLille3D Dataset ########")
        config_file="/workspace/schuett_temporal_lattice/seq_config/lnn_train_paris_lille.cfg"
        print(config_file)
    else:
        err="Dataset name not recognized. It is {}. Available options are semantickitti or parislille.".format(dataset_name) 
        sys.exit(err)

    # initialize the parameters used for training
    config_parser = cfgParser(config_file)
    #train_params=TrainParams.create(config_file)    
    model_params=ModelParams.create(config_file)  
    loader_params = config_parser.get_loader_vars()
    label_mngr_params = config_parser.get_label_mngr_vars()
    model_config = config_parser.get_model_vars()
    train_config = config_parser.get_train_vars()
    lattice_gpu_config = config_parser.get_lattice_gpu_vars()
    loader_config = config_parser.get_loader_vars()

    experiment_name="s"
    print("Sigma: ", str(lattice_gpu_config["sigma_0"])[0:3])

    if train_config["with_viewer"]:
        view=Viewer.create(config_file)

    first_time=True

    #torch stuff 
    lattice=Lattice.create(config_file, "lattice")

    cb_list = []
    if(train_config["with_visdom"]):
        cb_list.append(VisdomCallback(experiment_name))
    if(train_config["with_viewer"]):
        cb_list.append(ViewerCallback())
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)

    m_ignore_index = label_mngr_params["unlabeled_idx"]
    labels_file=str(label_mngr_params["labels_file"])
    colorscheme_file=str(label_mngr_params["color_scheme_file"])
    frequency_file=str(label_mngr_params["frequency_file"])
    label_mngr=LabelMngr(labels_file, colorscheme_file, frequency_file, m_ignore_index )
    
    #create loaders
    loader_train, loader_valid,_,dataset_valid = create_loader(train_config["dataset_name"], config_parser, model_config["sequence_learning"], loader_params["shuffle"], valid_border=53) #53 998
    #loader_train, loader_valid,_,dataset_valid = create_loader(train_config["dataset_name"], config_parser, model_config["sequence_learning"], False, train_border=45) #998
    overall_iteration = 0
    
    #create phases
    phases= [
        Phase('test', loader_valid, grad=False)
    ]

    if not torch.cuda.is_available():
        print("The GPU is not available!")
        exit(-1)

    model = None
    #model: we have 20 classes when we use SemanticKitti without moving objects
    if not loader_params["include_moving_classes"] and (train_config["dataset_name"] == "semantickitti"):
        model=LNN_SEQ(20, model_params, config_parser).to("cuda")
    elif (train_config["dataset_name"] == "semantickitti"):
        model=LNN_SEQ(26, model_params, config_parser).to("cuda")
    elif not loader_params["include_moving_classes"] and (train_config["dataset_name"] == "parislille"):
        model=LNN_SEQ(10, model_params, config_parser).to("cuda") # parislille has only 10 classes
    elif (train_config["dataset_name"] == "parislille"):
        model=LNN_SEQ(12, model_params, config_parser).to("cuda") 

    #loss_fn=GeneralizedSoftDiceLoss(ignore_index=loader_train.label_mngr().get_idx_unlabeled() ) 
    loss_fn=LovaszSoftmax(ignore_index=m_ignore_index)
    loss = None
    #class_weights_tensor=model.compute_class_weights(loader_train.label_mngr().class_frequencies(), loader_train.label_mngr().get_idx_unlabeled())
    secondary_fn=torch.nn.NLLLoss(ignore_index=m_ignore_index)  #combination of nll and dice  https://arxiv.org/pdf/1809.10486.pdf

    show_diff = True

    nr_batches_processed=0
    nr_epochs = 0

    while True:

        for phase in phases:
            print("Loader length: ", len(loader_valid))
            #if nr_epochs > 1:
            #    exit()

            if (nr_epochs > train_config["training_epochs"]) and phase.grad:
                return
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)
            model.train(phase.grad)
            torch.cuda.empty_cache()

            #pbar = tqdm(total=len(phase.loader.dataset)) # pbar has to be frame_num the length of phase.loader.dataset, because our sequences are this long
            
            loader_iter = phase.loader.__iter__()

            for batch_idx, (positions_seq, values_seq, target_seq, path_seq,_) in enumerate(loader_iter):
                           
                if positions_seq == None or values_seq == None or target_seq==None:
                    print("Error: Positions, values or target were None!")
                    #pbar.update(1)
                    continue

                for i in range(0,len(positions_seq)):
                    is_training = phase.grad

                    positions = positions_seq[i].squeeze(0).to("cuda") #.detach().clone().to("cuda")
                    values = values_seq[i].squeeze(0).to("cuda") #.detach().clone().to("cuda")
                    target = target_seq[i].squeeze(0).to("cuda") #.detach().clone().to("cuda")
                    
                    early_return = (i != len(positions_seq)-1)
                    #forward
                    with torch.set_grad_enabled(is_training):
                        if i == len(positions_seq)-1:
                            cb.before_forward_pass(lattice=lattice) #sets the appropriate sigma for the lattice
                        
                        pred_logsoftmax, pred_raw, lattice = model(lattice, positions, values, early_return, is_training) # lattice here is ls

                        #if its the first time we do a forward on the model we need to load here the checkpoint
                        if first_time and i==len(positions_seq)-1 and train_config["load_checkpoint"]:
                            # now that all the parameters are created we can fill them with a model from a file
                            model_path = os.path.join(train_config["checkpoint_path"], train_config["load_checkpoint_model"])
                            print("######## Loading state dict from ", model_path, " ########" )
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
                    
                    
                    # reset the hash map after each sequence
                    if (i == len(positions_seq)-1):

                        model.reset_sequence()
                        lattice=Lattice.create(config_file, "lattice")
                
                    #break # we only want to load the values
                break
            break
        break

    # NOW the network has been loaded and is ready for the visualization
    print("The network has been loaded and is ready for the visualization!")
    set_new_cloud = True
    show_arrows = True
    class_id = 20 # 20 = moving car, 1 = car
    secondary_class_id = None#1 
    num_rand_edges = None # 10

    # reset after loading the weights
    model.reset_sequence()
    lattice=Lattice.create(config_file, "lattice")


    loader_iter = phase.loader.__iter__()

    for batch_idx, (positions_seq, values_seq, target_seq, path_seq,_) in enumerate(loader_iter):
                
        while True:
            is_training = False
            
            if myview.m_new_cloud:
                #recorder.record(str(batch_idx)+".png", "./recordings/")
                myview.m_new_cloud = False
                set_new_cloud = True
                break
        
            if(set_new_cloud):
                set_new_cloud = False
                lattice_neighbors_previous_index = None

                Scene.set_floor_visible(False)

                # #print("Positions: ", positions_seq[0].shape, " ", positions_seq[1].shape, " ", positions_seq[2].shape)
                lattice=Lattice.create(config_file, "lattice")

                with torch.set_grad_enabled(is_training):
                    for i in range(0,len(positions_seq)):

                        positions = positions_seq[i].squeeze(0).to("cuda") 
                        values = values_seq[i].squeeze(0).to("cuda") 
                        target = target_seq[i].squeeze(0).to("cuda") 

                        #vis_cli = True #if i==(len(positions_seq)-1) else False
                        pred_logsoftmax, pred_raw, lattice = model(lattice, positions, values, early_return = False, with_gradient = is_training, vis_cli = True) 
                
                lattice_neighbors_previous_index_list, avg_position_per_vertex_list, weight_list = model.visualize_the_cli_module()
                prediction = torch.argmax(pred_logsoftmax, dim = 1).cpu()
                positions = positions_seq[-1].squeeze(0).clone()
                prediction_subsample = prediction.clone().numpy()
                positions_subsample = positions.clone().numpy()
                positions_subsample = positions_subsample[prediction_subsample == 20] 
                prediction_subsample = prediction_subsample[prediction_subsample == 20]
                #prediction_subsample = prediction_subsample[prediction_subsample != 11]
                #print(prediction.shape)

                pclMeshPred = Mesh()
                pclMeshPred.m_label_mngr=label_mngr
                #pclMeshPred.V = positions_seq[-1].squeeze(0).clone()
                pclMeshPred.V = positions_subsample
                #pclMeshPred.L_pred = prediction.numpy()       
                pclMeshPred.L_pred = prediction_subsample   
                pclMeshPred.m_vis.m_point_size=8.0 #4.0 #7.0        
                pclMeshPred.m_vis.m_show_points=True
                pclMeshPred.m_vis.set_color_semanticpred()
                Scene.show(pclMeshPred,"PCL pred (t)")
                
                #print(dir(pclMeshPred.m_vis))

                # pclMesh = Mesh()
                # pclMesh.m_label_mngr=label_mngr
                # pclMesh.V = positions_seq[-1].squeeze(0).clone()
                # pclMesh.L_gt = target_seq[-1].squeeze(0).clone()    
                # # pclMesh.L_gt = dataset_valid.__remapTwelveToTenClasses__(target_seq[-1].squeeze(0).clone())  
                # pclMesh.m_vis.m_point_size=4.0 #7.0        
                # pclMesh.m_vis.m_show_points=False
                # #pclMesh.m_vis.set_color_solid()
                # #pclMesh.m_vis.m_solid_color = [0.25, 0.25, 0.25]
                # #pclMesh.m_vis.m_solid_color = [0.5, 0.5, 0.5]
                # pclMesh.m_vis.set_color_semanticgt()    
                # Scene.show(pclMesh,"PCL gT (t)")

                positionsprev =  positions_seq[-2].squeeze(0).clone().numpy()
                labelprev = target_seq[-2].squeeze(0).clone().numpy()
                positionsprev = positionsprev[labelprev == 20]
                labelprev = labelprev[labelprev==20]

                pclMesh1 = Mesh()
                pclMesh1.m_label_mngr=label_mngr
                pclMesh1.V = positionsprev
                pclMesh1.L_gt = labelprev
                pclMesh1.m_vis.m_point_size=6.0 #7.0        
                pclMesh1.m_vis.m_show_points=True
                #pclMesh1.m_vis.set_color_semanticgt()
                Scene.show(pclMesh1,"PCL gT (t-1)")

                # pclMesh1 = Mesh()
                # pclMesh1.m_label_mngr=label_mngr
                # pclMesh1.V = positions_seq[-2].squeeze(0).clone()
                # pclMesh1.L_gt = target_seq[-2].squeeze(0).clone()    
                # # pclMesh1.L_gt = dataset_valid.__remapTwelveToTenClasses__(target_seq[-2].squeeze(0).clone())   
                # pclMesh1.m_vis.m_point_size=4.0 #7.0        
                # pclMesh1.m_vis.m_show_points=False
                # pclMesh1.m_vis.set_color_solid()
                # #pclMesh1.m_vis.m_solid_color = [0., 0., 0.]
                # #pclMesh1.m_vis.m_solid_color = [0.5, 0.5, 0.5]
                # #pclMesh1.m_vis.set_color_semanticgt()
                # Scene.show(pclMesh1,"PCL gT (t-1)")

                # pclMesh2 = Mesh()
                # pclMesh2.m_label_mngr=label_mngr
                # pclMesh2.V = positions_seq[-3].squeeze(0).clone()
                # pclMesh2.L_gt = target_seq[-3].squeeze(0).clone()    
                # # pclMesh2.L_gt = dataset_valid.__remapTwelveToTenClasses__(target_seq[-3].squeeze(0).clone())   
                # pclMesh2.m_vis.m_point_size=4.0 #7.0        
                # pclMesh2.m_vis.m_show_points=False
                # pclMesh2.m_vis.set_color_solid()
                # #pclMesh2.m_vis.m_solid_color = [0., 0., 0.]
                # #pclMesh2.m_vis.m_solid_color = [0.5, 0.5, 0.5]
                # #pclMesh2.m_vis.set_color_semanticgt()
                # Scene.show(pclMesh2,"PCL gT (t-2)")

                lattice_neighbors_previous_index = lattice_neighbors_previous_index_list[-1]
                weights = weight_list[-1].cpu()

                
                A = avg_position_per_vertex_list[-1].to("cpu")
                #print("Lattice positions shape: ", A.shape)
                #B = avg_position_per_vertex_list[-2].to("cpu")
                #C = avg_position_per_vertex_list[-3].to("cpu")
                
                # lattice vertices that are not hit by the points of the last cloud in the sequence are set to zero. Therefore we need to store the correct position in them. 
                for idx in reversed(range(0,len(avg_position_per_vertex_list))): 
                    if idx == (len(avg_position_per_vertex_list) -1):
                        continue           
                    dist = torch.cdist(A,torch.tensor([0.,0.,0.]).unsqueeze(0))
                    indices_tmp = np.expand_dims(np.arange(A.shape[0]),axis = 1)
                    #indices_not_origin = (indices_tmp[dist > 1e-10])
                    indices_origin = (indices_tmp[dist < 1e-10])
                    A[indices_origin] = (avg_position_per_vertex_list[idx].to("cpu"))[indices_origin]
               

                cmap = matplotlib.cm.get_cmap('Reds')
                # the colors we want to write into the mesh. The endpoint of each edge gives us the color for the line
                colors = np.zeros((A.shape[0],3))

                # save all possible edges and their color
                edges = np.ones((A.shape[0]*8,2), dtype =np.int64)
                color_edges = np.ones((A.shape[0]*8,3), dtype = np.float)
                weights_tmp = np.zeros((A.shape[0]*8), dtype = np.float)
                for i in range(A.shape[0]):
                    edges[i*8:i*8+8,0] = i #int(lattice_neighbors_previous_index[i,-1])
                    edges[i*8:i*8+8,1] = lattice_neighbors_previous_index[i,0:8].to("cpu").numpy()

                    corrected_weights = weights[lattice_neighbors_previous_index[i,-1],0:8]
                    weight_normalized = torch.div(corrected_weights, torch.sum(corrected_weights)+1e-4)
                    color_edges[i*8:i*8+8,:] = cmap(weight_normalized.numpy())[:,0:3]
                    weights_tmp[i*8:i*8+8] = weight_normalized.numpy()
                
                interesting_indices = np.arange((A.shape[0]))
                # pick disp_num_edges many neighborhoods at random
                if num_rand_edges is not None:
                    interesting_indices = np.random.randint(low=0,high=A.shape[0], size = num_rand_edges)

                # get all points with this class_id and their closest lattice vertex
                if (class_id is not None) and not (secondary_class_id is not None):
                    #class_id_positions = positions_seq[-1][target_seq[-1] == class_id].cpu()
                    class_id_positions = positions_seq[-1][prediction.unsqueeze(0) == class_id].cpu()
                    assert class_id_positions.shape[0] > 0, "This class id could not be found in the last cloud of the sequence!"

                    dist = torch.cdist(class_id_positions, A)
                    interesting_indices = torch.unique(torch.argmin(dist, dim = 1)).cpu().numpy()

                if (class_id is not None) and (secondary_class_id is not None):
                    #class_id_positions = positions_seq[-1][target_seq[-1] == secondary_class_id].cpu()
                    class_id_positions = positions_seq[-1][prediction.unsqueeze(0) == secondary_class_id].cpu()
                    assert class_id_positions.shape[0] > 0, "This class id could not be found in the last cloud of the sequence!"

                    dist = torch.cdist(class_id_positions, A)
                    secondary_interesting_indices = torch.unique(torch.argmin(dist, dim = 1)).cpu().numpy()
                    interesting_indices = np.append(interesting_indices, secondary_interesting_indices)

                # show a weighted vector for each lattice vertex 
                if show_arrows:
                    print("Showing the direction!")
                    endpoints = np.zeros((interesting_indices.shape[0],3), dtype = np.float)
                    arrow_edges = np.zeros((interesting_indices.shape[0],2), dtype = np.int)    
                    arrow_edges[:,0] = interesting_indices
                    arrow_edges[:,1] = np.arange(interesting_indices.shape[0])
                    #print(interesting_indices)
                    found_neighbors = np.array((edges[:,1] != -1))

                    for k in range(0,interesting_indices.shape[0]):
                        j = interesting_indices[k]
                        # calculate the vector connecting the start and the endpoint of the edge)
                        endpoint = A[edges[j*8:j*8+8,1],:] - A[edges[j*8,0],:]
                        # eliminate the edges that are not usable
                        endpoint *= np.expand_dims(found_neighbors[j*8:j*8+8],axis = 1).repeat(3,axis = 1)   
                        # normalize the vector to length 1
                        norm = np.linalg.norm(endpoint, axis = 1)
                        endpoint = endpoint * np.expand_dims(norm,axis = 1).repeat(3,axis = 1)
                        # weight them according to the weights
                        endpoint = endpoint * np.expand_dims(weights_tmp[j*8:j*8+8],axis = 1).repeat(3,axis = 1)
                        # max_weight_idx = np.argmax(weights_tmp[j*8:j*8+8])
                        # kill_weights = np.zeros_like(weights_tmp[j*8:j*8+8])
                        # kill_weights[max_weight_idx] = 1
                        # endpoint = endpoint * np.expand_dims(kill_weights,axis = 1).repeat(3,axis = 1)
                        # get the average
                        endpoint = torch.sum(endpoint, dim = 0)
                        # get the correct endpoint
                        endpoints[k] = endpoint + A[edges[j*8,0],:]
                        
                    arrow_edges[:,1] += A.shape[0]
                    #print(arrow_edges)
                    edges = arrow_edges
                    #print(endpoints)

                    # add the endpoints to the points list
                    A = torch.cat((A,torch.tensor(endpoints)))

                else:       
                    # only get the relevant edges                
                    indices_to_keep = np.ones((interesting_indices.shape[0]*8), dtype = np.int)*-2
                    for i in range(interesting_indices.shape[0]):
                        indices_to_keep[i*8:i*8+8] = np.ones((8))*interesting_indices[i]*8 + [0,1,2,3,4,5,6,7] 
                    edges = edges[indices_to_keep,:]
                    color_edges = color_edges[indices_to_keep,:]                          
                    # remove all edges to neighbors that were not found
                    color_edges = np.delete(color_edges,(edges[:,1] == -1), axis = 0)
                    edges = np.delete(edges,(edges[:,1] == -1), axis = 0)
                    
       
                #print(edges[edges[1,:]==np.argmax(counts),:])


                # if eliminate_non_unique:
                #     # now the problem arises that edges[:,1] has a lot of non unique elements 
                #     vals, inverse, count = np.unique(edges[:,1], return_inverse=True, return_counts=True)

                #     idx_vals_repeated = np.where(count > 1)[0]
                #     vals_repeated = vals[idx_vals_repeated]

                #     rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
                #     _, inverse_rows = np.unique(rows, return_index=True)
                #     res = np.split(cols, inverse_rows[1:])

                #     for i in range(0,len(res)):
                #         max_weight = np.amax(weights_tmp[res[i]])
                #         #max_index = np.where(weights_tmp[res[i]] == max_weight)
                #         #colors[edges[:,1],:][res[i][max_index]] = color_edges[res[i][max_index]]
                #         #weight_avg = np.sum(weights_tmp[res[i]]) / weights_tmp[res[i]].shape[0]
                #         colors[edges[res[i],1],:] = cmap(max_weight)[0:3]
                
                                
                latticeMesh = Mesh()
                latticeMesh.m_vis.m_point_size=2.0        
                latticeMesh.m_vis.m_show_points=False
                latticeMesh.V = A
                latticeMesh.E = edges
                #latticeMesh.C = colors
                latticeMesh.m_vis.m_show_lines=True
                latticeMesh.m_vis.m_line_width=10.0
                latticeMesh.m_vis.set_color_pervertcolor()
                #latticeMesh.m_vis.m_solid_color = [0., 0., 0.]
                #latticeMesh.m_vis.m_solid_color = [0.5, 0.5, 0.5]
                #latticeMesh.m_vis.set_color_solid()    

                Scene.show(latticeMesh,"Lattice")


                # latticeMesh1 = Mesh()
                # latticeMesh1.m_vis.m_point_size=2.0        
                # latticeMesh1.m_vis.m_show_points=True
                # latticeMesh1.V = B
                # latticeMesh1.m_vis.m_show_lines=True
                # latticeMesh1.m_vis.m_line_width=5.0
                # latticeMesh1.m_vis.set_color_pervertcolor()

                # Scene.show(latticeMesh1,"Lattice1")

            myview.update()

        model.reset_sequence()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the network on a dataset.')
    parser.add_argument('--dataset', type=str, nargs = "?", const = "semantickitti", 
                    help='the dataset name, options are semantickitti OR parislille')

    args = parser.parse_args()

    if args.dataset:
        vis_CLI(args.dataset)  
    else: # when you do not give any arguments the parser just assumes you want semantickitti
        vis_CLI()