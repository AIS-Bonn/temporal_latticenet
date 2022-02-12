import hjson

class cfgParser():

    def __init__(self, cfg_file):

        with open(cfg_file, 'r') as j:
            self.contents = hjson.loads(j.read())
           
    # all getter functions return an OrderedDict
    def get_core_vars(self):
        return self.contents['core']
    
    def get_train_vars(self):
        return self.contents['train']

    def get_eval_vars(self):
        return self.contents['eval']

    def get_model_vars(self):
        return self.contents['model']

    def get_lattice_gpu_vars(self):
        return self.contents['lattice_gpu']

    def get_loader_vars(self):
        try:
            if self.contents['train']["dataset_name"] == "semantickitti":
                return self.get_loader_semantic_kitti_vars()
            elif self.contents['train']["dataset_name"] == "parislille":
                return self.get_loader_paris_lille_vars()
            else:
                print("The dataloader you requested is not found: ", self.contents['train']["dataset_name"])
                return None
        except:
            if self.contents['eval']["dataset_name"] == "semantickitti":
                return self.get_loader_semantic_kitti_vars()
            elif self.contents['eval']["dataset_name"] == "parislille":
                return self.get_loader_paris_lille_vars()
            

    def get_loader_semantic_kitti_vars(self):
        return self.contents['loader_semantic_kitti']

    def get_loader_paris_lille_vars(self):
        return self.contents['loader_paris_lille']    

    def get_label_mngr_vars(self):
        try:
            if self.contents['train']["dataset_name"] == "semantickitti":
                return self.contents['loader_semantic_kitti']['label_mngr']
            elif self.contents['train']["dataset_name"] == "parislille":
                return self.contents['loader_paris_lille']['label_mngr'] 
        except:
            if self.contents['eval']["dataset_name"] == "semantickitti":
                return self.contents['loader_semantic_kitti']['label_mngr']
            elif self.contents['eval']["dataset_name"] == "parislille":
                return self.contents['loader_paris_lille']['label_mngr']

    def get_transformer_vars(self):
        try:
            if self.contents['train']["dataset_name"] == "semantickitti":
                return self.contents['loader_semantic_kitti']['transformer']
            elif self.contents['train']["dataset_name"] == "parislille":
                return self.contents['loader_paris_lille']['transformer']
        except:
            if self.contents['eval']["dataset_name"] == "semantickitti":
                return self.contents['loader_semantic_kitti']['transformer']
            elif self.contents['eval']["dataset_name"] == "parislille":
                return self.contents['loader_paris_lille']['transformer']


    def get_visualization_vars(self):
        return self.contents['visualization']
