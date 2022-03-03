from callbacks.callback import *
import wandb
import hjson

class WandBCallback(Callback):

    def __init__(self, experiment_name, config_path, entity):
        self.experiment_name=experiment_name
        # loading the config file like this and giving it to wandb stores them on the website
        with open(config_path, 'r') as j:
            cfg = hjson.loads(j.read())
        # Before this init can be run, you have to use wandb login in the console you are starting the script from (https://docs.wandb.ai/ref/cli/wandb-login, https://docs.wandb.ai/ref/python/init)
        # entity= your username
        wandb.init(project=experiment_name, entity=entity,config = cfg)
        

    def after_forward_pass(self, phase, loss, loss_dice, lr,**kwargs):
        # / act as seperators. If you would like to log train and test separately you would log test loss in test/loss 
        wandb.log({'{}/loss'.format(phase.name): loss}, step=phase.iter_nr)
        wandb.log({'{}/loss_dice'.format(phase.name): lr}, step=phase.iter_nr)
        wandb.log({'{}/lr'.format(phase.name): lr}, step=phase.iter_nr)


    def epoch_ended(self, phase, **kwargs):
        mean_iou=phase.scores.avg_class_iou(print_per_class_iou=False)
        best_iou=phase.scores.best_iou
        
        wandb.log({'{}/mean_iou'.format(phase.name): mean_iou}, step=phase.epoch_nr)
        wandb.log({'{}/best_iou'.format(phase.name): best_iou}, step=phase.epoch_nr)
