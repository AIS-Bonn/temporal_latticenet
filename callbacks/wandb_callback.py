from callbacks.callback import *
import wandb
import hjson
import numpy as np

class WandBCallback(Callback):

    def __init__(self, experiment_name, config_path, entity, model):
        self.experiment_name=experiment_name
        # loading the config file like this and giving it to wandb stores them on the website
        with open(config_path, 'r') as j:
            cfg = hjson.loads(j.read())
        # Before this init can be run, you have to use wandb login in the console you are starting the script from (https://docs.wandb.ai/ref/cli/wandb-login, https://docs.wandb.ai/ref/python/init)
        # entity= your username
        wandb.init(project=experiment_name, entity=entity,config = cfg)
        
        # logs all gradients every log_freq step
        wandb.watch(model, log_freq=1000)

        # define our custom x axis metric
        wandb.define_metric("train/step")
        wandb.define_metric("valid/step")
        # set all other train/ metrics to use this step
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("valid/*", step_metric="valid/step")
        

    def after_forward_pass(self, phase, loss, loss_dice, lr,**kwargs):
        # / act as seperators. If you would like to log train and test separately you would log test loss in test/loss 

        log_dict = {
            phase.name+"/loss": loss,
            phase.name+"/loss_dice": loss_dice,
            phase.name+"/lr": lr, 
            phase.name+"/step": phase.iter_nr 
        }
        wandb.log(log_dict)


    def epoch_ended(self, phase, **kwargs):
        phase.scores.update_best()
        mean_iou=phase.scores.avg_class_iou(print_per_class_iou=False)
        best_iou=phase.scores.best_iou if phase.epoch_nr > 0 else 0

        log_dict = {
            phase.name+"/mean_iou": mean_iou,
            phase.name+"/best_iou": best_iou,
            phase.name+"/step": phase.iter_nr 
        }
        wandb.log(log_dict)
