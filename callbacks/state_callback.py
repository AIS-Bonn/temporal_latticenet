from callbacks.callback import *
import os
import torch


class StateCallback(Callback):

    def __init__(self):
        pass

    def after_forward_pass(self, phase, loss, pred_softmax, target, cloud, **kwargs):
        phase.iter_nr+=1
        phase.samples_processed_this_epoch+=1
        phase.loss_acum_per_epoch+=loss

        phase.scores.accumulate_scores(pred_softmax, target, cloud.m_label_mngr.get_idx_unlabeled() )

    def epoch_started(self, phase, **kwargs):
        phase.loss_acum_per_epoch=0.0
        phase.scores.start_fresh_eval()

    def epoch_ended(self, phase, model, save_checkpoint, checkpoint_path, name, **kwargs):
        phase.scores.update_best()

        mean_iou=phase.scores.avg_class_iou(print_per_class_iou=False)
        best_iou=phase.scores.best_iou
        best_iou_dict=phase.scores.best_iou_dict
        print("iou for phase_", phase.name , " at epoch ", phase.epoch_nr , " is ", mean_iou, " best mean iou is ", best_iou )

        #save the checkpoint of the model if we are in testing mode
        if not phase.grad:
            if save_checkpoint and model is not None:
                info_txt_name=name+".csv"
                out_info_path=os.path.join(checkpoint_path, info_txt_name)
                phase.scores.write_iou_to_csv(out_info_path)
        

        phase.epoch_nr+=1

    def phase_started(self, phase, **kwargs):
        phase.samples_processed_this_epoch=0

    def phase_ended(self, phase, **kwargs):
        
        if(type(phase.loader) == torch.utils.data.DataLoader): # pytorchs dataloder has no reset 
            pass
        else:
            phase.loader.reset()
