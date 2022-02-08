
import torchnet
import numpy as np
import torch
import csv

            
class Scores():
    def __init__(self):
        self.clear()

    #adapted from https://github.com/NVlabs/splatnet/blob/f7e8ca1eb16f6e1d528934c3df660bfaaf2d7f3b/splatnet/semseg3d/eval_seg.py
    def accumulate_scores(self, pred_softmax, gt, unlabeled_idx):
        self.nr_classes=pred_softmax.shape[1]
        pred=pred_softmax.argmax(1)
        gt=gt.detach()
        pred=pred.detach()

        self.labels = torch.unique(gt) # vector containing an index of the classes that are in the cloud. The order of them is not defined
        
        if( self.intersection_per_class==None):
            self.intersection_per_class = [0] * self.nr_classes
            self.union_per_class = [0] * self.nr_classes

        for i in range(len(self.labels)):
            l=self.labels[i]
            if not l==unlabeled_idx:
                current_intersection=((pred==gt)*(gt==l)).sum().item()
                self.intersection_per_class[l]+= current_intersection
                self.union_per_class[l]+=  (gt==l).sum().item() + (pred==l).sum().item()  -  current_intersection

    def compute_stats(self, print_per_class_iou=False):
        valid_classes=0
        iou_sum=0
        iou_dict={}
        for i in range(self.nr_classes):
            if( self.union_per_class[i]>0 ):
                valid_classes+=1
                iou=self.intersection_per_class[i] / self.union_per_class[i]
                iou_sum+=iou
                if print_per_class_iou:
                    print("class iou for idx", i, " is ", iou )
                iou_dict[i]=iou
        avg_iou=iou_sum/valid_classes
        return avg_iou, iou_dict

    def avg_class_iou(self, print_per_class_iou=False):
        avg_iou, iou_dict= self.compute_stats(print_per_class_iou) 
        return avg_iou

    def iou_per_class(self, print_per_class_iou=False):
        avg_iou, iou_dict= self.compute_stats(print_per_class_iou) 
        return iou_dict

    def update_best(self):
        avg_iou, iou_dict= self.compute_stats(print_per_class_iou=False) 
        if avg_iou>self.best_iou:
            self.best_iou=avg_iou
            self.best_iou_dict=iou_dict


    def show(self, epoch_nr):
        avg_iou=self.avg_class_iou(print_per_class_iou=True)
    def clear(self):
        self.intersection_per_class=None
        self.union_per_class=None


        self.labels = None
        self.nr_classes =None

        #storing the best iou we got
        self.best_iou=-99999999
        self.best_iou_dict={}

    def start_fresh_eval(self):
        self.intersection_per_class=None
        self.union_per_class=None
        self.labels = None
        self.nr_classes =None

    def write_iou_to_csv(self,filename):
        iou_dict=self.iou_per_class(print_per_class_iou=False)
        avg_iou= self.avg_class_iou(print_per_class_iou=False)
        w = csv.writer(open(filename, "w"))
        for key, val in iou_dict.items():
            w.writerow([key, val])
        w.writerow(["mean_iou", avg_iou])

    def write_best_iou_to_csv(self,filename):
        iou_dict=self.best_iou_dict
        best_iou=self.best_iou
        w = csv.writer(open(filename, "w"))
        for key, val in iou_dict.items():
            w.writerow([key, val])
        w.writerow(["best_iou", best_iou])

