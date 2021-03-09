import argparse
import json
from os.path import join

import numpy as np
from PIL import Image


# Set label wide: W, height: H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a is a label that translates into a one-dimensional array:(H×W,)；
    #   bIt's a prediction that translates into a one-dimensional array: (H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount calculates the number of occurrences of each number between 0 and n**2-1(total n**2 numbers)，Returned value shape: (n, n)
    #   In returned values, the pixels writen on the diagonal are the correctly classified pixels
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):  
    print('Num classes', num_classes)  

    #-----------------------------------------#
    #   Create a matrix with all zeros, which is an confusion matrix
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    
    #------------------------------------------------#
    #   Get a list of validation set label paths for easy direct reading
    #   Get a list of validation set image segmentation results paths for easy direct reading
    #------------------------------------------------#
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]  

    #------------------------------------------------#
    #   Read every（Image- Label) pair
    #------------------------------------------------#
    for ind in range(len(gt_imgs)): 
        #------------------------------------------------#
        #   Read a image segmentation result，transform to numpy arrays
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))  
        #------------------------------------------------#
        #   Read a corresponding label，transform to numpy arrays
        #------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))  

        #   If the image segmentation result is not the same size as the label, the image will not be counted
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        #------------------------------------------------#
        #   Calculate a 21×21 HIST matrix for an image and sum it
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(),num_classes)  
        # For every 10 images calculated, output the average mIoU value for all classes in the calculated images
        if ind > 0 and ind % 10 == 0:  
            print('{:d} / {:d}: mIou-{:0.2f}; mPA-{:0.2f}'.format(ind, len(gt_imgs),
                                                    100 * np.nanmean(per_class_iu(hist)),
                                                    100 * np.nanmean(per_class_PA(hist))))
    #------------------------------------------------#
    #   Calculate the mIoU values ​​of all validation images set by classes
    #------------------------------------------------#
    mIoUs   = per_class_iu(hist)
    mPA     = per_class_PA(hist)
    #------------------------------------------------#
    #   Output the mIoU value by classes
    #------------------------------------------------#
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tmIou-' + str(round(mIoUs[ind_class] * 100, 2)) + '; mPA-' + str(round(mPA[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    #   Calculate the average mIoU value of all classes on all the validation set images
    #   ignore the NaN value in the calculation
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))  
    return mIoUs


if __name__ == "__main__":
    gt_dir = "VOCdevkit/VOC2007/SegmentationClass"
    pred_dir = "miou_pr_dir"
    png_name_list = open("VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt",'r').read().splitlines() 
    #------------------------------#
    #   Classes +1
    #   2+1
    #------------------------------#
    num_classes = 21
    #--------------------------------------------#
    #   Defined Classes, same as: json_to_dataset
    #--------------------------------------------#
    name_classes = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes)  # Execute the function to calculate mIoU
