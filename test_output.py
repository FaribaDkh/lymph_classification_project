import imageio
from torch.utils.data import DataLoader
from torchvision import transforms
import PIL
import matplotlib.pyplot as plt
import numpy as np
import sys
from tensorboardX import SummaryWriter
import scipy.ndimage
import scipy.misc
import time
import math
import tables
import random
import torchvision.datasets as datasets
import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
import cv2
import torchvision.models as models
# from unet import UNet #code borrowed from https://github.com/jvanvugt/pytorch-unet
import torch.nn.functional as F
#import pydensecrf.densecrf as dcrf
from PIL import Image
#from v2.UNETnew import R2U_Net, UNetnew
# dataname = "nucleiSmall"
# dataname = "epistroma"
dst_path = "E:\\camelyon16\\TrainingData\\merge_tile\\result_WSI_14_level_2\\"
#mask_path = "masks\\"
test_path = "E:\\camelyon16\\TrainingData\\merge_tile\\WSI_14_level_2\\"
model_path = 'E:/lymph_classification/outputs/gleason_model.pth'
#text_path = "E:\\camelyon16\\results\\result_tumor7722\\result\\text\\"
ignore_index = -100  #Unet has the possibility of masking out pixels in the output image, we can specify the index value here (though not used)
gpuid = 0
# -------------------------------------- unet params -------------------------------------------------------------------
patch_size = 256
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
def is_sorta_tumor(arr, threshold=0.1):
    tot = np.float(np.sum(arr))
    print (tot)
    print(tot/arr.size )
    if tot/arr.size <(threshold):
       # print ("is not black" )
       return False
    else:
       # print ("is kinda black")
       return True

    return dsc

print(torch.cuda.get_device_properties(gpuid))
torch.cuda.set_device(gpuid)
device = torch.device(f'cuda:{gpuid}')
model = models.vgg19(progress=True, num_classes=2)
model = model.to(device)
model.load_state_dict(torch.load('E:/lymph_classification/outputs/gleason_model.pth'))
model.eval()
print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True),
    # transforms.RandomResizedCrop(size=patch_size),
    # transforms.RandomRotation(180),
    # transforms.RandomGrayscale(),
    transforms.ToTensor()
])

img_transform = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True),
    # transforms.RandomResizedCrop(size=patch_size),
    # transforms.RandomRotation(180),
    # transforms.RandomGrayscale(),
    transforms.ToTensor()
])
# load images from folders
dataset = datasets.ImageFolder(test_path, transform=img_transform)
val_size = 0

# optim = torch.optim.Adam(model.parameters())
# nclasses = dataset.numpixels.shape[1]
# class_weight = dataset.numpixels[1, 0:2]
# class_weight = torch.from_numpy(1 - class_weight/class_weight.sum()).type('torch.FloatTensor').to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignore_index, reduce=False)
# criterion = nn.CrossEntropyLoss()
# checkpoint = torch.load(model_path + f"{dataname}_unet_best_model.pth")
# model.load_state_dict(checkpoint["model_dict"])


class LayerActivations():
    features = None
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.cpu()
    def remove(self):
        self.hook.remove()
#w = model.up_path[0].conv_block.block[3]
#plot_kernels(w.weight.detach().cpu(), 8)
# text_file = open(text_path + "result_new.txt", "w")
# val_size = 0
i = 0
# dsc_all = []
# sum_dsc = 0
# f1_sum = 0
for datasetx in dataset:
    [img,x] = datasetx

    val_size = val_size + 1
    plt.clf()
    output = model(img[None, ::].to(device))
    output = output.detach().squeeze().cpu().numpy()
    output = np.moveaxis(output, 0, -1)
    # imageio.imwrite(dst_path +str(i)+ ".png", output[:, :, 1])
    fig, ax = plt.subplots(1, 4, figsize=(10, 4))
    img = np.moveaxis(img.numpy(), 0, -1)
    res =np.argmax(output, axis=2)
    res = np.uint8(res)
    RGB_mask = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    # print(RGB_mask.shape)
    # print(RGB_mask)
    # print(mask.shape)
    # print(img)
    RGB_mask = RGB_mask
    RGB_mask = np.uint8(RGB_mask)
    RGB_mask[np.where((RGB_mask==[1,1,1]).all(axis=2))] = [255,0,0]
    RGB_mask = RGB_mask.astype(float)
    img = img.astype(float)
    alpha = 0.8
    blended = alpha * img + (1 - alpha) * RGB_mask
    # dst = cv2.addWeighted(RGB_mask, 0.4, img, 0.5, 0.0)
    ax[0].imshow(img)
    ax[1].imshow(output[:, :, 1], cmap=plt.cm.gray)
    ax[2].imshow(np.argmax(output, axis=2), cmap=plt.cm.gray)
    # ax[2].imshow(mask, cmap=plt.cm.gray)
    ax[3].imshow(blended)
    ax[0].title.set_text('Original Image')
    ax[1].title.set_text(' predicted output')
    # ax[2].title.set_text('GroundTruth')
    ax[2].title.set_text('Binary Result')
    ax[3].title.set_text('Overlaid output')
    fig.savefig(dst_path +str(i) + ".png")
    # imageio.imwrite(dst_path + "result_" + str(i) + ".png", np.argmax(output, axis=2))
    # fig = plt.imshow(mask, interpolation='nearest')
    # plt.axis('off')
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    # plt.savefig(dst_path + mask_path + "test_mask_" + str(i) + ".png", bbox_inches='tight', pad_inches=0)
    # imageio.imwrite(dst_path + mask_path + "test_mask_" + str(i) + ".png", mask)
    result = np.argmax(output, axis=2)
    # if (is_sorta_tumor(np.argmax(output, axis=2)) & is_sorta_tumor(mask)):
    #     tp, tn, fp, fn,dsc,f1 = get_confusion_matrix_intersection_mats(mask,np.argmax(output, axis=2))
    #     n = text_file.write("results for " + str(i) + "\n")
    #     n = text_file.write("TP: " + str(tp) + "\n")
    #     n = text_file.write("TN : " + str(tn) + "\n")
    #     n = text_file.write("FP : " + str(fp) + "\n")
    #     n = text_file.write("FN : " + str(fn) + "\n")
    #     n = text_file.write("dsc_result : " + str(dsc) + "\n")
    #     n = text_file.write("f1_result   : " + str(f1) + "\n")
    #     n = text_file.write("___________________" + "\n")
    #     dsc_all.append(dsc)
    #     sum_dsc = dsc + sum_dsc
    #     f1_sum = f1 + f1_sum
    i = i+1
# dsc_all_result = sum_dsc/len(dsc_all)
# f1_all_result = f1_sum/len(dsc_all)
# print (dsc_all_result)
# print (f1_all_result)
# n = text_file.write("___________________" + "\n")
# n = text_file.write("___________________" + "\n")
# n = text_file.write("DSC for all images : " + str(dsc_all_result)+ "\n")
# n = text_file.write("f1 for all images : " + str(f1_all_result) +  "\n")
# text_file.close()
#
# train_size = 0
# for datasetx in dataset["train"]:
#     [img, mask, mask_weight] = datasetx
#     # imageio.imwrite("train_" + str(train_size) + ".png", np.moveaxis(img.numpy(), 0, -1))
#
#     plt.clf()
#
#     fig = plt.imshow(mask, interpolation='nearest')
#     plt.axis('off')
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)
#     # plt.savefig("train_mask_" + str(train_size) + ".png", bbox_inches='tight', pad_inches=0)
#
#     train_size = train_size + 1
#
# print("val_size: " + str(val_size) + " train_size: " + str(train_size), )
