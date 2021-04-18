import glob

import cv2
import torch
import numpy as np
import torchvision
from PIL import Image
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import os

from torch.backends import cudnn
import random
# dst_path ="E:\\camelyon16\\TrainingData\\merge_tile\\test\\"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Run this if you want to load the model

model = models.vgg19(progress=True, num_classes=2)
model = model.to(device)
model.load_state_dict(torch.load('E:/lymph_classification/outputs/gleason_model.pth'))

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
        cudnn.benchmark = True

# Show a batch of images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(20, 20))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show(block=True)

# path = "E:\\camelyon16\\TrainingData\\merge_tile\\WSI_14\\"
# test_dir = glob.glob("E:\\camelyon16\\TrainingData\\merge_tile\\WSI_14_level_2\\*.png", recursive=True)
test_dir = 'E:/camelyon16/TrainingData/merge_tile/WSI_2_aperio_level_2'
# test_dir_1 = 'E:/camelyon16/TrainingData/merge_tile/WSI_14_level_2/tumor/'
# test_dir = os.listdir(test_dir_1)

# test_dir = 'E:/camelyon16/level2/2_centers/lymph_classification_dataset/test'
transform_test = transforms.Compose([
                                    transforms.CenterCrop(299),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # convert the image to a Tensor
                                   ])
test_dataset  = datasets.ImageFolder(test_dir, transform=transform_test)
test_load = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
correct = 0
total = 0
confusion_matrix = torch.zeros(2, 2)
model.eval()
with torch.no_grad():

    for i, (inputs, labels) in enumerate(test_load):

        f, e = os.path.splitext((test_dataset.imgs[i])[0])
        # print(f)


        #imshow(torchvision.utils.make_grid(inputs))

        # Convert torch tensor to Variable
        #inputs = Variable(inputs)
        #labels = Variable(labels)

        # CUDA = torch.cuda.is_available()
        # if CUDA:
        #     inputs = inputs.cuda()
        #     labels = labels.cuda()

        # inputs = Variable(inputs)
        inputs = inputs.to(device)
        labels = labels.to(device)

        #print("Path: " + str(path))

        outputs = model(inputs)

        # Record the correct predictions for training data
        _, predicted = torch.max(outputs, 1)
        # print(predicted)
        # print(outputs)
        outputs_numpy = outputs.cpu().numpy()

        # max = outputs_numpy[1]
        output = outputs_numpy.max()
        # print(output)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()
        im = np.zeros((256, 256, 3), np.uint8)
        if (predicted == 1):
            im[:] = (0,0,255)
            cv2.imwrite(str(f) + ".png", im)
        else:
            im[:] = (255,0,0)
            cv2.imwrite(str(f) + ".png", im)
        # if (output > 6):
        #     # im = np.zeros((256, 256,3))
        #     im[:] = (255,69,0)
        #     cv2.imwrite(str(f) + ".png", im)
        #
        # else:
        #     if (output > 3):
        #         # Fill image with red color(set each pixel to red)
        #         im[:] = (0,0,255)
        #         cv2.imwrite(str(f)+ '.png', im)
        #     else:
        #         if (output < 1):
        #             # Fill image with red color(set each pixel to red)
        #                 im[:] = (127,255,0)
        #                 cv2.imwrite(str(f) + '.png', im)
        #         else:
        #                 im[:] = (224,255,255)
        #                 cv2.imwrite(str(f) + '.png', im)

        # if (output < 5 & output > 2.5):
        #     # Fill image with red color(set each pixel to red)
        #     im[:] = (0, 255, 0)
        #     cv2.imwrite(dst_path + str(i) + '.png', im)
        # if (output < 2.5):
        #     # Fill image with red color(set each pixel to red)
        #     im[:] = (0, 255, 255)
        #     cv2.imwrite(dst_path + str(i) + '.png', im)

        # for t, p in zip(labels.view(-1), predicted.view(-1)):
        #     confusion_matrix[t.long(), p.long()] += 1
from sklearn.metrics import cohen_kappa_score
# predicted_numpy = predicted.cpu().numpy()
# labels_numpy = labels.cpu().numpy()
# cohen_kappa_score(labels_numpy, output)
print(confusion_matrix)
print ('train_accuracy = ' + str(100 * correct / len(test_load)))
print(confusion_matrix.diag()/confusion_matrix.sum(1))
#print("wrong_prediction_result: " + str(wrong_prediction_result))
#print("right_prediction_result: " + str(right_prediction_result))
