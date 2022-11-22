import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar

import medicalDataLoader
import argparse
from utils import *

from UNet_Base import *
import random
import torch
import pdb

import warnings
warnings.filterwarnings("ignore")

def runTraining():
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    ## DEFINE HYPERPARAMETERS (batch_size > 1)
    batch_size = 32
    batch_size_val = 1000
    lr = 0.1    # Learning Rate
    epoch = 2 # Number of epochs
    
    root_dir = './Data/'

    print(' Dataset: {} '.format(root_dir))

    ## DEFINE THE TRANSFORMATIONS TO DO AND THE VARIABLES FOR TRAINING AND VALIDATION
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set_full = medicalDataLoader.MedicalImageDataset('train',
                                                      root_dir,
                                                      transform=transform,
                                                      mask_transform=mask_transform,
                                                      augment=False,
                                                      equalize=False)

    train_loader_full = DataLoader(train_set_full,
                              batch_size=batch_size,
                              worker_init_fn=np.random.seed(0),
                              num_workers=0,
                              shuffle=True)


    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            worker_init_fn=np.random.seed(0),
                            num_workers=0,
                            shuffle=False)


    ## INITIALIZE YOUR MODEL
    num_classes = 4 # NUMBER OF CLASSES

    print("~~~~~~~~~~~ Creating the UNet model ~~~~~~~~~~")
    modelName = 'Test_Model'
    print(" Model Name: {}".format(modelName))

    ## CREATION OF YOUR MODEL
    net = UNet(num_classes)

    print("Total params: {0:,}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

    # DEFINE YOUR OUTPUT COMPONENTS (e.g., SOFTMAX, LOSS FUNCTION, ETC)
    softMax = torch.nn.Softmax()
    loss_fn = torch.nn.CrossEntropyLoss()

    ## PUT EVERYTHING IN GPU RESOURCES    
    if torch.cuda.is_available():
        print('gpu')
        net.cuda()
        softMax.cuda()
        loss_fn.cuda()

    ## DEFINE YOUR OPTIMIZER
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    ### To save statistics ####
    lossTotalTraining = []
    Best_loss_val = 1000
    BestEpoch = 0
    
    directory = 'Results/Statistics/' + modelName

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    if os.path.exists(directory)==False:
        os.makedirs(directory)

    ## START THE TRAINING
    
    ## FOR EACH EPOCH
    for i in range(epoch):
        net.train()
        lossEpoch = []
        DSCEpoch = []
        DSCEpoch_w = []
        num_batches = len(train_loader_full)
        
        ## FOR EACH BATCH
        for j, data in enumerate(train_loader_full):

            #plt.imshow(data[0][0,0,:,:]) --> image
            #plt.imshow(data[1][0,0,:,:]) --> label

            ### Set to zero all the gradients
            net.zero_grad()
            optimizer.zero_grad()

            ## GET IMAGES, LABELS and IMG NAMES
            images, labels, img_names = data

            ### From numpy to torch variables
            labels = to_var(labels)
            labels = torch.argmax(labels, dim=1)
            images = to_var(images)

            ################### Train ###################
            #-- The CNN makes its predictions (forward pass)
            #print(images.shape)
            net_predictions = net(images)
            #print(f'prediction : {net_predictions.shape}')
            #print(f'labels : {labels.shape}')

            #-- Compute the losses --#
            # THIS FUNCTION IS TO CONVERT LABELS TO A FORMAT TO BE USED IN THIS CODE
            segmentation_classes = getTargetSegmentation(labels)
            #print(segmentation_classes.shape)
            print('0')
            # COMPUTE THE LOSS
            #loss = DiceLoss(net_predictions, labels)
            #print(net_predictions.shape, labels.shape)
            #print(type(net_predictions[0,1,:,:]))
            #plt.imshow(net_predictions[0,1,:,:].detach().numpy())
            Dice_loss_value = loss_fn(net_predictions, labels) # XXXXXX and YYYYYYY are your inputs for the CE
            print('1')
            lossTotal = Dice_loss_value
            print('2')

            # DO THE STEPS FOR BACKPROP (two things to be done in pytorch)
            Dice_loss_value.backward()
            optimizer.step()

            # THIS IS JUST TO VISUALIZE THE TRAINING 
            lossEpoch.append(lossTotal.cpu().data.numpy())
            printProgressBar(j + 1, num_batches,
                             prefix="[Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Loss: {:.4f}, ".format(lossTotal))

        lossEpoch = np.asarray(lossEpoch)
        lossEpoch = lossEpoch.mean()

        lossTotalTraining.append(lossEpoch)

        printProgressBar(num_batches, num_batches,
                             done="[Training] Epoch: {}, LossG: {:.4f}".format(i,lossEpoch))

        # eval
        net.eval()
        test_loss = 0
        correct = 0
        
        for j, data in enumerate(val_loader):
            images, labels, img_names = data
            labels = to_var(labels)
            labels = torch.argmax(labels, dim=1)
            images = to_var(images)
            output = net(images)
            test_loss += F.nll_loss(output, labels, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(val_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))

        printProgressBar(num_batches, num_batches,
                             done="[Validation] Epoch: {}, LossG: {:.4f}".format(i,lossEpoch))

        ## THIS IS HOW YOU WILL SAVE THE TRAINED MODELS AFTER EACH EPOCH. 
        ## WARNING!!!!! YOU DON'T WANT TO SAVE IT AT EACH EPOCH, BUT ONLY WHEN THE MODEL WORKS BEST ON THE VALIDATION SET!!
        if not os.path.exists('./models/' + modelName):
            os.makedirs('./models/' + modelName)

            torch.save(net.state_dict(), './models/' + modelName + '/' + str(i) + '_Epoch')
            
        np.save(os.path.join(directory, 'Losses.npy'), lossTotalTraining)

    print('done')

runTraining()