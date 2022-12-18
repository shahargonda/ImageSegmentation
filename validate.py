# USAGE
# python validate.py
# import the necessary packages
from pyimagesearch import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from sklearn.metrics import jaccard_score
from imutils import paths

def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	# set the layout of the figure and display it
	figure.tight_layout()
	figure.show()
	plt.pause(1)

def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (128, 128))
		orig = image.copy()
		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]
		filename= os.path.splitext(filename)[0]+'_segmentation.png' # filename:'ISIC_0000217_segmentation.png'
		groundTruthPath = os.path.join(config.MASK_DATASET_PATH, filename)
		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		gtMask = cv2.imread(groundTruthPath, 0)
		print('gtMask shape', gtMask.shape)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_HEIGHT))

		# make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)
		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		# filter out the weak predictions and convert them to integers
		predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)
		# prepare a plot for visualization
		prepare_plot(orig, gtMask, predMask)

if __name__ == '__main__':
    # load our model from disk and flash it to the current device
    #print("[INFO] load up model...")
    #unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
	# validation image paths
    print("[INFO] prediction on all validation images")

    validation_images = sorted(list(paths.list_images(config.VALIDATION_DATASET_IMAGES_PATH)))
    validation_masks = sorted(list(paths.list_images(config.VALIDATION_DATASET_MASKS_PATH)))
	# iterate over the randomly selected test image paths

    # define transformations
    transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((config.INPUT_IMAGE_HEIGHT,
                                                        config.INPUT_IMAGE_WIDTH)),
                                     transforms.ToTensor()])
    # create the train and test datasets
    validationDS = SegmentationDataset(imagePaths=validation_images, maskPaths=validation_masks,
                                  transforms=transforms)
    validationLoader = DataLoader(validationDS, shuffle=True,
                             batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                             num_workers= os.cpu_count())
    with torch.no_grad():
        # set the model in evaluation mode
        unet.eval()
        # loop over the validation set
        for (x, y) in validationLoader:
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            # make the predictions and calculate the validation loss
            pred = unet(x)
            totalTestLoss += lossFunc(pred, y)

# USAGE
# python train.py
# import the necessary packages
from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

if __name__ == '!__main__':

    # load the image and mask filepaths in a sorted manner
    imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
    maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))
    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(imagePaths, maskPaths,
                             test_size=config.TEST_SPLIT, random_state=42)
    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]
    # write the testing image paths to disk so that we can use then
    # when evaluating/testing our model
    print("[INFO] saving testing image paths...")
    f = open(config.TEST_PATHS, "w")
    f.write("\n".join(testImages))
    f.close()

    # define transformations
    transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((config.INPUT_IMAGE_HEIGHT,
                                                        config.INPUT_IMAGE_WIDTH)),
                                     transforms.ToTensor()])
    # create the train and test datasets
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
                                  transforms=transforms)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
                                 transforms=transforms)
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")
    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True,
                             batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                             num_workers= os.cpu_count())
    testLoader = DataLoader(testDS, shuffle=False,
                            batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                            num_workers= os.cpu_count())

    # initialize our UNet model
    unet = UNet().to(config.DEVICE)
    # initialize loss function and optimizer
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=config.INIT_LR)
    # calculate steps per epoch for training and test set
    trainSteps = len(trainDS) // config.BATCH_SIZE
    testSteps = len(testDS) // config.BATCH_SIZE
    # initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": []}

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(config.NUM_EPOCHS)):
        # set the model in training mode
        unet.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        # loop over the training set
        batch_counter=0
        for (i, (x, y)) in enumerate(trainLoader):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            # perform a forward pass and calculate the training loss
            pred = unet(x)
            loss = lossFunc(pred, y)
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
            if (batch_counter+1)%5 == 0:
                print(f"\n finished processing {batch_counter + 1} batches")
            batch_counter += 1
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()
            # loop over the validation set
            for (x, y) in testLoader:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # make the predictions and calculate the validation loss
                pred = unet(x)
                totalTestLoss += lossFunc(pred, y)
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH)
    # serialize the model to disk
    torch.save(unet, config.MODEL_PATH)




if False:
    val_loader = zip(validation_imagePaths, validation_masksPaths)
    print(list(val_loader))

    y_pred_true_pairs = []
    for images, masks in val_loader:
        images = Variable(images.cuda())
        y_preds = model(images)
        for i, _ in enumerate(images):
            y_pred = y_preds[i]
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().data.numpy()
            y_pred_true_pairs.append((y_pred, masks[i].numpy()))

    # We use a method to calculate the IOU score as found in this kernel here: https://www.kaggle.com/leighplt/goto-pytorch-fix-for-v0-3.

    # https://www.kaggle.com/leighplt/goto-pytorch-fix-for-v0-3
    for threshold in np.linspace(0, 1, 11):
        ious = []
        for y_pred, mask in y_pred_true_pairs:
            prediction = (y_pred > threshold).astype(int)
            iou = jaccard_score(mask.flatten(), prediction.flatten())
            ious.append(iou)

        accuracies = [np.mean(ious > iou_threshold)
                      for iou_threshold in np.linspace(0.5, 0.95, 10)]
        print('Threshold: %.1f, Metric: %.3f' % (threshold, np.mean(accuracies)))
