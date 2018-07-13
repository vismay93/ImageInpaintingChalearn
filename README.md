# Image Inpainting code for Chalearn Track 1

## Running the code

### step0: Setting up Data
Save all the training images in './Data/train' folder in the respective './Data/train/X' and './Data/train/Y' folder and all the test images in './Data/test' folder in './Data/test/X' folder

### step1: Generate masks
Generate mask images from "maskdata.json" file for the given dataset before training the network for image inpainting.

For training images:
```
python genMask.py ./Data/train ./Data/train/M
```

For test images:
```
python genMask.py ./Data/test ./Data/test/M
```

### step2: Trainning 
Train the Network for the image inpainting task using the below mentioned python file, it will save the model files in './model' folder

```
python train_perceptual_skipconn.py 
```

### step3: Testing 
Test the above trained/pretrained network on the test dataset, output will get saved in './Output' folder.

```
python test_perceptual_skipconn.py
```