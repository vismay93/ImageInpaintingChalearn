# Image Inpainting code for Chalearn Track 1

## Running the code

### step0: Download vgg model weights
We use tensorflow implemention of VGG 16 based on [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg) for perceptual loss. Download the weights of the vgg model from [VGG16 NPY](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) and keep them in the current directory as './vgg16.npy'.

### step1: Setting up Data
Save all the training images in './Data/train' folder in the respective './Data/train/X' and './Data/train/Y' folder and all the test images in './Data/test' folder in './Data/test/X' folder

### step2: Generate masks
Generate mask images from "maskdata.json" file for the given dataset before training the network for image inpainting.

For training images:
```
python genMask.py ./Data/train ./Data/train/M
```

For test images:
```
python genMask.py ./Data/test ./Data/test/M
```

### step3: Trainning 
Train the Network for the image inpainting task using the below mentioned python file, it will save the model files in './model' folder.

```
python train_perceptual_skipconn.py 
```

### step4: Download models
The pretrained models are kept at [modelfile](https://drive.google.com/drive/folders/1WVhEblW0-Dxl2d41OYfy6LIZbkrcigY4?usp=sharing). Download all the files and keep in the './model' folder.

### step5: Testing 
Test the above trained/pretrained network on the test dataset, output will get saved in './Output' folder.

```
python test_perceptual_skipconn.py
```