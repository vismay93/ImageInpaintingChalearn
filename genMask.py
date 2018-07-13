import argparse
import numpy as np
import subprocess
#import shutil
import glob
from skimage import img_as_float
import matplotlib.pyplot as plt
import json
import os
import cv2 

parser = argparse.ArgumentParser()
parser.add_argument("DataFolder")
parser.add_argument("MaskFolder")
args = parser.parse_args()

#process =subprocess.Popen("imDir="+args.PreSubmissionFolder+"/ th pose-hg-demo/custom.lua",shell=True)
#process.wait()
#shutil.move("joints.dat",args.SubmissionFolder+"/joints.dat")

maskData = json.load(open(os.path.join(args.DataFolder,"maskdata.json")))
real = sorted(glob.glob(args.DataFolder+"/X/*.png"))
for rfn in real:
    #im = img_as_float(plt.imread(rfn))
    im = cv2.imread(rfn)
    print(im.shape)

    newfn= rfn.split("/")[-1]
    newim = np.zeros(im.shape,dtype=im.dtype)

    for x,y,sz in maskData[newfn]:
        newim[x:x+sz,y:y+sz,:]=255
    cv2.imwrite(args.MaskFolder+"/"+newfn,newim)

