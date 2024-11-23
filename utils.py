
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
import slicerio

SLICE_X = False
SLICE_Y = False
SLICE_Z = True

IMAGE_HEIGHT =128
IMAGE_WIDTH =128

IMG_SIZE =(IMAGE_HEIGHT,IMAGE_WIDTH)
MODEL_NAME = 'UNET_nd_cl_64v2.h5'
model = load_model(MODEL_NAME, compile=False)

def voxel2volume(nifti_file, mask_array):
    dim = nifti_file.header["pixdim"][1:4]
    per_voxel_volume_cm3 = (dim[0]*dim[1]*dim[2])/1000
    total_volume = round(mask_array.sum()*per_voxel_volume_cm3, 4)
    return total_volume

def normalizeImageIntensityRange(img):
    img_range = np.max(img)-np.min(img)
    return (img - np.min(img)) / img_range

def readmask(maskpath):
    segmentation_info = slicerio.read_segmentation(maskpath)
    segment_names = slicerio.segment_names(segmentation_info)
    print(f'Segment Names : {segment_names}')
        
    extracted_voxels = slicerio.extract_segments(
    segmentation_info, [(segment_names[0], 1)])
    return extracted_voxels['voxels']

def scaleImg(img, height, width):
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

def predictVolume(inImg, toBin=True):
    (xMax, yMax, zMax) = inImg.shape
    
    outImgX = np.zeros((xMax, yMax, zMax))
    outImgY = np.zeros((xMax, yMax, zMax))
    outImgZ = np.zeros((xMax, yMax, zMax))
    
    cnt = 0.0
    if SLICE_X:
        cnt += 1.0
        for i in range(xMax):
            img = scaleImg(inImg[i,:,:], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis,:,:,np.newaxis]
            tmp = model.predict(img)[0,:,:,0]
            outImgX[i,:,:] = scaleImg(tmp, yMax, zMax)
    if SLICE_Y:
        cnt += 1.0
        for i in range(yMax):
            img = scaleImg(inImg[:,i,:], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis,:,:,np.newaxis]
            tmp = model.predict(img)[0,:,:,0]
            outImgY[:,i,:] = scaleImg(tmp, xMax, zMax)
    if SLICE_Z:
        cnt += 1.0
        for i in range(zMax):
            img = scaleImg(inImg[:,:,i], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis,:,:,np.newaxis]
            tmp = model.predict(img)[0,:,:,0]
            outImgZ[:,:,i] = scaleImg(tmp, xMax, yMax)
            
    outImg = (outImgX + outImgY + outImgZ)/cnt
    if(toBin):
        outImg[outImg>0.45] = 1.0
        outImg[outImg<=0.45] = 0.0
    return outImg