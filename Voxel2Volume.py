import numpy as np
def voxel2volume(nifti_file, mask_array):
    dim = nifti_file.header["pixdim"][1:4]
    per_voxel_volume_cm3 = (dim[0]*dim[1]*dim[2])/1000
    total_volume = round(mask_array.sum()*per_voxel_volume_cm3, 4)
    return total_volume