import nibabel as nib
import sys
import os
import numpy as np
import random
from skimage import exposure

def create_img_mask_seg_arrays():
    image_dims = [100,100,100]
    mask_range1 = range(0,20)
    mask_range2 = range(80,100)
    mx1 = random.choice(mask_range1)
    mx2 = random.choice(mask_range2)
    my1 = random.choice(mask_range1)
    my2 = random.choice(mask_range2)
    mz1 = random.choice(mask_range1)
    mz2 = random.choice(mask_range2)
    mask_array = np.zeros(image_dims)
    mask_array[mx1:mx2,my1:my2,mz1:mz2]=1
    x = np.array([50,50])
    y = np.array([50,50])
    z = np.array([50,50])
    while np.any([np.diff(var)[0]<10 for var in [x,y,z]]):
        x = random.sample(range(20,80),2)
        x.sort()
        y = random.sample(range(20,80),2)
        y.sort()
        z = random.sample(range(20,80),2)
        z.sort()
    img_array = np.zeros(image_dims)
    img_array[x[0]:x[1],y[0]:y[1],z[0]:z[1]] = 1
    seg_array = img_array.copy()
    img_array +=  np.random.normal(np.zeros(image_dims),0.3)
    img_array = img_array*mask_array 
    img_array = exposure.rescale_intensity(img_array,out_range=(0,255))
    img_array = img_array.astype(np.float)
    seg_array = np.uint8(seg_array)
    mask_array = np.uint8(mask_array)
    return img_array, mask_array, seg_array

def write_nii(np_array,filename):
    img_nii = nib.Nifti1Image(np_array, affine=np.eye(4))
    nib.nifti1.save(img_nii,filename)

def write_test_cases(img_file_path,mask_file_path,seg_file_path):
    img_arr, mask_arr, seg_arr = create_img_mask_seg_arrays()
    write_nii(img_arr,img_file_path)
    write_nii(mask_arr,mask_file_path)
    write_nii(seg_arr,seg_file_path)

def write_test_case(img_name):
    full_img_name = img_name + '.nii.gz'
    full_mask_name = img_name + '_mask.nii.gz'
    full_seg_name = img_name + '_seg.nii.gz'
    img_arr, mask_arr, seg_arr = create_img_mask_seg_arrays()
    write_nii(img_arr,full_img_name)
    write_nii(mask_arr,full_mask_name)
    write_nii(seg_arr,full_seg_name)


def main():
    main_dir = sys.argv[1]
    for i in range(0,100):
        patient_dir = os.path.join(main_dir,str(i))
        if not os.path.exists(patient_dir): os.makedirs(patient_dir)
        img_name = os.path.join(patient_dir,str(i))
        img_file_path = img_name + '.nii.gz'
        mask_file_path = img_name + '_mask.nii.gz'
        seg_file_path = img_name + '_thresh.nii.gz'
        write_test_cases(img_file_path,mask_file_path,seg_file_path)

if __name__=='__main__':
    main()
