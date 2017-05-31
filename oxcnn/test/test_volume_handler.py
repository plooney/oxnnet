import numpy as np
import sys
import nibabel as nib
import tempfile
from oxcnn.test import utils
from oxcnn.volume_handler import VolumeSegment, ImageHandler

import unittest

class TestMethods(unittest.TestCase):

    def test_image_handler(self):
        _,img_file_path = tempfile.mkstemp(suffix='.nii.gz')
        _,seg_file_path = tempfile.mkstemp(suffix='_seg.nii.gz')
        _,mask_file_path = tempfile.mkstemp(suffix='_mask.nii.gz')
        utils.write_test_cases(img_file_path,mask_file_path,seg_file_path)
        img_handler = ImageHandler()
        stride = np.array([10]*3)
        window_shape = np.array([50]*3)
        image_arr =  nib.load(img_file_path).get_data()
        vol_list = img_handler.image_to_vols(image_arr,stride,window_shape,add_rnd_offset=False)
        mask_arr = nib.load(mask_file_path).get_data()
        img_recons_arr = img_handler.create_image_from_windows(vol_list,image_arr.shape)
        img_diff_arr = mask_arr*(image_arr - img_recons_arr)
        img_nii = nib.Nifti1Image(img_recons_arr, affine=np.eye(4))
        nib.nifti1.save(img_nii,'pred_test.nii.gz')
        self.assertFalse(np.sum(img_diff_arr))
        self.assertFalse(np.any(img_diff_arr))

        tuples = [(vol.seg_arr.shape,vol.start_voxel) for vol in vol_list]
        image_arr =  nib.load(seg_file_path).get_data()
        vol_list_segs = img_handler.image_vols_to_vols(image_arr,tuples)
        seg_recons_arr = img_handler.create_image_from_windows(vol_list_segs,image_arr.shape)
        seg_arr =  nib.load(seg_file_path).get_data()
        self.assertFalse(np.any(mask_arr*(seg_arr-seg_recons_arr)))


        vol1 = vol_list_segs[0]
        vol2 = VolumeSegment(vol1.start_voxel)
        vol2.read_array(seg_arr,vol1.seg_arr.shape)
        self.assertTrue(vol1 == vol2)
        
        tuples = [(vol.seg_arr.shape,vol.start_voxel) for vol in vol_list_segs]
        vol_list_segs2 = img_handler.image_vols_to_vols(image_arr,tuples)
        #for i, v in enumerate(vol_list_segs):
        #    if vol_list_segs2[0] ==v: print(i)
        for v1, v2 in zip(vol_list_segs,vol_list_segs2):
            self.assertTrue(v1 == v2)
