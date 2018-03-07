"""Module containing tests for loading the imagedata into segments and
creating an image from segments. The segments are the input for the neural network.
"""
import os
import tempfile
import unittest
import numpy as np
import nibabel as nib
from skimage.measure import block_reduce
from oxnnet.volume_handler import VolumeSegment, ImageHandler
from oxnnet.data_loader import TwoPathwayDataLoader
from tests import utils

class TestDataLoading(unittest.TestCase):
    """Class for testing the loading of image data into segments and visa versa"""

    def setUp(self):
        _, self.img_file_path = tempfile.mkstemp(suffix='.nii.gz')
        _, self.seg_file_path = tempfile.mkstemp(suffix='_seg.nii.gz')
        _, self.mask_file_path = tempfile.mkstemp(suffix='_mask.nii.gz')
        utils.write_test_cases(self.img_file_path, self.mask_file_path, self.seg_file_path)
        self.image_arr = nib.load(self.img_file_path).get_data()
        self.mask_arr = nib.load(self.mask_file_path).get_data()
        self.seg_arr = nib.load(self.seg_file_path).get_data()

    def tearDown(self):
        for fname in [self.img_file_path, self.seg_file_path, self.mask_file_path]:
            try:
                os.remove(fname)
            except OSError:
                pass

    def test_two_pathway(self):
        """Test the TwoPathway dataloader and see if reconstituting the volume
        segments gives the same as a block reduction of the imagei"""
        tup = (self.img_file_path, self.mask_file_path, self.seg_file_path)
        img_handler = ImageHandler()

        stride = np.array([9, 9, 9])
        segment_size = np.array([25, 25, 25])
        segment_size_ss = np.array([19, 19, 19])
        dl = TwoPathwayDataLoader(stride, segment_size, segment_size_ss)
        _, _, vl_ss = dl.vol_s(tup)
        img_recons_arr = img_handler.create_image_from_windows(vl_ss,
                                                               np.array(self.image_arr.shape)//3+1)

        #img_nii = nib.Nifti1Image(img_recons_arr, affine=np.eye(4))
        #nib.nifti1.save(img_nii,'blockreduced_recons_test.nii.gz')

        img_reduced_arr = block_reduce(self.image_arr, block_size=(3, 3, 3),
                                       func=np.median).astype(np.uint8)
        #img_nii = nib.Nifti1Image(img_reduced_arr, affine=np.eye(4))
        #nib.nifti1.save(img_nii,'block_reduced_test.nii.gz')
        img_diff_arr = (img_reduced_arr - img_recons_arr)
        self.assertFalse(np.any(img_diff_arr))

    def test_image_handler(self):
        """Check decomposing image to VoluemSegments and back again produces the same image"""
        img_handler = ImageHandler()
        stride = np.array([10]*3)
        window_shape = np.array([50]*3)
        vol_list = img_handler.image_to_vols(self.image_arr, stride, window_shape)
        img_recons_arr = img_handler.create_image_from_windows(vol_list, self.image_arr.shape)
        img_diff_arr = self.mask_arr*(self.image_arr - img_recons_arr)
        #img_nii = nib.Nifti1Image(img_recons_arr, affine=np.eye(4))
        #nib.nifti1.save(img_nii,'pred_test.nii.gz') #For testing
        self.assertFalse(np.any(img_diff_arr))

    def test_image_handler_seg_vs(self):
        """Check decomposing segmentation and reconstituting produces the same image"""
        stride = np.array([10]*3)
        window_shape = np.array([50]*3)
        img_handler = ImageHandler()
        vol_list = img_handler.image_to_vols(self.image_arr, stride, window_shape)
        tuples = [(vol.seg_arr.shape, vol.start_voxel) for vol in vol_list]
        vol_list_segs = img_handler.image_vols_to_vols(self.seg_arr, tuples)
        seg_recons_arr = img_handler.create_image_from_windows(vol_list_segs, self.image_arr.shape)
        self.assertFalse(np.any(self.mask_arr*(self.seg_arr-seg_recons_arr)))

        vol1 = vol_list_segs[0]
        vol2 = VolumeSegment(vol1.start_voxel)
        vol2.read_array(self.seg_arr, vol1.seg_arr.shape)
        self.assertTrue(vol1 == vol2)

        tuples = [(vol.seg_arr.shape, vol.start_voxel) for vol in vol_list_segs]
        vol_list_segs2 = img_handler.image_vols_to_vols(self.seg_arr, tuples)
        for v1, v2 in zip(vol_list_segs, vol_list_segs2):
            self.assertTrue(v1 == v2)

if __name__ == '__main__':
    unittest.main()
