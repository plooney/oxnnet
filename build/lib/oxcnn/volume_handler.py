import numpy as np
import sys
import nibabel as nib
from numpy import ndenumerate
import tempfile
from multiprocessing import Pool

class VolumeSegment(object):
    def __init__(self,start_voxel=None,inds=None,inds_rel=None,seg_arr=None):
        self.start_voxel = start_voxel
        self.inds = inds
        self.inds_rel = inds_rel
        self.seg_arr = seg_arr

    def compute_indices(self,size_voxels,np_array_shape):
        ind_min = self.start_voxel 
        ind_max = self.start_voxel + size_voxels -1
        ind_min_abs = np.array([max(i,0) for i in ind_min])
        ind_max_abs = np.array([min(i,j) for i,j in zip(ind_max,np.array(np_array_shape)-1)])
        ind_min_rel = ind_min_abs - ind_min
        ind_max_rel = ind_max_abs - ind_min
        self.inds = np.array([np.linspace(start,stop,stop-start+1,dtype=np.int) for start, stop in zip(ind_min_abs,ind_max_abs)])
        self.inds_rel = np.array([np.linspace(start,stop,stop-start+1,dtype=np.int) for start, stop in zip(ind_min_rel,ind_max_rel)])

    #Pads with zeroes if out of volume bounds
    def read_array(self,np_array,size_voxels):
        #self.start_voxel = start_voxel
        self.compute_indices(size_voxels,np_array.shape)
        mins_inds = np.array([ x.min() for x in  self.inds])
        maxs_inds = np.array([ x.max() for x in  self.inds])+1
        indices = tuple([slice(m1,m2) for m1,m2 in zip(mins_inds,maxs_inds)])
        segment_arr = np_array[indices]
        if not np.array_equal(segment_arr.shape,size_voxels):
            mins = np.array([ x.min() for x in  self.inds_rel])
            maxs = np.array([ x.max() for x in  self.inds_rel]) +1
            pad_tups = list(zip(mins,size_voxels-maxs))
            segment_arr = np.pad(segment_arr,pad_tups,'constant',constant_values=0)
        self.seg_arr = segment_arr

    def __eq__(self, other):
        is_eq = all([np.all(np.array_equal(o,s)) for o,s in  zip(other.inds, self.inds)])
        is_eq &= all([np.all(np.array_equal(o,s)) for o,s in  zip(other.inds_rel, self.inds_rel)])
        is_eq &= all([np.all(np.array_equal(o,s)) for o,s in  zip(other.start_voxel, self.start_voxel)])
        is_eq &= np.array_equal(other.seg_arr, self.seg_arr)
        return is_eq

class ReadArray(object):
    def __init__(self, image_arr, window_shape):
        self.__image_arr = image_arr
        self.__window_shape = window_shape

    def __call__(self, vol):
        vol.read_array(self.__image_arr, self.__window_shape)
        return vol

class ImageHandler(object):
    def image_to_vols(self, image_arr,stride,window_shape, mask_arr=None,crop_by=0, rnd_offset=None):
        if np.any(window_shape <= stride): raise ValueError('Stride is too large for window shape')
        #image_arr =  nib.load(image_file_path).get_data()
        #mask_arr = nib.load(image_mask_path).get_data() if image_mask_path else None
        divisions = (np.array(image_arr.shape)/stride).astype(np.int) + 1
        vol_segs = []
        for index in np.ndindex(*divisions):
            start_voxel = stride*np.array(index) - np.array(window_shape)//2
            if not rnd_offset is None:
                #offset_mag = np.floor(stride/2)-1
                offset = np.array([np.random.randint(w1,w2+1)
                                       for w1,w2 in zip(-rnd_offset,rnd_offset)])
                start_voxel += offset
            add_vol = True
            vol = VolumeSegment(start_voxel)
            if mask_arr is not None:
                vol.compute_indices(window_shape, mask_arr.shape )
                mins_inds = np.array([ x.min() for x in  vol.inds])
                maxs_inds = np.array([ x.max() for x in  vol.inds])+1
                indices = tuple([slice(m1,m2) for m1,m2 in zip(mins_inds,maxs_inds)])
                add_vol = np.any(mask_arr[indices])
                #print(np.mean(mask_arr[indices]))
                #add_vol = np.mean(mask_arr[indices]) > 0.5
            if add_vol:
                vol.read_array(image_arr,window_shape)
                vol_segs.append(vol)
        return vol_segs

    def image_vols_to_vols(self,image_arr,vol_segs_to_match_tuples):
        vol_segs = []
        for vol_seg_to_match in vol_segs_to_match_tuples:
            size_voxels = vol_seg_to_match[0] #.seg_arr.shape
            start_voxel = vol_seg_to_match[1] #.start_voxel
            vol = VolumeSegment(start_voxel)
            vol.read_array(image_arr,size_voxels)
            vol_segs.append(vol)
        return vol_segs

    def max_op_on_overlap(arr_recons,seg_arr,indices,indices_rel): 
        return np.maximum(arr_recons[indices],seg_arr[indices_rel])

    def create_image_from_windows(self,vol_segs,image_shape,op_on_overlap=max_op_on_overlap):
        arr_recons = np.zeros(image_shape) 
        for vol in vol_segs:
            mins_inds = np.array([ x.min() for x in  vol.inds])
            maxs_inds = np.array([ x.max() for x in  vol.inds])+1
            indices = tuple([slice(m1,m2) for m1,m2 in zip(mins_inds,maxs_inds)])
            mins_inds_rel = np.array([ x.min() for x in  vol.inds_rel])
            maxs_inds_rel = np.array([ x.max() for x in  vol.inds_rel])+1
            indices_rel = tuple([slice(m1,m2) for m1,m2 in zip(mins_inds_rel,maxs_inds_rel)])
            arr_recons[indices] = op_on_overlap(arr_recons,vol.seg_arr,indices,indices_rel) if op_on_overlap else vol.seg_arr[indices_rel]
        return arr_recons
