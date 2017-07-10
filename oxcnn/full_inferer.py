import numpy as np
import nibabel as nib
import os
from oxcnn.volume_handler import VolumeSegment, ImageHandler
from oxcnn.data_loader import StandardDataLoader, StandardDataLoaderDistMap, TwoPathwayDataLoader

class StandardFullInferer(object):
    def __init__(self, segment_size_in, segment_size_out, crop_by, stride, batch_size):
        self.segment_size_in = segment_size_in
        self.segment_size_out = segment_size_out
        self.crop_by = crop_by 
        self.stride = stride
        self.batch_size = batch_size

    def evaluate_case(self, sess, model, img_data, batch_size, vs, vsegs, *args,**kwargs):
        tup_seg_size_in = tuple(self.segment_size_in.tolist() + [1])
        full_batch = [v.seg_arr.reshape(tup_seg_size_in) for v in vs]
        num_batches = (len(vs)//batch_size)

        if len(vs) % batch_size: num_batches +=1
        y_out_list = []
        for i in range(0,num_batches):
            batch = full_batch[i*batch_size:min(len(vs),(i+1)*batch_size)]
            y_out = sess.run(model.pred,feed_dict={model.X:batch})
            y_out_list.append(y_out)
            print("Segmenting ", i+1, " of ", num_batches)
        v_out_shape = [len(vs)] + self.segment_size_out.tolist() 
        yr = np.vstack(y_out_list).reshape(v_out_shape)
        yr_labels = np.vstack([v.seg_arr for v in vsegs]).reshape(v_out_shape)
        dice_segments = 2*np.sum(yr*yr_labels)/(np.sum(yr)+np.sum(yr_labels))
        print("Full DICE: ", dice_segments, "No:", len(vsegs) )
        return yr_labels, yr

    def __call__(self, sess, tup, save_dir, model):
        print(tup)
        img = nib.load(tup[0])
        img_data = img.get_data()
        vol_shape = img_data.shape
        print(self.stride)
        data_loader = StandardDataLoader(self.stride, self.segment_size_in)
        vs, vsegs = data_loader.vol_s(tup, crop_by=self.crop_by)
        yr_labels, yr = self.evaluate_case(sess, model, img_data, self.batch_size, vs, vsegs)

        vpreds = [VolumeSegment(start_voxel=vol.start_voxel,seg_arr=seg_arr)
                  for vol,seg_arr in zip(vsegs,yr) if np.all(vol.start_voxel - vol_shape < 0)]
        for v_pred in  vpreds: v_pred.compute_indices(self.segment_size_out,vol_shape)
        img_handler = ImageHandler()
        pre_arr = img_handler.create_image_from_windows(vpreds,vol_shape)

        img_nii = nib.Nifti1Image(pre_arr.astype(np.float32), affine=img.affine)
        out_name = os.path.join(save_dir,'prob_' + os.path.basename(tup[0]).split('.')[0] + '.nii.gz')
        nib.nifti1.save(img_nii,out_name)

        seg_img = nib.load(tup[2]).get_data()
        pre_arr = pre_arr > 0.5

        img_nii = nib.Nifti1Image(pre_arr.astype(np.uint8), affine=img.affine)
        out_name = os.path.join(save_dir,'pred_' + os.path.basename(tup[0]).split('.')[0] + '.nii.gz')
        nib.nifti1.save(img_nii,out_name)

        dice = 2*np.sum(seg_img*pre_arr)/(np.sum(pre_arr)+np.sum(seg_img))
        fpr = np.sum((1-seg_img)*pre_arr)/np.sum(1-seg_img)
        tpr = np.sum(seg_img*pre_arr)/np.sum(seg_img)
        print("TPR:", tpr, " FPR: ", fpr)
        return dice

