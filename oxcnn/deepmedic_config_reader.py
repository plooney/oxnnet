import os

class DeepMedicConfigReader(object):
    def __init__(self,project_directory):
        self.train_config_filepath = os.path.join(project_directory,'train','trainConfig.cfg')
        self.train_channels_filepath = os.path.join(project_directory,'train','trainChannels.cfg')
        self.train_gtlabels_filepath = os.path.join(project_directory,'train','trainGtLabels.cfg')
        self.train_roimasks_filepath = os.path.join(project_directory,'train','trainRoiMasks.cfg')
        self.validation_channels_filepath = os.path.join(project_directory,'train','validation','validationChannels.cfg')
        self.validation_roimasks_filepath = os.path.join(project_directory,'train','validation','validationRoiMasks.cfg')
        self.validation_gtlabels_filepath = os.path.join(project_directory,'train','validation','validationGtLabels.cfg')
        self.validation_predictions_filepath = os.path.join(project_directory,'train','validation','validationNamesOfPredictions.cfg')
        self.test_channels_filepath = os.path.join(project_directory,'test','testChannels.cfg')
        self.test_roimasks_filepath = os.path.join(project_directory,'test','testRoiMasks.cfg')
        self.test_gtlabels_filepath = os.path.join(project_directory,'test','testGtLabels.cfg')
        self.test_predictions_filepath = os.path.join(project_directory,'test','testNamesOfPredictions.cfg')

    def read_file_list(self,cfg_filepath):
        with open(cfg_filepath,'r') as f:
            lines = f.read().splitlines()
            return lines

    def read_train_tups(self):
        train_img_list = self.read_file_list(self.train_channels_filepath)
        train_gt_list = self.read_file_list(self.train_gtlabels_filepath)
        train_mask_list = self.read_file_list(self.train_roimasks_filepath)
        return list(zip(train_img_list,train_mask_list,train_gt_list))

    def read_validation_tups(self):
        validation_img_list = self.read_file_list(self.validation_channels_filepath)
        validation_gt_list = self.read_file_list(self.validation_gtlabels_filepath)
        validation_mask_list = self.read_file_list(self.validation_roimasks_filepath)
        return list(zip(validation_img_list,validation_mask_list,validation_gt_list))

    def read_test_tups(self):
        test_img_list = self.read_file_list(self.test_channels_filepath)
        test_gt_list = self.read_file_list(self.test_gtlabels_filepath)
        test_mask_list = self.read_file_list(self.test_roimasks_filepath)
        return list(zip(test_img_list,test_mask_list,test_gt_list))
