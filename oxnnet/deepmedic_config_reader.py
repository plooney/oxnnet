"""Modules containing a single class to read a deep medic config"""
import os

def read_file_list(cfg_filepath):
    """Reads a file and returns a list split by lines"""
    with open(cfg_filepath, 'r') as f:
        lines = f.read().splitlines()
        return lines

class DeepMedicConfigReader(object):
    """Class to read a config file used for deepmedic
    and parse the train, validation and test cases"""
    def __init__(self, project_directory):
        self.train_channels_filepath = os.path.join(project_directory,
                                                    'train', 'trainChannels.cfg')
        self.train_gtlabels_filepath = os.path.join(project_directory,
                                                    'train', 'trainGtLabels.cfg')
        self.train_roimasks_filepath = os.path.join(project_directory,
                                                    'train', 'trainRoiMasks.cfg')
        self.validation_channels_filepath = os.path.join(project_directory, 'train',
                                                         'validation', 'validationChannels.cfg')
        self.validation_roimasks_filepath = os.path.join(project_directory, 'train',
                                                         'validation', 'validationRoiMasks.cfg')
        self.validation_gtlabels_filepath = os.path.join(project_directory, 'train',
                                                         'validation', 'validationGtLabels.cfg')
        self.test_channels_filepath = os.path.join(project_directory, 'test', 'testChannels.cfg')
        self.test_roimasks_filepath = os.path.join(project_directory, 'test', 'testRoiMasks.cfg')
        self.test_gtlabels_filepath = os.path.join(project_directory, 'test', 'testGtLabels.cfg')

    def read_train_tups(self):
        """Read the tules of filepaths for training"""
        train_img_list = read_file_list(self.train_channels_filepath)
        train_gt_list = read_file_list(self.train_gtlabels_filepath)
        train_mask_list = read_file_list(self.train_roimasks_filepath)
        return list(zip(train_img_list, train_mask_list, train_gt_list))

    def read_validation_tups(self):
        """Read the tules of filepaths for validation"""
        validation_img_list = read_file_list(self.validation_channels_filepath)
        validation_gt_list = read_file_list(self.validation_gtlabels_filepath)
        validation_mask_list = read_file_list(self.validation_roimasks_filepath)
        return list(zip(validation_img_list, validation_mask_list, validation_gt_list))

    def read_test_tups(self):
        """Read the tules of filepaths for testing"""
        test_img_list = read_file_list(self.test_channels_filepath)
        test_gt_list = read_file_list(self.test_gtlabels_filepath)
        test_mask_list = read_file_list(self.test_roimasks_filepath)
        return list(zip(test_img_list, test_mask_list, test_gt_list))
