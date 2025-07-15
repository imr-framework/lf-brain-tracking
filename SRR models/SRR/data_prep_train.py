import os
import glob
from colorama import Fore, Back, Style
import nibabel as nib
import numpy as np

def prepare_training_data(data_dir_list:list=None, file_extension_pattern:str=None,  xpattern:str=None,
                          batch_size:int=None, iteration_num: int = None):
    """
    Prepare the dataset to train based on batch size.

    Parameters
    ----------
    data_dir_list : list
        list of paths to the different input data
    xpattern : str
        string to identify xdata file
    batch_size : int
        Size of the batch for training
    iteration_num : int
        ith iteration to get the relevant x's for in a way that (batch_size * iteration_num) + 1 is the starting index of the file 
    file_extension_pattern : str
        pattern to identify file type such as .nii.gz

    Returns
    -------
    x_train : numpy.ndarray
        The training data in desired shape
    y_train : numpy.ndarray
        The label data in desired shape

    """
    start_index = (iteration_num * batch_size) 
    end_index = start_index + batch_size 
    if end_index > len(data_dir_list):
        print(Fore.RED + "Error: exceeded max. number of subjects")
    xpattern_len = len(xpattern)

    x_init_0 = True
    x_init_1 = True
    x_init_2 = True
    y_init = True

    for session in range(start_index, end_index):
        print(Fore.YELLOW + '---------------------------')
        fnames = data_dir_list[session] + '/*.'+ file_extension_pattern
        for fname in glob.glob(fnames, recursive=True):
            # print(Fore.BLUE + fname)
        # First check for xpattern because that is the modified
            found_x = False 
            if '._' in fname:
                 print(Fore.RED + 'some system file - not reading')
            else:  
                    if xpattern[0] in fname:
                        if x_init_0 is True:
                            img = nib.load(fname).get_fdata()
                            xm, xn, xo = img.shape
                            x_train_0 = np.zeros((xm, xn, xo, batch_size), dtype = float)
                            x_init_0 = False
                        head_tail = os.path.split(fname)
                        print(Fore.YELLOW + 'ds0: ' + fname)
                        x_train_0[:, :, :,  session] = nib.load(fname).get_fdata()
                        found_x = True

                    elif xpattern[1] in fname:
                        if x_init_1 is True:
                            img = nib.load(fname).get_fdata()
                            xm, xn, xo = img.shape
                            x_train_1 = np.zeros((xm, xn, xo, batch_size), dtype = float)
                            x_init_1 = False
                        head_tail = os.path.split(fname)
                        print(Fore.YELLOW + 'ds1: ' + fname)
                        x_train_1[:, :, :, session] = nib.load(fname).get_fdata()
                        found_x = True

                    elif xpattern[2] in fname:
                        if x_init_2 is True:
                            img = nib.load(fname).get_fdata()
                            xm, xn, xo = img.shape
                            x_train_2 = np.zeros((xm, xn, xo, batch_size), dtype = float)
                            x_init_2 = False
                        head_tail = os.path.split(fname)
                        print(Fore.YELLOW + 'ds2: ' + fname)
                        x_train_2[:, :, :,  session] = nib.load(fname).get_fdata()
                        found_x = True
                    else:
                        if found_x is False:
                            if y_init is True:
                                img = nib.load(fname).get_fdata()
                                ym, yn, yo = img.shape
                                y_train = np.zeros((ym, yn, yo, batch_size), dtype = float)
                                y_init = False
                            head_tail = os.path.split(fname)
                            print(Fore.YELLOW + 'y: ' + fname)
                            y_train[:, :, :, session] = nib.load(fname).get_fdata()

    return x_train_0, x_train_1, x_train_2, y_train




def get_datafile_paths(datadir:str=None, file_extension_pattern:str =None, 
                       folder_pattern:str=None, file_pattern_avoid:str = None):
    """
    Get the list of folders for training

    Parameters
    ----------
    datadir : str
        path to the main data folder
    file_extension_pattern : str
        pattern to identify file type such as .nii.gz
    folder_pattern : str
        pattern to identify relevant folder such as anat
    file_pattern_avoid : str
        pattern to identify file names to avoid such as 'ds' (downsampled)
   
 
    Returns
    -------
   data_dir_list : list
        list of paths - one for each folder
    """
    data_dir_list = []
    subjects = os.listdir(datadir)
    for subject in subjects:
        if 'sub' in subject:
            if '._' in subject:
                 print(Fore.RED + 'some system file - not reading')
            else:
                # session_dirs = os.listdir(os.path.join(topdir, subject))
                search_path = os.path.join(topdir, subject) + '/**/*.' + file_extension_pattern
                for fname in glob.glob(search_path, recursive=True):
                    if folder_pattern in fname:      # Focusing only on anatomical scans
                        if  not file_pattern_avoid in fname:        # avoids repeating foldernames
                            head_tail = os.path.split(fname)
                            sub_dir = head_tail[0]
                            data_dir_list.append(sub_dir)
    return data_dir_list


if __name__ == '__main__':
    # topdir = './Data'
    topdir = './SUDMEX/'
    batch_size = 6
    iteration_num = 0
    xpattern = ['ds0', 'ds1', 'ds2']
    file_extension_pattern = 'nii.gz'
    data_dir_list = get_datafile_paths(datadir=topdir, file_extension_pattern='nii.gz',
                                       folder_pattern='anat', file_pattern_avoid='ds')
    
    print(Fore.BLUE + "Found sessions: " + str(len(data_dir_list)))
    x_train_0, x_train_1, x_train_2, y_train = prepare_training_data(data_dir_list=data_dir_list, xpattern=xpattern,
                          file_extension_pattern=file_extension_pattern,batch_size=batch_size, iteration_num =iteration_num)

    print(x_train_2.shape)