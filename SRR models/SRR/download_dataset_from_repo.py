import datalad.api as dl
import pandas as pd
import os
import glob
from colorama import Fore, Back, Style
import pathlib



def download_dataset(dataset_url:str=None, 
        target_dir:str=None,file_filter:str=None,get_files:bool=False)-> bool:
    """
    Download the dataset based on filters from openneuro.

    Parameters
    ----------
    dataset_url : str
        URL of the openneuro dataset
    target_dir : str
        path to download the data to
    file_filter : str
        type of contrast file to filter - T2w for example

    Returns
    -------
    success : bool
        if download is successful

    """
    
    dl.clone(source=dataset_url, path=target_dir)
    ds = dl.Dataset(target_dir)
    results = ds.status(annex='all')
    search_path = target_dir + '/**/' + file_filter
    file_list = glob.glob(search_path, recursive=True)
    print(file_list[0])
    file_list.sort()
    print(Fore.GREEN, len(file_list))

    if get_files is True:
        for file in file_list:
            ds_path_file = os.path.relpath(file, target_dir)
            result = ds.get(ds_path_file)
    results = ds.status(annex='all')
    return True

if __name__ == '__main__':
    dataset_url = 'https://github.com/OpenNeuroDatasets/ds004146.git'
    target_dir = './QTAB'
    file_filter = '*_T2w.nii.gz'
    get_files = True
    download_dataset(dataset_url=dataset_url, 
                            target_dir=target_dir,file_filter=file_filter,get_files=get_files)
    print(Style.RESET_ALL)

