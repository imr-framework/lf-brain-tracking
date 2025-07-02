import scipy.ndimage as ndimage
import os
import numpy as np
import struct
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Read the acquisition parameters
class kea3d():
    def __init__(self, data_folder:str=None, sub_folder:str=None, 
                 noise_scan:str=None, freq_drift:int=None, do_drift_corr:bool=True,
                 do_Gauss_filter:bool=True, center_kspace: bool = True, p1_param:float = 1/2, p2_param=1/3, del_freq:int = -500):
    
        self.data_folder = data_folder
        self.sub_folder = sub_folder
        self.noise_scan = noise_scan
        self.freq_drift = freq_drift
        self.do_drift_corr = do_drift_corr
        self.do_Gauss_filter = do_Gauss_filter
        self.del_freq = del_freq
        self.acq_params = self.read_acq_par()
        
        # Determine resolution in image space
        self.res_dim1 = np.float64(self.acq_params["FOVread"])/ np.float64(self.acq_params["nrPnts"])
        self.res_dim2 = np.float64(self.acq_params["FOVphase1"])/ np.float64(self.acq_params["nPhase1"])
        self.res_dim3 = np.float64(self.acq_params["FOVphase2"])/ np.float64(self.acq_params["nPhase2"])
        self.p1_param = p1_param
        self.p2_param = p2_param
        
        
        self.trajectory = self.get_trajectory()
        self.kspace = self.get_kspace()
        
        if center_kspace is True:
            self.kspace = self.center()
        
        if self.do_drift_corr is True:
            self.kspace_drift_corr = self.drift_correction(self.del_freq)
        else:
            self.kspace_drift_corr = 0
            
        if self.do_Gauss_filter is True:
            self.kspace_gauss_filter = self.Gauss_filter(self.p1_param, self.p2_param)
        else:
            self.kspace_gauss_filter= 0
            
        
            
# Read the acqpar file to get the acq params
    def read_acq_par(self):
        fname = os.path.join(self.data_folder, self.sub_folder, 'acqu.par')
        acq_params = dict()
        
        with open(fname, 'r') as file:
            for line in file:
                param = line.strip()  
                acq_params[param.split()[0]] = param.split()[2] # still a string, later overloaded method to convert to relevant datatype
        return acq_params
            
# Read k-space trajectory 
    def get_trajectory(self):
        fname = os.path.join(self.data_folder, self.sub_folder, 'trajectory.csv')
        trajectory = pd.read_csv(fname, header=None, index_col = False).to_numpy()
        return trajectory

# Read k-space data - Improves on implementations by Craig Eccles (Magritek) and Tom O'Reilly (LUMC)
    def get_kspace(self):
            fname = os.path.join(self.data_folder, self.sub_folder, 'data.3d')
            # blocksize = 4 # Kea
            nro = int(self.acq_params["nrPnts"])
            npe1 = int(self.acq_params["nPhase1"])
            npe2 = int(self.acq_params["nPhase2"])
            kspace = np.zeros((nro, npe1, npe2), dtype=complex)
            
            data_size = nro * npe1 * npe2 * 2 # real and imaginary
            header = []
            f = open(fname,"rb")
            blocksize = 4
            for idx in range(8): # get the header bytes out of the way before reading data
                header.append(f.read(blocksize))
            rawData = f.read()
            f.close()

            rawData = np.array(struct.unpack(str(data_size)+'f',rawData))
            rawData = rawData[::2] + 1j*rawData[1::2]
            kspace = np.reshape(rawData .T, (nro, npe1, npe2), order = 'F') # pe order handled post acq. writing to file
            return kspace
            
# Perform drift correction - Improves on implementations by Craig Eccles (Magritek) and Tom O'Reilly (LUMC)
    def drift_correction(self, del_freq):
        shotLayout = np.zeros(np.shape(self.trajectory.T))
        acqTime = float(self.acq_params['acqTime'])*1e-3
        
        nro = int(self.acq_params["nrPnts"])
        npe1 = int(self.acq_params["nPhase1"])
        npe2 = int(self.acq_params["nPhase2"])
        
        try:
            ETL = int(self.acq_params['etLength'])
        except:
            ETL = 1
            
        numShots = (npe1 * npe2) / ETL
        driftPerShot = (del_freq)/numShots
        
        shotLayout = np.mod(self.trajectory.T, ETL)
        drift = shotLayout * driftPerShot 
                
        timeScale = np.linspace(-acqTime/2, acqTime/2, nro, endpoint = True)

        phaseCorrection = np.exp(-1j*2*np.pi*timeScale[:,np.newaxis,np.newaxis] * drift[np.newaxis,:,:])        
        shiftedData = np.multiply(self.kspace,phaseCorrection)
                
        return shiftedData  

# Perform any other optional Gaussian filtering - Improves on implementations by Craig Eccles (Magritek) and Tom O'Reilly (LUMC)
    def Gauss_filter(self, p1_param, p2_param):
        # self.kspace_gauss_filter  = ndimage.gaussian_filter(self.kspace , self.sigma) - needs more control params to tune better
        kspace = self.kspace
        input_shape = np.shape(kspace)
        filterMat = 1
        for dim_size in input_shape:
            N = dim_size
            p1 = N * p1_param
            p2 = N * p2_param
            filterVec = np.exp(-(np.square(np.arange(N) - p1)/(p2**2)))
            filterMat = np.multiply.outer(filterMat, filterVec)
        kspace_gauss_filtered = np.multiply(kspace, filterMat)
        return kspace_gauss_filtered
    
    def center(self):
        kspace = self.kspace
        max_xyz = np.asarray((np.where(np.abs(kspace) == np.abs(kspace).max())))
        dx = np.int32(max_xyz[0] - 0.5 * kspace.shape[0])
        dy = np.int32(max_xyz[1] - 0.5 * kspace.shape[1])
        dz = np.int32(max_xyz[2] - 0.5 * kspace.shape[2])

        kspace = np.roll(kspace, -dx, 0)
        kspace = np.roll(kspace, -dy, 1)
        kspace = np.roll(kspace, -dz, 2)
        
        return kspace


        
