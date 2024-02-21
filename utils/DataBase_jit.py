import numpy as np
from scipy.io import loadmat
from numba.experimental import jitclass
from numba import int32, float32,int8    # import the types

MpmData_spec = [
    ('nmp', int32),               # a simple scalar field
    ('nD', int8),          # an array field
]
@jitclass(MpmData_spec)
class MpmData:
    """
    MpmData :class for structured array with the following fields :
    #           - mpType : material point type (1 = MPM, 2 = GIMPM)
    #           - cmType : constitutive model type (1 = elastic, 2 = vM)
    #           - mpC    : material point coordinates
    #           - vp     : material point volume
    #           - vp0    : initial material point volume
    #           - mpM    : material point mass
    #           - nIN    : nodes linked to the material point
    #           - eIN    : element associated with the material point
    #           - Svp    : basis functions for the material point
    #           - dSvp   : basis function derivatives (at start of lstp)
    #           - Fn     : previous deformation gradient
    #           - F      : deformation gradient
    #           - sig    : Cauchy stress
    #           - epsEn  : previous logarithmic elastic strain
    #           - epsE   : logarithmic elastic strain
    #           - mCst   : material constants (or internal state parameters)
    #           - fp     : force at the material point
    #           - u      : material point displacement
    #           - lp     : material point domain lengths
    #           - lp0    : initial material point domain lengths
    """
    
    def __init__(self):
        pass
    def __len__(self):
        return len(self.mpC)
    def init_para(self,nmp,nD=2):
        self.mpType=np.ones((nmp,1),dtype=np.uint8)
        self.cmType=np.ones((nmp,1),dtype=np.uint8)
        self.mpC=np.zeros((nmp,nD),dtype=np.float64)
        self.vp=np.ones((nmp,1),dtype=np.float64)
        self.vp0=np.ones((nmp,1),dtype=np.float64)
        self.mpM=np.ones((nmp,1),dtype=np.float64)
        self.nIN=[[] for _ in range(nmp)]
        self.eIN=[[] for _ in range(nmp)]
        self.Svp=[[] for _ in range(nmp)]
        self.dSvp=[[] for _ in range(nmp)]
        self.Fn=np.asarray([np.eye(3) for _ in range(nmp)])
        self.F=np.asarray([np.eye(3) for _ in range(nmp)])
        self.sig=np.zeros((nmp,6,1))
        self.epsEn=np.zeros((nmp,6,1))
        self.epsE=np.zeros((nmp,6,1))
        self.mCst=np.ones((nmp,3),dtype=np.float64)
        self.fp=np.zeros((nmp,nD,1))
        self.u=np.zeros((nmp,nD,1))
        self.nSMe   = np.zeros(nmp,dtype=np.uint32)
        self.lp = np.zeros((nmp,2))
        self.lp0 = np.zeros((nmp,2))
    def load_mat(self,mat_path):
        """from matlab load mat file
        used for debug
        """
        mpData = loadmat(mat_path)['mpData'][0]
        para_map = ['mpType','cmType','mpC','vp','vp0',
                    'mpM','nIN','eIN','Svp','dSvp',
                    'Fn','F','sig','epsEn','epsE',
                    'mCst','fp','u','lp','lp0','nSMe']
        if len(mpData[0]) == 20:
            para_map = para_map[:-1]
        nD = len(mpData[0][2][0])
        num = len(mpData)
        self.init_para(num,nD)
        for mp in range(num):
            for para in range(len(para_map)):
                if para_map[para] in ['dSvp','Fn','F','sig','eIN','epsEn','epsE','fp','u'] :
                    self[para_map[para]][mp] = mpData[mp][para]
                else:
                    self[para_map[para]][mp] = mpData[mp][para][0]
        self.mpM = self.mpM.flatten()
        self.vp = self.vp.flatten()
        self.vp0 = self.vp0.flatten()
        


class MeshData:
    etpl:np.ndarray
    coord:np.ndarray
    bc:np.ndarray
    h:np.ndarray
    def __init__(self,etpl:np.ndarray,coord:np.ndarray,bc:np.ndarray,h:np.ndarray) -> None:
        self.etpl  = etpl   # element topology                                                          # element topology
        self.coord = coord   # nodal coordinates                                               # nodal coordinates
        self.bc    = bc   # boundary conditions                                                            # boundary conditions
        self.h     = h   # mesh size
    def load_mat(self,mat_path):
        """from matlab load mat file
        used for debug
        """
        mpData = loadmat(mat_path)['mesh'][0]
        para_map = ['etpl','coord','bc','h','eInA']
        if len(mpData[0]) == 4:
            para_map = para_map[:-1]
        for para in range(len(para_map)):
            self[para_map[para]] = mpData[0][para]
        self.bc[:,0] = self.bc[:,0]-1
        self.etpl= self.etpl.astype(np.int32)
        self.coord = self.coord.astype(np.float64)
        #self.bc = mpData[0][2]
        self.h = self.h.astype(np.float64)
        if len(mpData[0]) != 4:
            self.eInA = self.eInA.flatten()
            
def caculate_different(class1,class2):
    erro_list = []
    for key in class1.__dict__:
        if key in ['nIN','eIN','Svp','dSvp']:
            for i in range(len(class1[key])):
                if not np.allclose(class1[key][i],class2[key][i], rtol=1e-5, atol=1e-10):
                    erro_list.append(key)
        else:
            if not np.allclose(class1[key],class2[key], rtol=1e-5, atol=1e-10):
                erro_list.append(key)
    erro_list = list(set(erro_list))
    class1_name = class1.__class__.__name__
    class2_name = class2.__class__.__name__
    if len(erro_list):
        print(f'{class1_name}-{class2_name}','not match keys:',erro_list)
        print('  detail:')
        for key in erro_list:
            c_list = []
            print(key)
            if key in ['nIN','eIN','Svp','dSvp']:
                for i in range(len(class1[key])):
                    if not np.allclose(class1[key][i],class2[key][i], rtol=1e-5, atol=1e-10):
                       c_list.append(i)
                print(f"    {key}:",c_list)
            else:
                is_close =np.isclose(class1[key],class2[key], rtol=1e-5, atol=1e-10)
                indices = np.where(~is_close)
                print(f"    {key}:",indices)
    