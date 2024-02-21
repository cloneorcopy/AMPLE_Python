from utils import mesh_generator
import numpy as np
#from utils.DataBase_jit import MeshData, MpmData
from utils.DataBase import MeshData, MpmData
class AnalysisParametersBase:
    '''为ample 传入计算模型
    对于自定义的模型，应当覆写mesh_generator 这个方法
    '''
    def __init__(self) -> None:
        self.g=10
        self.lstps=40 
    def call(self):
        mesh,mpData = self.mesh_generator()
        return self.lstps,self.g,mpData,mesh
    def mesh_generator(self):
        pass
    
class SetupGridCollapse(AnalysisParametersBase):
    """Analysis parameters
    =setupGrid_collapse=
    """
    def __init__(self) :
        super().__init__()
        # setupGrid_collapse
        self.E=1e6                                                                # Young's modulus
        self.v=0.3                                                          # Poisson's ratio
        self.fc=20e3                                                       # yield strength                                                              
        self.mCst = np.array([self.E, self.v, self.fc],dtype=np.float64)                # material constants
        self.g=10                                                                       # gravity
        self.rho=1000                                                                   # material density
        self.lstps=40                                                                   # number of loadsteps
        self.a = 2                                                                      # element multiplier
        self.nelsx=4*self.a                                                                  # number of elements in the x direction
        self.nelsy=4*self.a                                                                    # number of elements in the y direction
        self.ly=8                                                              # domain dimensions
        self.lx=8                                                                # domain dimensions
        self.mp=6                                                                       # number of material points in each direction per element
        self.mpType = 2                                                                 # material point type: 1 = MPM, 2 = GIMP
        self.cmType = 2
        #mesh,mpData = self.mesh_generator()
    #def call(self):
    #    mesh,mpData = self.mesh_generator()
    #    return self.lstps,self.g,mpData,mesh

    def mesh_generator(self):
        etpl,coord = mesh_generator.formCoord2D(2*self.nelsx,self.nelsy,2*self.lx,self.ly)
        _,nen = etpl.shape
        nodes,nD = coord.shape
        h = np.array([self.lx/self.nelsx,self.ly/self.nelsy],dtype=np.float64)
        # boundary conditions on backgroun mesh
        bc = np.zeros((nodes*nD,2),dtype=np.int32)
        nodes_index  = np.argwhere(coord[:,0] == 0)
        bc[(nodes_index+1)*2-1,0] = (nodes_index+1)*2-1
        nodes_index = np.argwhere(coord[:,1] == 0)
        bc[(nodes_index+1)*2,0] = (nodes_index+1)*2
        bc = bc[bc[:, 0] > 0, :]
        bc[:,0] = bc[:,0]-1 #fix
        #Mesh data structure generation
        mesh = MeshData(etpl,coord,bc,h)
        #Material point generation
        ngp    = self.mp**nD
        GpLoc = mesh_generator.detMpPos(self.mp,nD)
        N = mesh_generator.shapefunc(nen,GpLoc,nD)
        etplmp,coordmp = mesh_generator.formCoord2D(self.nelsx,self.nelsy,self.lx,self.ly)
        nelsmp = etplmp.shape[0]
        nmp = ngp*nelsmp
        mpC = np.zeros((nmp,nD),dtype=np.float64)

        for nel in range(1, nelsmp + 1):
            indx = np.arange((nel - 1) * ngp, nel * ngp)  # MP locations within mpC
            eC = coordmp[etplmp[nel - 1] - 1, :]  # element coordinates
            mpPos = N@eC  # global MP coordinates
            mpC[indx, :] = mpPos  # store MP positions
            #np.matmul(N, eC)
        lp = np.zeros((nmp,2))
        lp[:,0] = h[0]/(2*self.mp)
        lp[:,1] = h[1]/(2*self.mp)
        vp = 2**(nD)*lp[:,0]*lp[:,1]
        #Material point structure generation
        mpmdata = MpmData()
        mpmdata.mpType=np.ones((nmp,1),dtype=np.uint8)*self.mpType
        mpmdata.cmType=np.ones((nmp,1),dtype=np.uint8)*self.cmType
        mpmdata.mpC=mpC
        mpmdata.vp=vp
        mpmdata.vp0=vp
        mpmdata.mpM=vp*self.rho
        mpmdata.nIN=[[] for _ in range(nmp)]
        mpmdata.eIN=[[] for _ in range(nmp)]
        mpmdata.Svp=[[] for _ in range(nmp)]
        mpmdata.dSvp=[[] for _ in range(nmp)]
        mpmdata.Fn=np.asarray([np.eye(3) for _ in range(nmp)])
        mpmdata.F=np.asarray([np.eye(3) for _ in range(nmp)])
        mpmdata.sig=np.zeros((nmp,6,1))
        mpmdata.epsEn=np.zeros((nmp,6,1))
        mpmdata.epsE=np.zeros((nmp,6,1))
        mpmdata.mCst=np.ones((nmp,1),dtype=np.float64)*self.mCst
        mpmdata.fp=np.zeros((nmp,nD,1))
        mpmdata.u=np.zeros((nmp,nD,1))
        mpmdata.nSMe   = np.zeros(nmp,dtype=np.uint32)

        if self.mpType == 2:
            mpmdata.lp=np.array([lp[mp] for mp in range(nmp)],dtype=np.float64)
            mpmdata.lp0=np.array([lp[mp] for mp in range(nmp)],dtype=np.float64)
        else: #bug
            mpmdata.lp=np.zeros((nmp,1,nD),dtype=np.float64)
            mpmdata.lp0=np.zeros((nmp,1,nD),dtype=np.float64)
        return mesh,mpmdata
