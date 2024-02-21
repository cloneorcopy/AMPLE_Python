from typing import Any
import numpy as np
from utils.AnalysisParameters import SetupGridCollapse
from utils import vtk_helper
import os 
import time
import CONFIG 
if CONFIG.USE_JIT:
    from utils import jit_fun as mpm_fun
    #from utils.DataBase_jit import  MeshData, MpmData ,caculate_different
else:
    from utils import mpm_fun
from utils.DataBase import MeshData, MpmData ,caculate_different
def TD(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"     [info]:函数 {func.__name__} 的执行时间为: {execution_time} 秒")
        return result
    return wrapper
class MainFunction:
    """
        %AMPLE 1.1: A Material Point Learning Environment
    %--------------------------------------------------------------------------
    % Author: William Coombs
    % Date:   27/08/2020
    % Description:
    % Large deformation elasto-plastic (EP) material point method (MPM) code
    % based on an updated Lagrangian (UL) descripition of motion with a 
    % quadrilateral background mesh. 
    %
    %--------------------------------------------------------------------------
    % See also:
    % SETUPGRID             - analysis specific information
    % ELEMMPINFO            - material point-element information
    % DETEXTFORCE           - external forces
    % DETFDOFS              - mesh unknown degrees of freedom
    % LINSOLVE              - linear solver
    % DETMPS                - material point stiffness and internal force
    % UPDATEMPS             - update material points
    % POSTPRO               - post processing function including vtk output
    %--------------------------------------------------------------------------
        """
    def __init__(self,setup_class,DEBUG=0) -> None:
        if DEBUG:
            self.lstps = 40
            self.g=10
            self.mpData = MpmData()
            self.mpData.load_mat('mpmdata_raw.mat')
            self.mpData.nIN=[[] for _ in range(len(self.mpData.dSvp))]
            self.mpData.eIN=[[] for _ in range(len(self.mpData.dSvp))]
            self.mpData.Svp=[[] for _ in range(len(self.mpData.dSvp))]
            self.mpData.dSvp=[[] for _ in range(len(self.mpData.dSvp))]
            self.mesh = MeshData(etpl=None,coord=None,bc=None,h=None)
            self.mesh.load_mat('mesh_raw.mat')
            _,_,mpData_test,mesh_test = setup_class.call()
            caculate_different(self.mesh,self.mesh)
            caculate_different(self.mesh,mesh_test)
            caculate_different(self.mpData,self.mpData)
            caculate_different(self.mpData,mpData_test)
            def debug_load_compare(mpmdata_mat,
                                   mesh_mat,
                                   mpmdata_test,
                                   mesh_test):
                mpmdata = MpmData()
                mpmdata.load_mat(mpmdata_mat)
                mesh = MeshData(etpl=None,coord=None,bc=None,h=None)
                mesh.load_mat(mesh_mat)
                caculate_different(mpmdata,mpmdata_test)
                caculate_different(mesh,mesh_test)
                return mpmdata,mesh
        else:
            self.lstps,self.g,self.mpData,self.mesh = setup_class.call()
            
        
        self.NRitMax = 20
        self.tol = 1e-9
        self.nodes,self.nD = self.mesh.coord.shape
        #nels,nen = mesh.etpl.shape
        self.nDoF = self.nodes*self.nD
        self.nmp  = len(self.mpData)
        self.lstp = 0
        self.uvw = np.zeros((self.nDoF,1))
        if not DEBUG:
            #run postPro
            self.sig = np.reshape([self.mpData.sig], (self.nmp, 6))  # all material point stresses (nmp, 6)
            self.mpC = np.reshape([self.mpData.mpC], (self.nmp, self.nD))  # all material point coordinates (nmp, nD)
            self.mpU = self.mpData.u  # all material point displacements
            os.makedirs('output',exist_ok=True)
            self.mpDataName = f'output/mpData_{self.lstp}.vtk'  # MP output data file name
            vtk_helper.makeVtkMP(self.mpC, self.sig, self.mpU, self.mpDataName)  # generate material point VTK file
            self.meshName = f'output/mesh_{self.lstp}.vtk'  # MP output data file name
            vtk_helper.makeVtk(self.mesh.coord, self.mesh.etpl, self.uvw, self.meshName)
            ## 
        #start = time.time()
        for self.lstp in range(1,self.lstps+1):
            print(f"loadstep {self.lstp} of {self.lstps}")
            self.loop()
            #self.mesh,self.mpData = mpm_fun.elemMPinfo(self.mesh,self.mpData)
            ##mpData1,mesh1 = debug_load_compare(mpmdata_mat=f'debug_data\mpm{lstp}.mat',
            ##                    mesh_mat=f'debug_data\\mesh{lstp}.mat',
            ##                    mpmdata_test=mpData,
            ##                    mesh_test=mesh
            ##                    )
            #self.fext = mpm_fun.detExtForce(self.nodes,self.nD,self.g,self.mpData)       #external force calculation (total)
            #self.fext = self.fext*self.lstp/self.lstps                              #current external force value
            #self.oobf = self.fext                                         #initial out-of-balance force
            #self.fErr = 1                                            
            #self.frct = np.zeros((self.nDoF,1))                           #zero the reaction forces
            #self.uvw = np.zeros((self.nDoF,1))                            #zero the displacements
            #self.fd  = mpm_fun.detFDoFs(self.mesh)                        #free degrees of freedom
            #self.NRit = 0
            #self.Kt   = 0                                            #zero global stiffness matrix
            #while (self.fErr > self.tol) and (self.NRit < self.NRitMax) or (self.NRit < 2):
            #    self.duvw,self.drct = mpm_fun.linSolve(self.mesh.bc,self.Kt,self.oobf,self.NRit,self.fd) 
            #    self.uvw  = self.uvw+self.duvw
            #    self.frct = self.frct+self.drct
            #    self.fint,self.Kt,self.mpData = mpm_fun.detMPs(self.uvw,self.mpData)     #global stiffness & internal force
            #    #mpData1,mesh1 =debug_load_compare(
            #    #                mpmdata_mat=f'debug_data\mpm{lstp}_{NRit}.mat',
            #    #                mesh_mat=f'debug_data\\mesh{lstp}_{NRit}.mat',
            #    #                mpmdata_test=mpData,
            #    #                mesh_test=mesh
            #    #                )
            #    self.oobf = self.fext-self.fint.ravel()+self.frct.ravel()
            #    self.fErr = np.linalg.norm(self.oobf) / np.linalg.norm(self.fext + self.frct.ravel() + np.finfo(float).eps)
            #    self.NRit = self.NRit + 1
            #    print(f'  iteration {self.NRit} NR error {self.fErr:.3e} ')
            #    #print(f'  iteration {NRit} NR oobf {np.linalg.norm(oobf):.3e} ')
            #self.mpData = mpm_fun.updateMPs(self.uvw,self.mpData)              #update material points
            #end = time.time()
            #print(f"time: {end-start}")
            #print("")
                #run postPro
                #sig = np.reshape([mpData.sig], (nmp, 6))  # all material point stresses (nmp, 6)
                #mpC = np.reshape([mpData.mpC], (nmp, nD))  # all material point coordinates (nmp, nD)
                #mpU = mpData.u  # all material point displacements
                #mpDataName = f'output/mpData_{lstp}.vtk'  # MP output data file name
                #vtk_helper.makeVtkMP(mpC, sig, mpU, mpDataName)  # generate material point VTK file
                #meshName = f'output/mesh_{lstp}.vtk'  # MP output data file name
                #vtk_helper.makeVtk(mesh.coord, mesh.etpl, uvw, meshName)
                ## 
                # print('Time: ',lstp)
                # print('Time step: ',lstp)
        
    def loop(self):
        self.mesh,self.mpData = mpm_fun.elemMPinfo(self.mesh,self.mpData)
        #mpData1,mesh1 = debug_load_compare(mpmdata_mat=f'debug_data\mpm{lstp}.mat',
        #                    mesh_mat=f'debug_data\\mesh{lstp}.mat',
        #                    mpmdata_test=mpData,
        #                    mesh_test=mesh
        #                    )
        self.fext = mpm_fun.detExtForce(self.nodes,self.nD,self.g,self.mpData)       #external force calculation (total)
        self.fext = self.fext*self.lstp/self.lstps                              #current external force value
        self.oobf = self.fext                                         #initial out-of-balance force
        self.fErr = 1                                            
        self.frct = np.zeros((self.nDoF,1))                           #zero the reaction forces
        self.uvw = np.zeros((self.nDoF,1))                            #zero the displacements
        self.fd  = mpm_fun.detFDoFs(self.mesh)                        #free degrees of freedom
        self.NRit = 0
        self.Kt   = 0                                            #zero global stiffness matrix
        while (self.fErr > self.tol) and (self.NRit < self.NRitMax) or (self.NRit < 2):
            self.solve()
            print(f'  iteration {self.NRit} NR error {self.fErr:.3e} ')
        self.mpData = mpm_fun.updateMPs(self.uvw,self.mpData)  
    
    def solve(self):
        self.duvw,self.drct = mpm_fun.linSolve(self.mesh.bc,self.Kt,self.oobf,self.NRit,self.fd) 
        self.uvw  = self.uvw+self.duvw
        self.frct = self.frct+self.drct
        self.fint,self.Kt,self.mpData = mpm_fun.detMPs(self.uvw,self.mpData)     #global stiffness & internal force
        #mpData1,mesh1 =debug_load_compare(
        #                mpmdata_mat=f'debug_data\mpm{lstp}_{NRit}.mat',
        #                mesh_mat=f'debug_data\\mesh{lstp}_{NRit}.mat',
        #                mpmdata_test=mpData,
        #                mesh_test=mesh
        #                )
        self.oobf = self.fext-self.fint.ravel()+self.frct.ravel()
        self.fErr = np.linalg.norm(self.oobf) / np.linalg.norm(self.fext + self.frct.ravel() + np.finfo(float).eps)
        self.NRit = self.NRit + 1
if "__main__" == __name__:
    test1 = SetupGridCollapse()
    test1.call()
    MainFunction(test1)