from numba import jit 
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as spla
import time
def TD(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"     [info]:函数 {func.__name__} 的执行时间为: {execution_time} 秒")
        return result
    return wrapper
#@jit()
def elemForMP(coord,etpl,mpC,lp):
    """Find elements associated with the material point
    
    Author: William Coombs
    Date:   06/05/2015
    Description:
    Function to determine the elements that are associated with a material
    point assuming that the material point's domain is symmetric about the
    particle position.
    
    [eIN] = ELEMFORMP(coord,etpl,mpC,lp)
    
     Input(s):
     coord - element coordinates (nen,nD)
     etpl  - element topology (nels,nen)
     mpC   - material point coordinates (1,nD) 
     lp    - domain half width
    
     Ouput(s);
     eIN   - vector containing the elements associated with the mp
    
     See also:
    
    """
    nD = coord.shape[1]
    nels = etpl.shape[0]
    Pmin = mpC - lp
    Pmax = mpC + lp
    #print('mpC',repr(mpC[1]))
    #print('lp',repr(lp[1]))
    #print('Pmax',repr(Pmax[1]))
    a = np.ones((nels, 1)) 
    for i in range(nD):
        ci = coord[:, i]  # nodal coordinates in current i direction
        c = ci[etpl - 1]  # reshaped element coordinates in current i direction
        Cmin = np.min(c, axis=1)  # element lower coordinate limit
        Cmax = np.max(c, axis=1)  # element upper coordinate limit
        #test = Cmin- Pmax[i]
        #print(repr(Cmin[17]))
        #print(repr(Pmax[i]))
        a = a * (np.reshape(np.less(Cmin , Pmax[i]),(-1,1)) * np.reshape(np.greater(Cmax , Pmin[i]),(-1,1)))  # element overlap with mp domain
    eIN = np.reshape(np.arange(1, nels + 1),(-1,1))  # list of all elements
    #eIN = np.arange(1, nels + 1)
    eIN = eIN[a.flatten() > 0]
    return eIN
#@jit(nopython=True)
def nodesForMP(etpl,elems):
    """
    Unique list of nodes associated with a material point

     Author: William Coombs
     Date:   07/05/2015
     Description:
     Function to determine a unique list of nodes for a group of elements
     connected to a material point.


     [nodes] = NODESFORMP(etpl,elems)

     Input(s):
     etpl  - element topology (nels,nen)
     elems - elements in the group (n,1)

     Ouput(s);
     nodes - vector containing the nodes associated with the elements

     See also:


    """
    nen = etpl.shape[1]
    n = elems.shape[0]*elems.shape[1]
    #n  = len(elems)
    nn = n*nen
    e = np.sort(etpl[elems.flatten() - 1, :].reshape(nn, 1), axis=0)
    d = np.concatenate(([True], np.diff(e.flatten()) > 0))
    nodes = e[d]
    return nodes.flatten()
@jit(nopython=True)
def SvpGIMP(xp,xv,h,lp): #
    if (-h-lp)<(xp-xv) and (xp-xv)<=(-h+lp) : #A
        Svp = (h+lp+(xp-xv))**2/(4*h*lp) #c
        dSvp= (h+lp+(xp-xv))/(2*h*lp)    #c
    elif (-h+lp)<(xp-xv) and (xp-xv)<=(  -lp): #B
        Svp = 1+(xp-xv)/h #c
        dSvp= 1/h #c
    elif (  -lp)<(xp-xv) and (xp-xv)<=(   lp): #C
        Svp = 1-((xp-xv)**2+lp**2)/(2*h*lp) #c
        dSvp=-(xp-xv)/(h*lp)#c
    elif (   lp)<(xp-xv) and (xp-xv)<=( h-lp):#D
        Svp = 1-(xp-xv)/h#c
        dSvp=-1/h #
    elif ( h-lp)<(xp-xv) and (xp-xv)<=( h+lp): #E
        Svp = (h+lp-(xp-xv))**2/(4*h*lp) #c
        dSvp=-(h+lp-(xp-xv))/(2*h*lp)#c
    else:
        Svp =0 
        dSvp=0
    return Svp,dSvp
@jit(nopython=True)
def SvpMPM(xp,xv,h): #c
    if -h<(xp-xv) and (xp-xv)<=0:                                               
        Svp = 1+(xp-xv)/h
        dSvp= 1/h
    elif  0<(xp-xv) and (xp-xv)<=h:                                          
        Svp = 1-(xp-xv)/h  
        dSvp=-1/h
    else :                                                                     
        Svp =0
        dSvp=0
    return Svp,dSvp
#@jit()
def MPMbasis(mesh,mpData_mpC,mpData_lp,mpData_mpType,node):
    """
        Basis functions for the material point method

         Author: William Coombs
         Date:   29/01/2019
         Description:
         Function to determine the multi-dimensional MPM shape functions from the
         one dimensional MPM functions.  The function includes both the standard
         and generalised interpolation material point methods. 


         [Svp,dSvp] = MPMBASIS(coord,mpC,L)

         Input(s):
         mesh   - mesh data structured array. Function requires:
                   - coord  : nodal coordinates  
                   - h      : grid spacing

         mpData - material point structured array.  Function requires:
                   - mpC    : material point coordinates (single point)
                   - lp     : particle domain lengths
                   - mpType : material point type (1 or 2)

         node   - background mesh node number

         Ouput(s);
         Svp   - particle characteristic function
         dSvp  - gradient of the characterstic function 

         See also:
         SVPMPM    - MPM basis functions in 1D (mpType = 1
         SVPGIMP   - GIMPM basis functions in 1D (mpType = 2)

    """
    #coord  = np.reshape(mesh.coord[node,:],(-1,1))
    #h      = np.reshape(mesh.h,(-1,1))
    #mpC    = np.reshape(mpData_mpC,(-1,1))
    #lp     = np.reshape(mpData_lp,(-1,1))
    
    coord = mesh.coord[node,:].flatten()
    h = mesh.h.flatten()
    mpC = mpData_mpC.flatten()
    lp = mpData_lp.flatten()
    mpType = mpData_mpType
    nD = mpC.shape[0]
    
    #S = np.zeros((nD,1))
    #dS = np.zeros((nD,1))
    #dSvp = np.zeros((nD,1))
    
    S = np.zeros(nD)
    dS = np.zeros(nD)
    dSvp = np.zeros(nD)
    for i in range(nD):
        if mpType == 1:
            S[i],dS[i] = SvpMPM(mpC[i],coord[i],h[i])
        elif mpType == 2:
            S[i],dS[i] = SvpGIMP(mpC[i],coord[i],h[i],lp[i])
    
    if nD == 1:
        indx = []  # index for basis derivatives (1D)
    elif nD == 2:
        indx = [[1], [0]]  # index for basis derivatives (2D)
    elif nD == 3:
        indx = [[1, 2], [0, 2], [0, 1]]  # index for basis derivatives (3D)
    Svp = np.prod(S)  # basis function
    #dSvp = np.zeros(nD)
    for i in range(nD):
        dSvp[i] = dS[i] * np.prod(S[indx[i]])
    return Svp,dSvp
#@jit()
def elemMPinfo(mesh,mpData):
    """
Determine the basis functions for material points 

 Author: William Coombs
 Date:   29/01/2019
 Description:
 Function to determine the basis functions and spatial derivatives of each
 material point.  The function works for regular background meshes with
 both the standard and generalised interpolation material point methods.
 The function also determines, and stores, the elements associated with
 the material point and a unique list of nodes that the material point
 influences.  The number of stiffness matrix entries for each material
 point is determined and stored. 


 [fbdy,mpData] = ELEMMPINFO(mesh,mpData)

 Input(s):
 mesh   - mesh structured array. Function requires: 
           - coord : coordinates of the grid nodes (nodes,nD)
           - etpl  : element topology (nels,nen) 
           - h     : background mesh size (nD,1)
 mpData - material point structured array.  Function requires:
           - mpC   : material point coordinates

 Ouput(s);
 mesh   - mesh structured array. Function modifies:
           - eInA  : elements in the analysis 
 mpData - material point structured array. Function modifies:
           - nIN   : nodes linked to the material point
           - eIN   : element associated with the material point
           - Svp   : basis functions for the material point
           - dSvp  : basis function derivatives (at start of lstp)
           - nSMe  : number stiffness matrix entries for the MP

 See also:
 ELEMFORMP         - find elements for material point
 NODESFORMP        - nodes associated with a material point 
 MPMBASIS          - MPM basis functions

    """
    nmp = len(mpData)
    _,nD = mesh.coord.shape
    nels,_ = mesh.etpl.shape
    mpc = mpData.mpC
    lp = mpData.lp
    
    eInA = np.zeros(nels,dtype=np.uint8)
    for mp in range(nmp): #fix 大循环
        eIN = elemForMP(mesh.coord,mesh.etpl,mpc[mp,:],lp[mp,:])
        nIN  = nodesForMP(mesh.etpl,eIN)
        nn = len(nIN)
        #Svp = np.zeros((1,nn))
        Svp = np.zeros((nn))
        dSvp = np.zeros((nD,nn))
        for i in range(nn):
            node = nIN[i]-1
            S,dS = MPMbasis(mesh,mpc[mp],lp[mp],mpData.mpType[mp],node)
            #Svp[:,i] = Svp[:,i] + S
            Svp[i] += S
            dSvp[:,i] = dSvp[:,i] + np.reshape(dS,dSvp[:,i].shape)
        mpData.nIN[mp] = nIN
        mpData.eIN[mp] = eIN
        mpData.Svp[mp] = Svp
        mpData.dSvp[mp] = dSvp
        mpData.nSMe[mp] = (nn*nD)**2
        eInA[eIN-1] = 1
    #tt = np.sum(mpData.nSMe) #bug
    mesh.eInA = eInA
    return mesh,mpData
#@jit()
def detExtForce(nodes,nD,g,mpData):
    """
%Global external force determination  
%--------------------------------------------------------------------------
% Author: William Coombs
% Date:   23/01/2019
% Description:
% Function to determine the external forces at nodes based on body forces
% and point forces at material points.
%
%--------------------------------------------------------------------------
% [fbdy,mpData] = DETEXTFORCE(coord,etpl,g,eIN,mpData)
%--------------------------------------------------------------------------
% Input(s):
% nodes  - number of nodes (total in mesh)
% nD     - number of dimensions
% g      - gravity
% mpData - material point structured array. Function requires:
%           mpM   : material point mass
%           nIN   : nodes linked to the material point
%           Svp   : basis functions for the material point
%           fp    : point forces at material points
%--------------------------------------------------------------------------
% Ouput(s);
% fext   - external force vector (nodes*nD,1)
%--------------------------------------------------------------------------
% See also:
% 
%--------------------------------------------------------------------------

"""
    nmp = len(mpData)
    fext = np.zeros((nodes*nD))
    grav = np.zeros((nD,1))
    grav[nD-1] = -g
    for mp in range(nmp):
        nIN = mpData.nIN[mp]
        nn = len(nIN)
        Svp = mpData.Svp[mp].reshape(1,-1)
        fp = (mpData.mpM[mp]*grav + mpData.fp[mp])@Svp
        ed = np.tile((nIN.astype(np.int64)-1)*nD, (nD, 1)).ravel() + np.tile(np.arange(0, nD).reshape(-1, 1), (1, nn)).ravel()
        fext[ed] = fext[ed] + fp.ravel()
    return fext
@jit(nopython=True)
def parDerGen(X,eV,eP,yP,ydash):
    """%Partial derivative of a second order tensor function
%--------------------------------------------------------------------------
% Author: William Coombs
% Date:   27/05/2015
% Description:
% Function to determine the partial derivative of a second order tensor
% function with respect to its arguement (X) based on the implementation 
% described by in the following paper:
%
% C. Miehe, Comparison of two algorithms for the computation of fourth-
% order isotropic tensor functions, Computers & Structures 66 (1998) 37-43.
%
% For example, in order to determine the derivative of log(X) with respect
% to X the inputs to the function should be:
%
% [L] = PARDERGEN(X,eV,eP,log(eP),1./eP)
%
% as the derivative of the log(x) is 1/x
%
% The symbols used in the code follow, as closely as possible, those used
% in the Miehe (1998) paper.  There are a number of different cases that
% have to be checked (all zero and repeated eigenvalues) in addition to the
% general case where there are no repeated eigenvalues.  
%
%--------------------------------------------------------------------------
% [L] = PARDERGEN(X,eV,eP,yP,ydash)
%--------------------------------------------------------------------------
% Input(s):
% X     - second order tensor in matrix format (3,3)
% eV    - eigenvectors of X (3,3)
% eP    - eigenvalues of X (1,3) 
% yP    - function applied to eP (1,3)
% ydash - derivative of the function applied to eP (1,3)
%--------------------------------------------------------------------------
% Ouput(s);
% L      - partial derivative of the second order tensor with respect to 
%          its arguement (6,6)
%--------------------------------------------------------------------------
% See also:
% 
%--------------------------------------------------------------------------"""
    tol=1e-9
    Is=np.diag([1, 1, 1, 0.5, 0.5, 0.5])
    if (np.abs(eP[0]) < tol and np.abs(eP[1]) < tol and np.abs(eP[2]) < tol):
        L = Is
    elif (np.abs(eP[0] - eP[1]) < tol and np.abs(eP[0] - eP[2]) < tol):
        L = ydash[0] * Is
    elif abs(eP[0] - eP[1]) < tol or abs(eP[1] - eP[2]) < tol or abs(eP[0] - eP[2]) < tol:
        
        if abs(eP[0] - eP[1])<tol:
            xa = eP[2]
            xc = eP[0]
            ya = yP[2]
            yc = yP[0]
            yda = ydash[2]
            ydc = ydash[0]
        elif abs(eP[1] - eP[2]) < tol:
            xa = eP[0]
            xc = eP[1]
            ya = yP[0]
            yc = yP[1]
            yda = ydash[0]
            ydc = ydash[1]
        else:
            xa = eP[1]
            xc = eP[0]
            ya = yP[1]
            yc = yP[0]
            yda = ydash[1]
            ydc = ydash[0]
        #X = X.ravel()
        x = X.flat[[0, 4, 8, 3, 5, 2]]
        s1 = (ya-yc)/(xa-xc)**2-ydc/(xa-xc)
        s2 = 2*xc*(ya-yc)/(xa-xc)**2-(xa+xc)/(xa-xc)*ydc
        s3 = 2*(ya-yc)/(xa-xc)**3-(yda+ydc)/(xa-xc)**2
        s4 = xc*s3
        s5 = xc**2*s3
        dX2dX = np.array([  [2*X.flat[0],    0,           0,           X[1],                     0,                        X.flat[2]            ],
                            [0,              2*X.flat[4], 0,           X[1],                     X.flat[5],                0               ],
                            [0,              0,           2*X.flat[8], 0,                        X.flat[5],                X.flat[2]            ],
                            [X.flat[1],      X.flat[1],   0,           (X.flat[0]+X.flat[4])/2,  X[2]/2,                   X.flat[5]/2          ],
                            [0,              X.flat[5],   X.flat[5],   X[2].flat/2,              (X.flat[4]+X.flat[8])/2,  X.flat[1]/2          ],
                            [X.flat[2],      0,           X.flat[2],   X[5].flat/2,              X.flat[1]/2,              (X.flat[0]+X.flat[8])/2   ]])
        bm1 = np.array([1, 1, 1, 0, 0, 0])
        bm11 = np.array([   [1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]])
        L = s1 * dX2dX - s2 * Is - s3 * np.outer(x, x) + s4 * (np.outer(x, bm1) + np.outer(bm1, x)) - s5 * bm11
    else:
        
        D = np.array([  (eP[0] - eP[1]) * (eP[0] - eP[2]),
                        (eP[1] - eP[0]) * (eP[1] - eP[2]),
                        (eP[2] - eP[0]) * (eP[2] - eP[1])])
        alfa = 0
        bta = 0
        gama = np.zeros((3, 1))
        eDir = np.zeros((6, 3))

        for i in range(3):
            alfa += yP[i] * eP[i] / D[i]
            bta += yP[i] / D[i] * np.linalg.det(X)
            for j in range(3):
                gama[i] += yP[j] * eP[j] / D[j] * (np.linalg.det(X) / eP[j] - eP[i]**2) * 1 / eP[i]**2
            esq = np.outer(eV[:, i], eV[:, i])
            eDir[:, i] = [esq[0, 0], esq[1, 1], esq[2, 2], esq[0, 1], esq[1, 2], esq[2, 0]]

        y = np.linalg.inv(X)
        
        Ib = np.array([[y.flat[0]**2,           y.flat[1]**2,           y.flat[6]**2,           y.flat[0]*y.flat[1],                        y.flat[1]*y.flat[6],                            y.flat[0]*y.flat[6]],
                       [y.flat[1]**2,           y.flat[4]**2,           y.flat[5]**2,           y.flat[4]*y.flat[1],                        y.flat[4]*y.flat[6],                            y.flat[1]*y.flat[5]],
                       [y.flat[6]**2,           y.flat[5]**2,           y.flat[8]**2,           y.flat[5]*y.flat[6],                        y.flat[8]*y.flat[5],                            y.flat[8]*y.flat[6]],
                       [y.flat[0]*y.flat[1],    y.flat[4]*y.flat[1],    y.flat[5]*y.flat[6],    (y.flat[0]*y.flat[4]+y.flat[1]**2)/2,       (y.flat[1]*y.flat[5]+y.flat[4]*y.flat[6])/2,    (y.flat[0]*y.flat[5]+y.flat[1]*y.flat[6])/2],
                       [y.flat[1]*y.flat[6],    y.flat[4]*y.flat[5],    y.flat[8]*y.flat[5],    (y.flat[1]*y.flat[5]+y.flat[4]*y.flat[6])/2,(y.flat[8]*y.flat[4]+y.flat[5]**2)/2,           (y.flat[8]*y.flat[1]+y.flat[5]*y.flat[6])/2],
                       [y.flat[0]*y.flat[6],    y.flat[1]*y.flat[5],    y.flat[8]*y.flat[6],    (y.flat[0]*y.flat[5]+y.flat[1]*y.flat[6])/2,(y.flat[8]*y.flat[1]+y.flat[5]*y.flat[6])/2,    (y.flat[8]*y.flat[0]+y.flat[6]**2)/2]]) #check done
        L = alfa * Is - bta * Ib
        for i in range(3):
            L += (ydash[i] + gama[i]) * np.outer(eDir[:, i], eDir[:, i])
    return L
@jit(nopython=True)
def formULstiff(F,D,s,B):
    """%Updated Lagrangian material stiffness matrix
%--------------------------------------------------------------------------
% Author: William Coombs
% Date:   27/05/2015
% Description:
% Function to determine consistent material stiffness matrix based on an
% updated Lagrangian formulation of finite deformation mechanics.  See
% equations (25) and (26) of the following paper for full details:
% Charlton, T.J., Coombs, W.M. & Augarde, C.E. (2017). iGIMP: An implicit 
% generalised interpolation material point method for large deformations. 
% Computers and Structures 190: 108-125.
%
%--------------------------------------------------------------------------
% [A] = FORMULSTIFF(F,D,s,BeT)
%--------------------------------------------------------------------------
% Input(s):
% F  - deformation gradient (3,3)
% D  - small strain material stifness matrix (6,6)
% s  - Cauchy stress (6,1)
% B  - trial elastic left Cauchy-Green strain matrix (3,3)
%--------------------------------------------------------------------------
% Ouput(s);
% A   - consistent tangent stifness matrix (9,9)
%--------------------------------------------------------------------------
% See also:
% PARDERGEN  - partial derivative of a second order tensor
%--------------------------------------------------------------------------
"""
    t = np.array([0,1,2,3,3,4,4,5,5],dtype=np.uint8)
    J = np.linalg.det(F) #volume ratio
    bP, bV = np.linalg.eig(B)
    L = parDerGen(B, bV, bP, np.log(bP), 1. / bP)
    #s = s.flatten()
    S = np.array([  [s.flat[0],     0,          0,          s.flat[3],      0,          0,          0,          0,          s.flat[5]], #matrix form of sigma_{il}delta_{jk}
                    [0,             s.flat[1],  0,          0,              s.flat[3],  s.flat[4],  0,          0,          0        ],
                    [0,             0,          s.flat[2],  0,              0,          0,          s.flat[4],  s.flat[5],  0        ],
                    [0,             s.flat[3],  0,          0,              s.flat[0],  s.flat[5],  0,          0,          0        ],
                    [s.flat[3],     0,          0,          s.flat[1],      0,          0,          0,          0,          s.flat[4]],
                    [0,             0,          s.flat[4],  0,              0,          0,          s.flat[1],  s.flat[3],  0        ],
                    [0,             s.flat[4],  0,          0,              s.flat[5],  s.flat[2],  0,          0,          0        ],
                    [s.flat[5],     0,          0,          s.flat[4],      0,          0,          0,          0,          s.flat[2]],
                    [0,             0,          s.flat[5],  0,              0,          0,          s.flat[3],  s.flat[0],  0        ]])
    #B = B.ravel()
    T = np.array([  [2*B.flat[0],   0,              0,              2*B.flat[3],    0,              0,              2*B.flat[6],    0,              0], #matrix form of delta_{pk}b^e_{ql}+delta_{qk}b^e_{pl}
                    [0,             2*B.flat[4],    0,              0,              2*B.flat[1],    2*B.flat[7],    0,              0,              0],
                    [0,             0,              2*B.flat[8],    0,              0,              0,              2*B.flat[5],    2*B.flat[2],    0],
                    [B.flat[1],     B.flat[3],      0,              B.flat[4],      B.flat[0],      B.flat[6],      0,              0,              B.flat[7]],
                    [B.flat[1],     B.flat[3],      0,              B.flat[4],      B.flat[0],      B.flat[6],      0,              0,              B.flat[7]],
                    [0,             B.flat[5],      B.flat[7],      0,              B.flat[2],      B.flat[8],      B.flat[4],      B.flat[1],      0],
                    [0,             B.flat[5],      B.flat[7],      0,              B.flat[2],      B.flat[8],      B.flat[4],      B.flat[1],      0],
                    [B.flat[2],     0,              B.flat[6],      B.flat[5],      0,              0,              B.flat[3],      B.flat[0],      B.flat[8]],
                    [B.flat[2],     0,              B.flat[6],      B.flat[5],      0,              0,              B.flat[3],      B.flat[0],      B.flat[8]]])
    A =  D[t, :][:, t] @ L[t, :][:, t] @ T / (2 * J) - S
    return A
#@jit
def yieldFuncDerivatives(sig,rhoY):
    """%von Mises yield function derivatives
        %--------------------------------------------------------------------------
        % Author: William Coombs
        % Date:   16/05/2016
        % Description:
        % First and second derivatives of the von Mises yield function with respect
        % to stress.
        %
        %--------------------------------------------------------------------------
        % [df,ddf] = YIELDFUNCDERIVATIVES(sigma,rhoY)
        %--------------------------------------------------------------------------
        % Input(s):
        % sigma - Cauchy stress (6,1)
        % rhoY  - von Mises yield strength (1)
        %--------------------------------------------------------------------------
        % Ouput(s);
        % df    - derivative of the yield function wrt. sigma (6,1)
        % ddf   - second derivative of the yield function wrt. sigma (6,6)
        %--------------------------------------------------------------------------
        % See also:
        % 
        %--------------------------------------------------------------------------
"""
    bm1 = np.array([1, 1, 1, 0, 0, 0]).reshape(-1, 1)
    s = sig.reshape(-1, 1) - np.sum(sig[:3]) / 3 * bm1
    j2 = (np.dot(s.T, s) + np.dot(s[3:6].T, s[3:6])) / 2
    dj2=s
    dj2[3:6]=2*dj2[3:6]

    ddj2 = np.block([[np.eye(3) - np.ones((3, 3)) / 3, np.zeros((3, 3))],
                     [np.zeros((3, 3)), 2 * np.eye(3)]])
    df =dj2/(rhoY*np.sqrt(2*j2))
    ddf = 1 / rhoY * (ddj2 / np.sqrt(2 * j2) - (dj2 @ dj2.T) / (2 * j2) ** (3 / 2))
    return df,ddf
#@jit
def VMconst(epsEtr,mCst):
    """%von Mises linear elastic perfectly plastic constitutive model
%--------------------------------------------------------------------------
% Author: William Coombs
% Date:   16/05/2016
% Description:
% von Mises perfect plasticity constitutive model with an implicit backward
% Euler stress integration algorithm based on the following thesis:
%
% Coombs, W.M. (2011). Finite deformation of particulate geomaterials: 
% frictional and anisotropic Critical State elasto-plasticity. School of 
% Engineering and Computing Sciences. Durham University. PhD.
%
%--------------------------------------------------------------------------
% [Dalg,sigma,epsE] = VMCONST(epsEtr,mCst)
%--------------------------------------------------------------------------
% Input(s):
% epsEtr - trial elastic strain (6,1)
% mCst   - material constants 
%--------------------------------------------------------------------------
% Ouput(s);
% sig    - Cauchy stress (6,1)
% epsE   - elastic strain (6,1)
% Dalg   - algorithmic consistent tangent (6,6)
%--------------------------------------------------------------------------
% See also:
% YILEDFUNCDERIVATIVES - yield function 1st and 2nd derivatives
%--------------------------------------------------------------------------
"""
    E, v,rhoY=mCst[0:3]
    tol = 1e-9
    maxit = 5
    bm1 = np.array([1, 1, 1, 0, 0, 0]).reshape(-1, 1)
    Ce = np.block([[-np.ones((3, 3))*v+(1+v)*np.eye(3),np.zeros((3, 3))],
                   [np.zeros((3, 3)), 2*(1+v)*np.eye(3)]])/E
    De = E / ((1 + v) * (1 - 2 * v)) * (np.outer(bm1, bm1) * v + np.block([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), np.eye(3) / 2]]) * (1 - 2 * v))
    sig = De @ epsEtr
    s = sig.reshape(-1, 1) - np.sum(sig[:3]) / 3 * bm1
    j2 = (np.dot(s.T, s) + np.dot(s[3:6].T, s[3:6])) / 2
    f = np.sqrt(2 * j2)/rhoY - 1
    epsE = epsEtr
    Dalg = De
    if f>tol: #bug
        b = np.zeros((7,1))
        b[6] = f
        itnum,dgam = 0,0
        df,ddf = yieldFuncDerivatives(sig,rhoY)
        while (itnum < maxit) and ((np.linalg.norm(b[:6]) > tol) or (abs(b[6]) > tol)):
            A = np.block([[np.eye(6) + dgam * ddf @ De, df], [df.T @ De, 0]])
            dx = -np.linalg.inv(A) @ b
            epsE = epsE + dx[:6]
            dgam = dgam + dx[6]
            sig = De @ epsE
            s = sig-np.sum(sig[:3])/3*bm1
            j2 = (np.dot(s.T, s) + np.dot(s[3:6].T, s[3:6])) / 2
            df,ddf = yieldFuncDerivatives(sig,rhoY)
            b = np.concatenate([(epsE - epsEtr + dgam * df), (np.sqrt(2 * j2) / rhoY - 1)])
            itnum+=+1
        B = np.linalg.inv(np.block([[Ce + dgam * ddf, df], [df.T, 0]]))
        Dalg = B[:6,:] [:,:6]
    return Dalg,sig,epsE
#@jit
def Hooke3d(epsE,mCst):
    """%Linear elastic constitutive model
    %--------------------------------------------------------------------------
    % Author: William Coombs
    % Date:   29/01/2019
    % Description:
    % Small strain linear elastic constitutive model 
    %
    %--------------------------------------------------------------------------
    % [D,sig,epsE] = HOOKE3D(epsE,mCst)
    %--------------------------------------------------------------------------
    % Input(s):
    % epsE  - elastic strain (6,1)
    % mCst  - material constants 
    %--------------------------------------------------------------------------
    % Ouput(s);
    % D     - elastic stiffness matrix (6,6)
    % sig   - stress (6,1)
    % epsE  - elastic strain (6,1)
    %--------------------------------------------------------------------------
    % See also:
    % 
    %--------------------------------------------------------------------------
    """
    E=mCst[0]
    v=mCst[1]
    bm11 = np.array([   [1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]])
    D = E / ((1 + v) * (1 - 2 * v)) * (bm11 * v + np.diag([1, 1, 1, 0.5, 0.5, 0.5]) * (1 - 2 * v))
    sig = D @ epsE
    return D,sig,epsE
#@jit()
def detExtForce(nodes,nD,g,mpData):
    """
%Global external force determination  
%--------------------------------------------------------------------------
% Author: William Coombs
% Date:   23/01/2019
% Description:
% Function to determine the external forces at nodes based on body forces
% and point forces at material points.
%
%--------------------------------------------------------------------------
% [fbdy,mpData] = DETEXTFORCE(coord,etpl,g,eIN,mpData)
%--------------------------------------------------------------------------
% Input(s):
% nodes  - number of nodes (total in mesh)
% nD     - number of dimensions
% g      - gravity
% mpData - material point structured array. Function requires:
%           mpM   : material point mass
%           nIN   : nodes linked to the material point
%           Svp   : basis functions for the material point
%           fp    : point forces at material points
%--------------------------------------------------------------------------
% Ouput(s);
% fext   - external force vector (nodes*nD,1)
%--------------------------------------------------------------------------
% See also:
% 
%--------------------------------------------------------------------------
"""
    nmp = len(mpData)
    fext = np.zeros((nodes*nD))
    grav = np.zeros((nD,1))
    grav[nD-1] = -g
    for mp in range(nmp):
        nIN = mpData.nIN[mp]
        nn = len(nIN)
        Svp = mpData.Svp[mp].reshape(1,-1)
        fp = (mpData.mpM[mp]*grav + mpData.fp[mp])@Svp
        ed = np.tile((nIN.astype(np.int64)-1)*nD, (nD, 1)).ravel() + np.tile(np.arange(0, nD).reshape(-1, 1), (1, nn)).ravel()
        fext[ed] = fext[ed] + fp.ravel()
    return fext
#@jit()
def detFDoFs(mesh):
    """%Determine the free degrees of freedom on the background mesh
%--------------------------------------------------------------------------
% Author: William Coombs
% Date:   17/12/2018
% Description:
% Function to determine the free degrees of freedom of the background mesh
% based on the elements that contain material points and the displacement
% boundary conditions. 
%
%--------------------------------------------------------------------------
% [fd] = DETFDOFS(etpl,eInA,bc,nD,nDoF)
%--------------------------------------------------------------------------
% Input(s):
% mesh  - mesh structured array. Function requires: 
%           - etpl : element topology (nels,nen) 
%           - eInA : elements "active" in the analysis
%           - bc   : boundary conditions (*,2)
%--------------------------------------------------------------------------
% Ouput(s);
% fd    - free degrees of freedom on the background mesh (*,1)
%--------------------------------------------------------------------------
% See also:
%
%--------------------------------------------------------------------------
"""
    nodes,nD = mesh.coord.shape
    nDoF = nodes*nD
    incN = np.unique(mesh.etpl[mesh.eInA > 0, :]).astype(np.uint64)
    iN = len(incN)
    #t1 = np.ones((nD, 1)) * incN.reshape(1, -1) * nD
    #t2 = np.arange(nD-1, -1, -1).reshape(-1, 1) * np.ones((1, iN))
    incDoF = (np.ones((nD, 1),dtype=np.uint64) * incN.reshape(1, -1) * nD - np.arange(nD-1, -1, -1,dtype=np.uint64).reshape(-1, 1) * np.ones((1, iN),dtype=np.uint64)).ravel(order='F')
    fd = np.arange(1, nDoF+1,dtype=np.uint64)
    fd[mesh.bc[:, 0]] = 0
    fd = fd[incDoF-1]
    fd = fd[fd > 0]
    fd = fd-1
    return fd
#@TD
@jit(forceobj=True)
def linSolve(bc,Kt,oobf,NRit,fd):#check 
    #bc = bc-1 #fix
    nDoF = len(oobf)
    duvw = np.zeros((nDoF,1))
    drct = np.zeros((nDoF,1))
    if NRit>0:
        duvw[bc[:, 0]] = ((1 + np.sign(1 - NRit)) * bc[:, 1].astype(np.float64)).reshape(-1,1)  # apply non-zero boundary conditions 
        #ttt = Kt[fd, :][:, fd]
        duvw[fd] = spla.spsolve(Kt[fd, :][:, fd], oobf[fd].reshape(-1,1)  - Kt[fd, :][:, bc[:, 0]] @ duvw[bc[:, 0]]).reshape(-1,1)
        drct[bc[:, 0]] = Kt[bc[:, 0], :] @ duvw - oobf[bc[:, 0]].reshape(-1,1)
    return duvw,drct
@jit(forceobj=True)
def parDerGen(X,eV,eP,yP,ydash):
    """%Partial derivative of a second order tensor function
%--------------------------------------------------------------------------
% Author: William Coombs
% Date:   27/05/2015
% Description:
% Function to determine the partial derivative of a second order tensor
% function with respect to its arguement (X) based on the implementation 
% described by in the following paper:
%
% C. Miehe, Comparison of two algorithms for the computation of fourth-
% order isotropic tensor functions, Computers & Structures 66 (1998) 37-43.
%
% For example, in order to determine the derivative of log(X) with respect
% to X the inputs to the function should be:
%
% [L] = PARDERGEN(X,eV,eP,log(eP),1./eP)
%
% as the derivative of the log(x) is 1/x
%
% The symbols used in the code follow, as closely as possible, those used
% in the Miehe (1998) paper.  There are a number of different cases that
% have to be checked (all zero and repeated eigenvalues) in addition to the
% general case where there are no repeated eigenvalues.  
%
%--------------------------------------------------------------------------
% [L] = PARDERGEN(X,eV,eP,yP,ydash)
%--------------------------------------------------------------------------
% Input(s):
% X     - second order tensor in matrix format (3,3)
% eV    - eigenvectors of X (3,3)
% eP    - eigenvalues of X (1,3) 
% yP    - function applied to eP (1,3)
% ydash - derivative of the function applied to eP (1,3)
%--------------------------------------------------------------------------
% Ouput(s);
% L      - partial derivative of the second order tensor with respect to 
%          its arguement (6,6)
%--------------------------------------------------------------------------
% See also:
% 
%--------------------------------------------------------------------------"""
    tol=1e-9
    Is=np.diag([1, 1, 1, 0.5, 0.5, 0.5])
    if (np.abs(eP[0]) < tol and np.abs(eP[1]) < tol and np.abs(eP[2]) < tol):
        L = Is
    elif (np.abs(eP[0] - eP[1]) < tol and np.abs(eP[0] - eP[2]) < tol):
        L = ydash[0] * Is
    elif abs(eP[0] - eP[1]) < tol or abs(eP[1] - eP[2]) < tol or abs(eP[0] - eP[2]) < tol:
        
        if abs(eP[0] - eP[1])<tol:
            xa = eP[2]
            xc = eP[0]
            ya = yP[2]
            yc = yP[0]
            yda = ydash[2]
            ydc = ydash[0]
        elif abs(eP[1] - eP[2]) < tol:
            xa = eP[0]
            xc = eP[1]
            ya = yP[0]
            yc = yP[1]
            yda = ydash[0]
            ydc = ydash[1]
        else:
            xa = eP[1]
            xc = eP[0]
            ya = yP[1]
            yc = yP[0]
            yda = ydash[1]
            ydc = ydash[0]
        #X = X.ravel()
        x = X.flat[[0, 4, 8, 3, 5, 2]]
        s1 = (ya-yc)/(xa-xc)**2-ydc/(xa-xc)
        s2 = 2*xc*(ya-yc)/(xa-xc)**2-(xa+xc)/(xa-xc)*ydc
        s3 = 2*(ya-yc)/(xa-xc)**3-(yda+ydc)/(xa-xc)**2
        s4 = xc*s3
        s5 = xc**2*s3
        dX2dX = np.array([  [2*X.flat[0],    0,           0,           X[1],                     0,                        X.flat[2]            ],
                            [0,              2*X.flat[4], 0,           X[1],                     X.flat[5],                0               ],
                            [0,              0,           2*X.flat[8], 0,                        X.flat[5],                X.flat[2]            ],
                            [X.flat[1],      X.flat[1],   0,           (X.flat[0]+X.flat[4])/2,  X[2]/2,                   X.flat[5]/2          ],
                            [0,              X.flat[5],   X.flat[5],   X[2].flat/2,              (X.flat[4]+X.flat[8])/2,  X.flat[1]/2          ],
                            [X.flat[2],      0,           X.flat[2],   X[5].flat/2,              X.flat[1]/2,              (X.flat[0]+X.flat[8])/2   ]])
        bm1 = np.array([1, 1, 1, 0, 0, 0])
        bm11 = np.array([   [1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]])
        L = s1 * dX2dX - s2 * Is - s3 * np.outer(x, x) + s4 * (np.outer(x, bm1) + np.outer(bm1, x)) - s5 * bm11
    else:
        
        D = np.array([  (eP[0] - eP[1]) * (eP[0] - eP[2]),
                        (eP[1] - eP[0]) * (eP[1] - eP[2]),
                        (eP[2] - eP[0]) * (eP[2] - eP[1])])
        alfa = 0
        bta = 0
        gama = np.zeros((3, 1))
        eDir = np.zeros((6, 3))

        for i in range(3):
            alfa += yP[i] * eP[i] / D[i]
            bta += yP[i] / D[i] * np.linalg.det(X)
            for j in range(3):
                gama[i] += yP[j] * eP[j] / D[j] * (np.linalg.det(X) / eP[j] - eP[i]**2) * 1 / eP[i]**2
            esq = np.outer(eV[:, i], eV[:, i])
            eDir[:, i] = [esq[0, 0], esq[1, 1], esq[2, 2], esq[0, 1], esq[1, 2], esq[2, 0]]

        y = np.linalg.inv(X)
        
        Ib = np.array([[y.flat[0]**2,           y.flat[1]**2,           y.flat[6]**2,           y.flat[0]*y.flat[1],                        y.flat[1]*y.flat[6],                            y.flat[0]*y.flat[6]],
                       [y.flat[1]**2,           y.flat[4]**2,           y.flat[5]**2,           y.flat[4]*y.flat[1],                        y.flat[4]*y.flat[6],                            y.flat[1]*y.flat[5]],
                       [y.flat[6]**2,           y.flat[5]**2,           y.flat[8]**2,           y.flat[5]*y.flat[6],                        y.flat[8]*y.flat[5],                            y.flat[8]*y.flat[6]],
                       [y.flat[0]*y.flat[1],    y.flat[4]*y.flat[1],    y.flat[5]*y.flat[6],    (y.flat[0]*y.flat[4]+y.flat[1]**2)/2,       (y.flat[1]*y.flat[5]+y.flat[4]*y.flat[6])/2,    (y.flat[0]*y.flat[5]+y.flat[1]*y.flat[6])/2],
                       [y.flat[1]*y.flat[6],    y.flat[4]*y.flat[5],    y.flat[8]*y.flat[5],    (y.flat[1]*y.flat[5]+y.flat[4]*y.flat[6])/2,(y.flat[8]*y.flat[4]+y.flat[5]**2)/2,           (y.flat[8]*y.flat[1]+y.flat[5]*y.flat[6])/2],
                       [y.flat[0]*y.flat[6],    y.flat[1]*y.flat[5],    y.flat[8]*y.flat[6],    (y.flat[0]*y.flat[5]+y.flat[1]*y.flat[6])/2,(y.flat[8]*y.flat[1]+y.flat[5]*y.flat[6])/2,    (y.flat[8]*y.flat[0]+y.flat[6]**2)/2]]) #check done
        L = alfa * Is - bta * Ib
        for i in range(3):
            L += (ydash[i] + gama[i]) * np.outer(eDir[:, i], eDir[:, i])

    return L
@jit(forceobj=True)
def formULstiff(F,D,s,B):
    """%Updated Lagrangian material stiffness matrix
%--------------------------------------------------------------------------
% Author: William Coombs
% Date:   27/05/2015
% Description:
% Function to determine consistent material stiffness matrix based on an
% updated Lagrangian formulation of finite deformation mechanics.  See
% equations (25) and (26) of the following paper for full details:
% Charlton, T.J., Coombs, W.M. & Augarde, C.E. (2017). iGIMP: An implicit 
% generalised interpolation material point method for large deformations. 
% Computers and Structures 190: 108-125.
%
%--------------------------------------------------------------------------
% [A] = FORMULSTIFF(F,D,s,BeT)
%--------------------------------------------------------------------------
% Input(s):
% F  - deformation gradient (3,3)
% D  - small strain material stifness matrix (6,6)
% s  - Cauchy stress (6,1)
% B  - trial elastic left Cauchy-Green strain matrix (3,3)
%--------------------------------------------------------------------------
% Ouput(s);
% A   - consistent tangent stifness matrix (9,9)
%--------------------------------------------------------------------------
% See also:
% PARDERGEN  - partial derivative of a second order tensor
%--------------------------------------------------------------------------
"""
    t = np.array([0,1,2,3,3,4,4,5,5],dtype=np.uint8)
    J = np.linalg.det(F) #volume ratio
    bP, bV = np.linalg.eig(B)
    L = parDerGen(B, bV, bP, np.log(bP), 1. / bP)
    #s = s.flatten()
    S = np.array([  [s.flat[0],     0,          0,          s.flat[3],      0,          0,          0,          0,          s.flat[5]], #matrix form of sigma_{il}delta_{jk}
                    [0,             s.flat[1],  0,          0,              s.flat[3],  s.flat[4],  0,          0,          0        ],
                    [0,             0,          s.flat[2],  0,              0,          0,          s.flat[4],  s.flat[5],  0        ],
                    [0,             s.flat[3],  0,          0,              s.flat[0],  s.flat[5],  0,          0,          0        ],
                    [s.flat[3],     0,          0,          s.flat[1],      0,          0,          0,          0,          s.flat[4]],
                    [0,             0,          s.flat[4],  0,              0,          0,          s.flat[1],  s.flat[3],  0        ],
                    [0,             s.flat[4],  0,          0,              s.flat[5],  s.flat[2],  0,          0,          0        ],
                    [s.flat[5],     0,          0,          s.flat[4],      0,          0,          0,          0,          s.flat[2]],
                    [0,             0,          s.flat[5],  0,              0,          0,          s.flat[3],  s.flat[0],  0        ]])
    #B = B.ravel()
    T = np.array([  [2*B.flat[0],   0,              0,              2*B.flat[3],    0,              0,              2*B.flat[6],    0,              0], #matrix form of delta_{pk}b^e_{ql}+delta_{qk}b^e_{pl}
                    [0,             2*B.flat[4],    0,              0,              2*B.flat[1],    2*B.flat[7],    0,              0,              0],
                    [0,             0,              2*B.flat[8],    0,              0,              0,              2*B.flat[5],    2*B.flat[2],    0],
                    [B.flat[1],     B.flat[3],      0,              B.flat[4],      B.flat[0],      B.flat[6],      0,              0,              B.flat[7]],
                    [B.flat[1],     B.flat[3],      0,              B.flat[4],      B.flat[0],      B.flat[6],      0,              0,              B.flat[7]],
                    [0,             B.flat[5],      B.flat[7],      0,              B.flat[2],      B.flat[8],      B.flat[4],      B.flat[1],      0],
                    [0,             B.flat[5],      B.flat[7],      0,              B.flat[2],      B.flat[8],      B.flat[4],      B.flat[1],      0],
                    [B.flat[2],     0,              B.flat[6],      B.flat[5],      0,              0,              B.flat[3],      B.flat[0],      B.flat[8]],
                    [B.flat[2],     0,              B.flat[6],      B.flat[5],      0,              0,              B.flat[3],      B.flat[0],      B.flat[8]]])
    A =  D[t, :][:, t] @ L[t, :][:, t] @ T / (2 * J) - S
    return A
#@jit()
@TD
def detMPs(uvw,mpData): #check
    """
%Stiffness and internal force calculation for all material points
%--------------------------------------------------------------------------
% Author: William Coombs
% Date:   23/01/2019
% Description:
% Function to determine the stiffness contribution of a particle to the
% nodes that it influences based on a Updated Lagrangian finite deformation 
% formulation.  The function also returns the stresses at the particles and 
% the internal force contribution.  This function allows for elasto-
% plasticity at the material points.  The functionis applicable to 1, 2 and
% 3 dimensional problems without modification as well as different material 
% point methods and background meshes.   
% 
%--------------------------------------------------------------------------
% [fint,Kt,mpData] = DETMPS(uvw,mpData)
%--------------------------------------------------------------------------
% Input(s):
% uvw    - nodal displacements that influence the MP (nn*nD,1)
% mpData - material point structured array. The following fields are
%          required by the function:
%           - dSvp  : basis function derivatives (nD,nn)
%           - nIN   : background mesh nodes associated with the MP (1,nn)
%           - Fn    : previous deformation gradient (3,3) 
%           - epsEn : previous elastic logarithmic strain (6,1)
%           - mCst  : material constants
%           - vp    : material point volume (1)
%           - nSMe  : number stiffness matrix entries
% nD     - number of dimensions
%--------------------------------------------------------------------------
% Ouput(s);
% fint   - global internal force vector
% Kt     - global stiffness matrix
% mpData - material point structured array (see above).  The following
%          fields are updated by the function:
%           - F     : current deformation gradient (3,3)
%           - sig   : current Cauchy stress (6,1)
%           - epsE  : current elastic logarithmic strain (6,1)
%--------------------------------------------------------------------------
% See also:
% FORMULSTIFF      - updated Lagrangian material stiffness calculation
% HOOKE3D          - linear elastic constitutive model
% VMCONST          - von Mises elasto-plastic constitutive model
%--------------------------------------------------------------------------
"""
    nmp = len(mpData)
    #fint = np.zeros((uvw.shape))
    #npCnt = 0
    tnSMe = np.sum(mpData.nSMe)
    #krow = np.zeros(tnSMe,dtype=np.uint64)
    #kcol = np.zeros(tnSMe,dtype=np.uint64)
    #kval = np.zeros(tnSMe,dtype=np.float64)
    #ddF = np.zeros((3,3))
    nD = len(mpData.mpC[0])
    if nD ==1:
        fPos = 0
        aPos = 0
        sPos = 0
    elif nD ==2:
        fPos = np.array([1,5,4,2],dtype=np.uint8)-1
        aPos = np.array([1,2,4,5],dtype=np.uint8)-1
        sPos = np.array([1,2,4,4],dtype=np.uint8)-1
    else:
        fPos = np.array([1 ,5 ,9 ,4, 2 ,8 ,6, 3 ,7],dtype=np.uint8)-1
        #fPos = np.array([[0,0],[1,1],[2,2]
        #                 ,[1,0],[0,1],[2,1],
        #                 [1,2],[2,1],[2,0]],dtype=np.uint8)
        aPos = np.array([1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9],dtype=np.uint8)-1
        sPos = np.array([1 ,2 ,3 ,4 ,4 ,5 ,5 ,6 ,6],dtype=np.uint8)-1
    #@memory.cache
    fint,kval,krow,kcol,nDoF,Fs,sigs,epsEs = detMPs_loop(nD,nmp,
                                        tnSMe,mpData.nIN,
                                        mpData.dSvp,fPos,
                                        mpData.Fn,mpData.epsEn,
                                        mpData.cmType,mpData.mCst,
                                        mpData.vp,uvw,aPos,
                                        sPos)
    mpData.F    = Fs
    mpData.sig  = sigs
    mpData.epsE = epsEs
    Kt = coo_matrix((kval, ((krow-1), (kcol-1))), shape=(nDoF, nDoF)).tocsc()
    #mpData.F    = np.zeros((len(nmp),3,3))
    #mpData.sig  = np.zeros((len(nmp),6,1))
    #mpData.epsE = np.zeros((len(nmp),6,1))
    #for mp in range(nmp):
    #    #if mp == 2303 :
    #    #    print("")
    #    F,sig ,epsE,ed,krow_i,kcol_i,kval_i,fp= worker(nD,mpData.nIN[mp],mpData.dSvp[mp],
    #                                fPos,mpData.Fn[mp],mpData.epsEn[mp],
    #                                mpData.cmType[mp],mpData.mCst[mp],
    #                                mpData.vp[mp],uvw,aPos,sPos)
    #    mpData.F[mp] = F
    #    mpData.sig[mp] = sig
    #    mpData.epsE[mp] = epsE
    #    npDoF = len(ed) ** 2
    #    krow[npCnt:npCnt+npDoF], kcol[npCnt:npCnt+npDoF], kval[npCnt:npCnt+npDoF] = krow_i,kcol_i,kval_i
    #    npCnt += npDoF
    #    fint[ed-1] += fp
    #nDoF=len(uvw)
    #Kt = coo_matrix((kval, ((krow-1), (kcol-1))), shape=(nDoF, nDoF)).tocsc()
    return fint,Kt,mpData
#@jit()
def updateMPs(uvw,mpData):
    """%Material point update: stress, position and volume
%--------------------------------------------------------------------------
% Author: William Coombs
% Date:   29/01/2019
% Description:
% Function to update the material point positions and volumes (and domain 
% lengths for GIMPM).  The function also updates the previously converged 
% value of the deformation gradient and the logarithmic elastic strain at 
% each material point based on the converged value and calculates the total 
% displacement of each material point.  
%
% For the generalised interpolation material point method the domain
% lengths are updated according to the stretch tensor following the
% approach of:
% Charlton, T.J., Coombs, W.M. & Augarde, C.E. (2017). iGIMP: An implicit 
% generalised interpolation material point method for large deformations. 
% Computers and Structures 190: 108-125.
%
%--------------------------------------------------------------------------
% [mpData] = UPDATEMPS(uvw,mpData)
%--------------------------------------------------------------------------
% Input(s):
% uvw    - nodal displacements (nodes*nD,1)
% mpData - material point structured array.  The function requires:
%           - mpC : material point coordinates
%           - Svp : basis functions
%           - F   : deformation gradient
%           - lp0 : initial domain lenghts (GIMPM only)
%--------------------------------------------------------------------------
% Ouput(s);
% mpData - material point structured array.  The function modifies:
%           - mpC   : material point coordinates
%           - vp    : material point volume
%           - epsEn : converged elastic strain
%           - Fn    : converged deformation gradient
%           - u     : material point total displacement
%           - lp    : domain lengths (GIMPM only)
%--------------------------------------------------------------------------
% See also:
% 
%--------------------------------------------------------------------------
"""
    t = [0, 4, 8]  # stretch components for domain updating
    nmp = len(mpData)  # number of material points
    nD = len(mpData.mpC[0])  # number of dimensions

    for mp in range(nmp):
        nIN = mpData.nIN[mp]  # nodes associated with material point
        nn = len(nIN)  # number of nodes
        N = mpData.Svp[mp].reshape(1,-1)  # basis functions
        F = mpData.F[mp]  # deformation gradient
        
        ed = np.tile((nIN.astype(np.int64) -1)*nD, (nD, 1)).T+ np.tile(np.arange(1, nD+1), (nn, 1)) # nodal degrees of freedom
        mpU =  N @ uvw.flat[ed-1]  # material point displacement 
        mpData.mpC[mp] = mpData.mpC[mp] + mpU  # update material point coordinates
        mpData.vp[mp] = np.linalg.det(F) * mpData.vp0[mp]  # update material point volumes
        mpData.epsEn[mp] = mpData.epsE[mp]  # update material point elastic strains
        mpData.Fn[mp] = mpData.F[mp]  # update material point deformation gradients
        mpData.u[mp] += mpU.reshape(-1,1) # update material point displacements
        
        if mpData.mpType[mp] == 2:  # GIMPM only (update domain lengths)
            D, V = np.linalg.eig(np.matmul(F.T, F))  # eigen values and vectors F'F
            U = np.matmul(np.matmul(V,np.diag(np.sqrt(D))), V.T)  # material stretch matrixnp.diag(
            mpData.lp[mp] = mpData.lp0[mp] * U.flat[t[:nD]]  # update domain lengths
    return mpData
@jit(forceobj=True)
def worker(nD,nIN,dNx,fPos,Fn,epsEn,cmType,mCst,vp,uvw,aPos,sPos):
    nn = dNx.shape[1]
    ed = (np.tile((nIN.astype(np.int64)-1)*nD, (nD, 1)) + np.tile(np.arange(1, nD+1).reshape(-1, 1), (1, nn))).flatten(order="F")
    #ed = (np.tile((nIN.astype(np.int64)-1)*nD, (nD, 1)) + np.tile(np.arange(1, nD+1).reshape(-1, 1), (1, nn))).reshape(1,-1)
    if nD ==1:G = dNx
    elif nD == 2:
        G = np.zeros((4,nD*nn),dtype=np.float64)
        G[np.ix_([0, 2], range(0, nD * nn, nD))] = dNx
        G[np.ix_([3, 1], range(1, nD * nn, nD))] = dNx
    else:
        G = np.zeros((9,nD*nn),dtype=np.float64)
        G[np.ix_([0, 3, 8], range(0, nD * nn, nD))] = dNx
        G[np.ix_([4, 1, 5], range(1, nD * nn, nD))] = dNx
        G[np.ix_([7, 6, 2], range(2, nD * nn, nD))] = dNx
    ddF = np.zeros((3,3),dtype=np.float64)
    ddF.flat[fPos] = (G @ uvw[ed-1])
    dF = np.eye(3,dtype=np.float64) +ddF.T #bug 不知道为什么ddf是转置的结果
    F = dF @ Fn
    epsEn = np.zeros_like(epsEn)+epsEn
    epsEn.flat[[3,4,5]] *= 0.5
    epsEn = epsEn.flat[[[0,3,5],[3,1,4],[5,4,2]]]
    
    D, V = np.linalg.eig(epsEn)
    BeT = dF @ (V @ np.diag(np.exp(2 * D)) @ V.T) @ dF.T
    D, V = np.linalg.eig(BeT)
    epsEtr = 0.5 * V @ np.diag(np.log(D)) @ V.T
    epsEtrPos = np.array([[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]])
    epsEtr = np.diag([1, 1, 1, 2, 2, 2]) @ epsEtr[epsEtrPos[:,0],epsEtrPos[:,1]]
    epsEtr = epsEtr.reshape(-1,1)
    if cmType==1:
        D,Ksig,epsE=Hooke3d(epsEtr,mCst)
    elif cmType == 2:
        D,Ksig,epsE=VMconst(epsEtr,mCst)
    sig = Ksig / np.linalg.det(F)
    A   = formULstiff(F,D,sig,BeT)
    iF = np.linalg.inv(dF)#.ravel(order="F")
    dXdx = np.array([        [iF.flat[0], 0, 0, iF.flat[1], 0, 0, 0, 0, iF.flat[2]],
                             [0, iF.flat[4], 0, 0, iF.flat[3], iF.flat[5], 0, 0, 0],
                             [0, 0, iF.flat[8], 0, 0, 0, iF.flat[7], iF.flat[6], 0],
                             [iF.flat[3], 0, 0, iF.flat[4], 0, 0, 0, 0, iF.flat[5]],
                             [0, iF.flat[1], 0, 0, iF.flat[0], iF.flat[2], 0, 0, 0],
                             [0, iF.flat[7], 0, 0, iF.flat[6], iF.flat[8], 0, 0, 0],
                             [0, 0, iF.flat[5], 0, 0, 0, iF.flat[4], iF.flat[3], 0],
                             [0, 0, iF.flat[2], 0, 0, 0, iF.flat[1], iF.flat[0], 0],
                             [iF.flat[6], 0, 0, iF.flat[7], 0, 0, 0, 0, iF.flat[8]]]).T
    G = np.matmul(dXdx[aPos, :][:, aPos], G)
    kp = vp * np.linalg.det(dF) *((G.T@A[aPos, :][:, aPos])@ G)
    fp = vp * np.linalg.det(dF) * (G.T@ sig[sPos]) 
    
    nnDoF = len(ed)
    krow_i = np.tile(ed, nnDoF).flatten(order="F")
    kcol_i = np.repeat(ed, nnDoF).flatten()
    kval_i = kp.flatten(order="F")
    return F,sig.reshape(-1,1),epsE.reshape(-1,1),ed,krow_i,kcol_i,kval_i,fp.reshape(-1, 1)
#@jit()
def detMPs_loop(nD,nmp,tnSMe,nINs,dSvps,fPos,Fns,epsEns,cmTypes,mCsts,vps,uvw,aPos,sPos):
    fint = np.zeros((uvw.shape))
    Fs    = np.zeros((nmp,3,3))
    sigs  = np.zeros((nmp,6,1))
    epsEs = np.zeros((nmp,6,1))
    krow = np.zeros(tnSMe,dtype=np.uint64)
    kcol = np.zeros(tnSMe,dtype=np.uint64)
    kval = np.zeros(tnSMe,dtype=np.float64)
    npCnt = 0
    for mp in range(nmp):
        F,sig ,epsE,ed,krow_i,kcol_i,kval_i,fp= worker(nD,nINs[mp],dSvps[mp],
                                    fPos,Fns[mp],epsEns[mp],
                                    cmTypes[mp],mCsts[mp],
                                    vps[mp],uvw,aPos,sPos)
        Fs[mp] = F
        sigs[mp] = sig
        epsEs[mp] = epsE
        npDoF = len(ed) ** 2
        krow[npCnt:npCnt+npDoF], kcol[npCnt:npCnt+npDoF], kval[npCnt:npCnt+npDoF] = krow_i,kcol_i,kval_i
        npCnt += npDoF
        fint[ed-1] += fp
    nDoF=len(uvw)
    #Kt = coo_matrix((kval, ((krow-1), (kcol-1))), shape=(nDoF, nDoF)).tocsc()
    return fint,kval,krow,kcol,nDoF,Fs,sigs,epsEs
