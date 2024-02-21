

import numpy as np
#from joblib import Memory
#import CONFIG 
#import os  
#os.makedirs(os.path.join(CONFIG.CACHE_DIR,'constitutive'),exist_ok=True)
#memory = Memory(CONFIG.CACHE_DIR, verbose=0)
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

#@memory.cache
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
        while (itnum < maxit) and ((np.linalg.norm(b[:6]) > tol) or (np.abs(b[6]) > tol)):
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

#@memory.cache
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