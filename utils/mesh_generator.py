import numpy as np
def formCoord2D(nelsx,nelsy,lx,ly):
    """
    #Two dimensional finite element grid generation
    #--------------------------------------------------------------------------
    # Author: William Coombs
    # Date:   06/05/2015
    # Description:
    # Function to generate a 2D finite element grid of linear quadrilateral 
    # elements.
    #
    #--------------------------------------------------------------------------
    # [etpl,coord] = FORMCOORD2D(nelsx,nelsy,lx,ly)
    #--------------------------------------------------------------------------
    # Input(s):
    # nelsx - number of elements in the x direction
    # nelsy - number of elements in the y direction
    # lx    - length in the x direction
    # ly    - length in the y direction
    #--------------------------------------------------------------------------
    # Ouput(s);
    # etpl  - element topology
    # coord - nodal coordinates
    #--------------------------------------------------------------------------
    # See also:
    #
    #--------------------------------------------------------------------------
    """
    nels  = nelsx*nelsy         #number of elements
    nodes = (nelsx+1)*(nelsy+1) # number of nodes
    # node generation
    coord = np.zeros((nodes, 2),dtype=np.float64)  # zero coordinates
    node = 0  # zero node counter
    
    y = ly * np.arange(nelsy+1) / nelsy
    x = lx * np.arange(nelsx+1) / nelsx
    node = np.arange(nodes)
    coord[:,0] = np.tile(x,nelsy+1)
    coord[:,1] = np.repeat(y,nelsx+1)


    # element generation
    etpl = np.zeros((nels, 4),dtype=np.int32)  # zero element topology
    
    
    nely_range = np.arange(1, nelsy + 1)
    nelx_range = np.arange(1, nelsx + 1)
    etpl[:, 0] = np.repeat((nely_range-1)*(nelsx+1),nelsx).flatten()+np.tile(nelx_range,nelsy).flatten()
    etpl[:, 1] = etpl[:, 0] + (nelsx+1)
    etpl[:, 2] = etpl[:, 1] + 1
    etpl[:, 3] = etpl[:, 0] + 1
    return etpl,coord

def detMpPos(mp,nD):
    """
    Material point local positions for point generation
    #--------------------------------------------------------------------------
    # Author: William Coombs
    # Date:   29/01/2019
    # Description:
    # Function to return the local positions of the material points for initial
    # set up of a problem.  The material points are evenly spaced through the
    # elements that can be regular line, quadrilateral and hexahedral elements
    # in 1D, 2D and 3D. 
    #
    #--------------------------------------------------------------------------
    # [mpPos] = DETMPPOS(mp,nD)
    #--------------------------------------------------------------------------
    # Input(s):
    # mp    - number of material points in each direction
    # nD    - number of dimensions
    #--------------------------------------------------------------------------
    # Ouput(s);
    # mpPos - local material point positions (nmp,nD)
    #--------------------------------------------------------------------------
    # See also:
    #
    #--------------------------------------------------------------------------
    """
    nmp=mp**nD
    mpPos = np.zeros((nmp,nD),dtype=np.float64)
    a = 2/mp 
    b = np.arange(a/2, 2, a) - 1
    if nD == 1:
        mpPos = np.reshape(b,(nmp,1))
    elif nD==2 :
        #mpPos((i-1)*mp+j,1)=b(i);
        #mpPos((i-1)*mp+j,2)=b(j);
        mpPos[:,0] = np.repeat(b,mp)
        mpPos[:,1] = np.tile(b,mp)
    else: #debug this
        #mpPos((i-1)*mp+j,1)=b(i);
        #mpPos((i-1)*mp+j,2)=b(j);
        #mpPos((i-1)*mp+j,3)=b(k);
        mpPos[:,0] = np.repeat(b,mp**2)
        mpPos[:,1] = np.tile(np.repeat(b,mp),mp)
        mpPos[:,2] = np.tile(b,mp**2)
    return mpPos

def shapefunc(nen,GpLoc,nD):
    """
    #Finite element basis functions
    #--------------------------------------------------------------------------
    # Author: William Coombs
    # Date:   29/01/2019
    # Description:
    # Function to provide finite element shapefunctions in 1D, 2D and 3D.  The
    # function includes the following elements:
    # nen = 8, nD = 3 : tri-linear eight noded hexahedral
    # nen = 4, nD = 2 : bi-linear four noded quadrilateral
    # nen = 2, nD = 1 : linear two noded line
    #
    # The function is vectorised so will return the basis functions for any
    # number of points.
    #
    #--------------------------------------------------------------------------
    # [N] = SHAPEFUNC(nen,GpLoc,nD)
    #--------------------------------------------------------------------------
    # Input(s):
    # nen    - number of nodes associated with the element
    # GpLoc  - local position within the finite element (n,nD)
    # nD     - number of dimensions
    #--------------------------------------------------------------------------
    # Ouput(s);
    # N      - shape function matrix (n,nen)
    #--------------------------------------------------------------------------
    # See also:
    # 
    #--------------------------------------------------------------------------

    """
    n = GpLoc.shape[0]
    N = np.zeros((n,nen),dtype=np.float64)
    if nD == 3:
        xsi = GpLoc[:,0]
        eta = GpLoc[:,1]
        zet = GpLoc[:,2]
        if nen == 8 :
            N[:,0] = (1-xsi)*(1-eta)*(1-zet)/8#0.125*(1-xsi)*(1-eta)*(1-zet)
            N[:,1] = (1-xsi)*(1-eta)*(1+zet)/8#0.125*(1+xsi)*(1-eta)*(1-zet)
            N[:,2] = (1+xsi)*(1-eta)*(1+zet)/8#0.125*(1+xsi)*(1eta)*(1-zet)
            N[:,3] = (1+xsi)*(1-eta)*(1-zet)/8#0.125*(1-xsi)*(1+eta)*(1-zet)
            N[:,4] = (1-xsi)*(1+eta)*(1-zet)/8#0.125*(1-xsi)*(1-eta)*(1+zet)
            N[:,5] = (1-xsi)*(1+eta)*(1+zet)/8#0.125*(1+xsi)*(1-eta)*(1+zet)
            N[:,6] = (1+xsi)*(1+eta)*(1+zet)/8#0.125*(1+xsi)*(1+eta)*(1+zet)
            N[:,7] = (1+xsi)*(1+eta)*(1-zet)/8#0.125*(1-xsi)*(1+eta)*(1+zet)
    elif nD == 2:
        xsi = GpLoc[:,0]
        eta = GpLoc[:,1]
        if nen == 4 :
            N[:,0] = 0.25*(1-xsi)*(1-eta)
            N[:,1] = 0.25*(1-xsi)*(1+eta)
            N[:,2] = 0.25*(1+xsi)*(1+eta)
            N[:,3] = 0.25*(1+xsi)*(1-eta)
    else:
        xsi = GpLoc[:,0]
        if nen == 2 :
            N[:,0] = 0.5*(1-xsi)
            N[:,1] = 0.5*(1+xsi)
    return N
if "__main__" == __name__:
    if 0:
        etpl,coord = formCoord2D(16,8,16,8)
        print(etpl)
        print(coord)
        print(etpl.shape)
    mpPos = detMpPos(6,2)
    #print(mpPos)
    #print(mpPos.shape)
    N = shapefunc(4,mpPos,2)
    print(N)