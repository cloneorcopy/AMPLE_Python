import numpy as np 
def makeVtkMP(mpC,sig,uvw,mpFileName):
    nmp, nD = mpC.shape  # number of material points and dimensions
    with open(mpFileName, 'wt') as fid :
        fid.write('# vtk DataFile Version 2.0\n')
        fid.write('MATLAB generated vtk file, WMC\n')
        fid.write('ASCII\n')
        fid.write('DATASET UNSTRUCTURED_GRID\n')
        fid.write('POINTS %i double\n' % nmp)
        # position output
        if nD < 3:
            mpC = np.hstack((mpC, np.zeros((nmp, 3 - nD))))
        np.savetxt(fid, mpC, fmt='%f %f %f', delimiter=' ')
        fid.write('\n')

        fid.write('POINT_DATA %i\n' % nmp)

        # stress output
        fid.write('SCALARS sigma_xx FLOAT %i\n' % 1)
        fid.write('LOOKUP_TABLE default\n')
        np.savetxt(fid, sig[:, 0], fmt='%f')
        fid.write('\n')

        fid.write('SCALARS sigma_yy FLOAT %i\n' % 1)
        fid.write('LOOKUP_TABLE default\n')
        np.savetxt(fid, sig[:, 1], fmt='%f')
        fid.write('\n')

        fid.write('SCALARS sigma_zz FLOAT %i\n' % 1)
        fid.write('LOOKUP_TABLE default\n')
        np.savetxt(fid, sig[:, 2], fmt='%f')
        fid.write('\n')

        fid.write('SCALARS sigma_xy FLOAT %i\n' % 1)
        fid.write('LOOKUP_TABLE default\n')
        np.savetxt(fid, sig[:, 3], fmt='%f')
        fid.write('\n')

        fid.write('SCALARS sigma_yz FLOAT %i\n' % 1)
        fid.write('LOOKUP_TABLE default\n')
        np.savetxt(fid, sig[:, 4], fmt='%f')
        fid.write('\n')

        fid.write('SCALARS sigma_zx FLOAT %i\n' % 1)
        fid.write('LOOKUP_TABLE default\n')
        np.savetxt(fid, sig[:, 5], fmt='%f')
        fid.write('\n')
        if nD == 3:
            fid.write('SCALARS u_x FLOAT %i\n' % 1)
            fid.write('LOOKUP_TABLE default\n')
            np.savetxt(fid, uvw[:, 0], fmt='%f')
            fid.write('\n')

            fid.write('SCALARS u_y FLOAT %i\n' % 1)
            fid.write('LOOKUP_TABLE default\n')
            np.savetxt(fid, uvw[:, 1], fmt='%f')
            fid.write('\n')

            fid.write('SCALARS u_z FLOAT %i\n' % 1)
            fid.write('LOOKUP_TABLE default\n')
            np.savetxt(fid, uvw[:, 2], fmt='%f')
            fid.write('\n')
        elif nD == 2:
            fid.write('SCALARS u_x FLOAT %i\n' % 1)
            fid.write('LOOKUP_TABLE default\n')
            np.savetxt(fid, uvw[:, 0], fmt='%f')
            fid.write('\n')

            fid.write('SCALARS u_y FLOAT %i\n' % 1)
            fid.write('LOOKUP_TABLE default\n')
            np.savetxt(fid, uvw[:, 1], fmt='%f')
            fid.write('\n')
        elif nD == 1:
            fid.write('SCALARS u_x FLOAT %i\n' % 1)
            fid.write('LOOKUP_TABLE default\n')
            np.savetxt(fid, uvw, fmt='%f')
            fid.write('\n')
def makeVtk(coord, etpl, uvw, meshName):

    nodes, nD = coord.shape
    nels, nen = etpl.shape

    # FEM etpl to VTK format
    if nD == 3:
        if nen == 20:
            tvtk = [1, 7, 19, 13, 3, 5, 17, 15, 8, 12, 20, 9, 4, 11, 16, 10, 2, 6, 18, 14]
            elemId = 25
            elemFormat = '%i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i\n'
        elif nen == 8:
            tvtk = [1, 4, 8, 5, 2, 3, 7, 6]
            elemId = 12
            elemFormat = '%i %i %i %i %i %i %i %i %i\n'
        elif nen == 10:
            tvtk = [1, 2, 3, 4, 5, 6, 7, 8, 10, 9]
            elemId = 24
            elemFormat = '%i %i %i %i %i %i %i %i %i %i %i\n'
        elif nen == 4:
            tvtk = [1, 3, 2, 4]
            elemId = 10
            elemFormat = '%i %i %i %i %i\n'
        elif nen == 9:
            tvtk = [3, 1, 7, 5, 2, 8, 6, 4, 9]
            elemId = 10
            elemFormat = '%i %i %i %i %i %i %i %i %i %i\n'
    elif nD == 2:
        if nen == 3:
            tvtk = [1, 3, 2]
            elemId = 5
            elemFormat = '%i %i %i %i\n'
        elif nen == 4:
            tvtk = [1, 4, 2, 3]
            elemId = 8
            elemFormat = '%i %i %i %i %i\n'
        elif nen == 8:
            tvtk = [1, 7, 5, 3, 8, 6, 4, 2]
            elemId = 23
            elemFormat = '%i %i %i %i %i %i %i %i %i\n'
    tvtk = np.array(tvtk) - 1
    with open(meshName, 'wt') as fid :
        # Generation of vtk file
        fid.write('# vtk DataFile Version 2.0\n')
        fid.write('MATLAB generated vtk file, WMC\n')
        fid.write('ASCII\n')
        fid.write('DATASET UNSTRUCTURED_GRID\n')
        fid.write('POINTS %i double\n' % nodes)

        # nodal coordinates
        if nD < 3:
            coord = np.hstack((coord, np.zeros((nodes, 3 - nD))))
        np.savetxt(fid, coord, fmt='%f %f %f', delimiter=' ')
        fid.write('\n')

        # element topology
        fid.write('CELLS %i %i\n' % (nels, (nen + 1) * nels))
        etplOutput = np.hstack((nen * np.ones((nels, 1)), etpl[:, tvtk] - 1))
        np.savetxt(fid, etplOutput, fmt=elemFormat, delimiter=' ')
        fid.write('\n')

        # element types
        fid.write('CELL_TYPES %i\n' % nels)
        np.savetxt(fid, elemId * np.ones((nels, 1)), fmt='%i')
        fid.write('\n')

        # displacement output
        fid.write('POINT_DATA %i\n' % nodes)
        if nD == 3:
            fid.write('SCALARS u_x FLOAT %i\n' % 1)
            fid.write('LOOKUP_TABLE default\n')
            np.savetxt(fid, uvw[::nD], fmt='%f')
            fid.write('\n')

            fid.write('SCALARS u_y FLOAT %i\n' % 1)
            fid.write('LOOKUP_TABLE default\n')
            np.savetxt(fid, uvw[1::nD], fmt='%f')
            fid.write('\n')

            fid.write('SCALARS u_z FLOAT %i\n' % 1)
            fid.write('LOOKUP_TABLE default\n')
            np.savetxt(fid, uvw[2::nD], fmt='%f')
            fid.write('\n')
        elif nD == 2:
            fid.write('SCALARS u_x FLOAT %i\n' % 1)
            fid.write('LOOKUP_TABLE default\n')
            np.savetxt(fid, uvw[::nD], fmt='%f')
            fid.write('\n')

            fid.write('SCALARS u_y FLOAT %i\n' % 1)
            fid.write('LOOKUP_TABLE default\n')
            np.savetxt(fid, uvw[1::nD], fmt='%f')
            fid.write('\n')
        elif nD == 1:
            fid.write('SCALARS u_x FLOAT %i\n' % 1)
            fid.write('LOOKUP_TABLE default\n')
            np.savetxt(fid, uvw, fmt='%f')
            fid.write('\n')
    
    