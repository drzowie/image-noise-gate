# -*- coding: utf-8 -*-
"""
Tools for the image-noise-gate functionality
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, sqrt, floor, ceil, round, exp, fabs
import sys
import copy

def shred(
        source,
        size,
        step
        ):
    '''
    shred - cut an N-dimensional numpy array into a collection of smaller slices, 
    with programmable size and overlap.  
    
    If size and/or step are array-like objects then they must have exactly N
    elements.  If either one is a scalar, it is copied N times.
    
    Limitation
    ----------
     In the current implementation, N must be between 1 and 4.

    Parameters
    ----------
     source : numpy NDArray 
         This is the data to shred.
     size: tuple or list of describing the size of each cutout
     step: tuple or list of describing the step between cutouts

    Returns
    -------
    A 2N-dimensional NDArray with sizes (...,Y,X,...,y,x) (capitals indexing 
    which chunk, lowercase indexing within the chunks)

    '''
    
    #
    # CED - separate cases because they'll run faster, I think, than a 
    # single general-purpose case.  But the API is unified in case I (or you)
    # later think of a general-purpose way of doing this that is more elegant.
    #
    N = len(source.shape)
    if( len(size)==1 ):
        size = size + np.zeros(N)
    if( len(step)==1 ):
        step = step + np.zeros(N)
        
    if(len(size) != N  or  len(step) != N):
        raise(ValueError("noisegate.tools.shred: size and step must match the input array"))

    cdef long chunk_xi, chunk_yi, chunk_zi, chunk_wi
    cdef long xstep, ystep, zstep, wstep
    cdef long xsize, ysize, zsize, wsize
    cdef long chunk_x0, chunk_y0, chunk_z0, chunk_w0
        
    if(N==1):
        xsize = size[0]
        xstep = step[0]
        chunk_x_count = np.floor((source.shape[0]-xsize+1)/xstep).astype(int)
        
        output = np.empty([chunk_x_count, size[0]],dtype=float,order='C')

        chunk_x0 = 0
        for chunk_xi in range(chunk_x_count):
            output[ chunk_xi, : ] = \
                source[ chunk_x0:chunk_x0+xsize ]
            chunk_x0 += xstep
        return output
 
    elif(N==2):
        xsize = size[1]
        xstep = step[1]
        ysize = size[0]
        ystep = step[0]

        chunk_y_count = np.floor((source.shape[0]-ysize+1)/ystep).astype(int)
        chunk_x_count = np.floor((source.shape[1]-xsize+1)/xstep).astype(int)

        output = np.empty(  [ chunk_y_count, chunk_x_count,
                              ysize,         xsize
                             ],
                          dtype=float,   order='C'
                          )

        chunk_y0 = 0
        for chunk_yi in range(chunk_y_count):
            chunk_x0 = 0                
            for chunk_xi in range(chunk_x_count):
                output[ chunk_yi, chunk_xi, :, : ] = \
                    source[ chunk_y0:chunk_y0+ysize,
                            chunk_x0:chunk_x0+xsize
                        ]
                chunk_x0 += xstep
            chunk_y0 += ystep
        return output
    
    elif(N==3):
        xsize = size[2]
        xstep = step[2]
        ysize = size[1]
        ystep = step[1]
        zsize = size[0]
        zstep = step[0]
        
        chunk_z_count = np.floor((source.shape[0]-zsize+1)/zstep).astype(int)
        chunk_y_count = np.floor((source.shape[1]-ysize+1)/ystep).astype(int)
        chunk_x_count = np.floor((source.shape[2]-xsize+1)/xstep).astype(int)
    
        output = np.empty(  [ chunk_z_count, chunk_y_count, chunk_x_count,
                              zsize,         ysize,          xsize
                             ],
                          dtype=float,   order='C'
                          )
        chunk_z0 = 0
        for chunk_z in range(chunk_z_count):
            chunk_y0 = 0
            for chunk_yi in range(chunk_y_count):
                chunk_x0 = 0                
                for chunk_xi in range(chunk_x_count):
                    output[ chunk_zi, chunk_yi, chunk_xi, :, :, : ] = \
                        source[ chunk_z0:chunk_z0+zsize,
                                chunk_y0:chunk_y0+ysize,
                                chunk_x0:chunk_x0+xsize
                        ]
                    chunk_x0 += xstep
                chunk_y0 += ystep
            chunk_z0 += zstep
        return output
 
    else:
        raise(AssertionError("noisegate.tools.shred: N must be 1, 2, or 3"))
            

   
            
            

    