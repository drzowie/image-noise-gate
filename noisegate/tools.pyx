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
     source : numpy array with 1, 2, 3, or 4 axes
         This is the data to shred.
     size: tuple or list of describing the size of each cutout, in descending
         (...,Y,X) axis order (since it is essentially an index, not a vector).
         
     step: tuple or list of describing the step between cutouts, in descending
         (...,Y,X) axis order (since it is esentially an index, not a vector).

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
    #
    # About the only real optimizations here are:
    #  (1) compilation of the hotspot variables through cdef
    #  (2) ordering of the loop nesting to reduce L2/L3 cache violations
    
    # Detect dimensionality
    N = len(source.shape)
    
    # "Thread" if a scalar is passed in for either size or step
    if( len(size)==1 ):
        size = size + np.zeros(N)
    if( len(step)==1 ):
        step = step + np.zeros(N)
        
    # Make sure everything shook out to the right dimensionality
    if(len(size) != N  or  len(step) != N):
        raise(ValueError("noisegate.tools.shred: size and step must match the input array"))

    # Declare compiled C-like variables to make loops run faster
    cdef long chunk_xi, chunk_yi, chunk_zi, chunk_wi
    cdef long xstep, ystep, zstep, wstep
    cdef long xsize, ysize, zsize, wsize
    cdef long chunk_x0, chunk_y0, chunk_z0, chunk_w0
    cdef long chunk_x_count, chunk_y_count, chunk_z_count, chunk_w_count
        
    if(N==1):
        xsize = size[0]
        xstep = step[0]
        chunk_x_count = np.floor((source.shape[0]-xsize+1)/xstep).astype(int)
        
        output = np.empty([chunk_x_count, xsize],dtype=float,order='C')

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
                              zsize,         ysize,         xsize
                             ],
                          dtype=float,   order='C'
                          )
        chunk_z0 = 0
        for chunk_zi in range(chunk_z_count):
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
    
    elif(N==4):
        xsize = size[3]
        xstep = step[3]
        ysize = size[2]
        ystep = step[2]
        zsize = size[1]
        zstep = step[1]
        wsize = size[0]
        wstep = step[0]
        
        chunk_w_count = np.floor((source.shape[0]-wsize+1)/wstep).astype(int)
        chunk_z_count = np.floor((source.shape[1]-zsize+1)/zstep).astype(int)
        chunk_y_count = np.floor((source.shape[2]-ysize+1)/ystep).astype(int)
        chunk_x_count = np.floor((source.shape[3]-xsize+1)/xstep).astype(int)
    
        output = np.empty(  [ chunk_w_count, chunk_z_count, chunk_y_count, chunk_x_count,
                              wsize,         zsize,         ysize,         xsize
                             ],
                          dtype=float,   order='C'
                          )
        chunk_w0 = 0
        for chunk_wi in range(chunk_w_count):
            chunk_z0 = 0
            for chunk_zi in range(chunk_z_count):
                chunk_y0 = 0
                for chunk_yi in range(chunk_y_count):
                    chunk_x0 = 0                
                    for chunk_xi in range(chunk_x_count):
                        output[ chunk_wi, chunk_zi, chunk_yi, chunk_xi, :, :, : ] = \
                            source[ chunk_w0:chunk_w0+wsize,
                                    chunk_z0:chunk_z0+zsize,
                                    chunk_y0:chunk_y0+ysize,
                                    chunk_x0:chunk_x0+xsize
                                    ]
                        chunk_x0 += xstep
                    chunk_y0 += ystep
                chunk_z0 += zstep
            chunk_w0 += wstep
        return output

    else:
        raise(AssertionError("noisegate.tools.shred: N must be 1, 2, 3, or 4"))
            

   
            
            

    