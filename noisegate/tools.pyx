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

def shred2d(
        source,
        size,
        step
        ):
    '''
    shred2d - cut a 2D numpy array into a collection of smaller slices, with 
    programmable size and overlap.  This is the Python interface to shred2d_cy,
    which has more rigid calling convention.

    Parameters
    ----------
     source : 2-dimensional NDArray 
         This is the data to shred.
     size: tuple or list of 2 integers describing the size of each cutout
     step: tuple or list of 2 integers describing the step between cutouts

    Returns
    -------
    A 4-dimensional NDArray containing (Y,X,y,x) (capitals indexing which 
    chunk, lowercase indexing within the chunks)

    '''
    
    cdef long source_xsiz = source.shape[1]
    cdef long source_ysiz = source.shape[0]
    cdef long chunk_x,chunk_y, x,y;
    cdef long chunk_xi, chunk_yi
    
    cdef long xsiz = size[1]
    cdef long ysiz = size[0]
    cdef long xstep = step[1]
    cdef long ystep = step[0]
    
    cdef long chunk_y_count = np.floor((source_ysiz-ysiz+1)/ystep)
    cdef long chunk_x_count = np.floor((source_xsiz-xsiz+1)/xstep)
    
    output = np.empty(  [ chunk_y_count, chunk_x_count,
                         ysiz,          xsiz
                         ],
                         dtype=float,   order='C'
                        )
    print( f"output size is {output.size}")
    chunk_y0 = 0
    for chunk_yi in range(chunk_y_count):
        chunk_x0 = 0                
        for chunk_xi in range(chunk_x_count):
            output[ chunk_yi, chunk_xi, :, : ] = \
                source[ chunk_y0:chunk_y0+ysiz,
                        chunk_x0:chunk_x0+xsiz
                        ]
            chunk_x0 += xstep
        chunk_y0 += ystep
    return output


    
            
            

    