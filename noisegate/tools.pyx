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
    # single general-purpose case unless I drop all the way in to C.  But 
    # the API is unified in case I (or you) later think of a general-purpose 
    # way of doing this that is more elegant.
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
        (xsize,xstep) = (size[0],step[0])
        chunk_x_count = np.floor((source.shape[0]-xsize+1)/xstep).astype(int)
        
        output = np.empty([chunk_x_count, xsize],dtype=float,order='C')

        chunk_x0 = 0
        for chunk_xi in range(chunk_x_count):
            output[ chunk_xi, : ] = \
                source[ chunk_x0:chunk_x0+xsize ]
            chunk_x0 += xstep
        return output
 
    elif(N==2):
        ( xsize, xstep ) = ( size[1], step[1] )
        ( ysize, ystep ) = ( size[0], step[0] )

        chunk_x_count = np.floor((source.shape[1]-xsize+1)/xstep).astype(int)
        chunk_y_count = np.floor((source.shape[0]-ysize+1)/ystep).astype(int)

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
        ( xsize, xstep ) = ( size[2], step[2] )
        ( ysize, ystep ) = ( size[1], step[1] )
        ( zsize, zstep ) = ( size[0], step[0] )
        
        chunk_x_count = np.floor((source.shape[2]-xsize+1)/xstep).astype(int)
        chunk_y_count = np.floor((source.shape[1]-ysize+1)/ystep).astype(int)
        chunk_z_count = np.floor((source.shape[0]-zsize+1)/zstep).astype(int)
    
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
        ( xsize, xstep ) = ( size[3], step[3] )
        ( ysize, ystep ) = ( size[2], step[2] )
        ( zsize, zstep ) = ( size[1], step[1] )
        ( wsize, wstep ) = ( size[0], step[0] )
    
        chunk_x_count = np.floor((source.shape[3]-xsize+1)/xstep).astype(int)
        chunk_y_count = np.floor((source.shape[2]-ysize+1)/ystep).astype(int)
        chunk_z_count = np.floor((source.shape[1]-zsize+1)/zstep).astype(int)
        chunk_w_count = np.floor((source.shape[0]-wsize+1)/wstep).astype(int)
    
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
            

def unshred(
        source,
        step,
        average=False
        ):
    '''
    unshred - reconstitute a previously-shredded array from the neighborhoods
    that were chopped up.  You have to specify the step size through the 
    output array associated with jumping from chunk to chunk.  This should be
    left over from when you shredded the array earlier.
    
    When a given pixels is present in more than one chunk, unshred can either
    sum (the default) or average (if a flag is set) across the multiple values.
    (NOTE: averaging is not yet implemented, only summing)

    Parameters
    ----------
    source : Numpy array with 2N axes
        A previously-shredded array to recombine
    step : Arraylike 
        Τhis is the cross-neighborhood stride in the reconstituted array.
        It should contain either one integer (in which case it applies to 
        all the axes) or N integers.
    average : boolean
        If this flag is true, average across pixel values that are duplicated
        in more than one chunk of the shredded array.  Normally, simple summing
        is used -- this requires proper apodization.

    Returns
    -------
    A numpy array with N axes, containing the reconstituted data
    '''
    cdef long N
    NN = len(source.shape)
    if(NN%2):
        raise ValueError("unshred: source array must have 2N axes")
    N = NN/2
    
    if(len(step)==1):
        step = np.zeros(N)+step
    
    if(step.size != N):
        raise ValueError("unshred: step array must match original source")
    
    if(average):
        raise AssertionError("unshred: averaging is not implemented")
   
    cdef long xsize,ysize,zsize,wsize
    cdef long xn,yn,zn,wn
    cdef long xi,yi,zi,wi
    cdef long x0,y0,z0,w0
    cdef long xstep,ystep,zstep,wstep
    
    if( N==1 ):
        ( xstep, xn, xsize ) = ( step[0], source.shape[0], source.shape[0+1])
       
        out = np.zeros( [
            xn * xstep + xsize
            ] )
        
        x0 = 0
        for xi in range(xn):
            out[x0:x0+xsize] += source[xi,:]
            x0 += xstep
        return out
    
    elif( N==2) :
        ( xstep, xn, xsize ) = ( step[1], source.shape[1], source.shape[1+2] )
        ( ystep, yn, ysize ) = ( step[0], source.shape[0], source.shape[0+2] )
            
        out = np.zeros( [
            yn * ystep + ysize,
            xn * xstep + xsize
            ] )
        
        y0 = 0
        for yi in range(yn):
            x0 = 0
            for xi in range(xn):
                out[y0:y0+ysize, x0:x0+xsize] += source[yi,xi,:,:]
                x0 += xstep
            y0 += ystep
        return out
    
    elif( N==3 ):
        ( xstep, xn, xsize ) = ( step[2], source.shape[2], source.shape[2+3] )
        ( ystep, yn, ysize ) = ( step[1], source.shape[1], source.shape[1+3] )
        ( zstep, zn, zsize ) = ( step[0], source.shape[0], source.shape[0+3] )
        
        out = np.zeros( [
            zn * zstep + zsize,
            yn * ystep + ysize,
            xn * xstep + xsize
            ] )
        
        z0 = 0
        for zi in range(zn):
            y0 = 0
            for yi in range(yn):
                x0 = 0
                for xi in range(xn):
                    out[z0:z0+zsize, y0:y0+ysize, x0:x0+xsize] += source[zi,yi,xi,:,:,:]
                    x0 += xstep
                y0 += ystep
            z0 += zstep
        return out
    
    elif( N==4 ):
        ( xstep, xn, xsize ) = ( step[3], source.shape[3], source.shape[3+4] )
        ( ystep, yn, ysize ) = ( step[2], source.shape[2], source.shape[2+4] )
        ( zstep, zn, zsize ) = ( step[1], source.shape[1], source.shape[1+4] )
        ( wstep, wn, wsize ) = ( step[0], source.shape[0], source.shape[0+4] )
        
        out = np.zeros( [
            wn * wstep + wsize,
            zn * zstep + zsize,
            yn * ystep + ysize,
            xn * xstep + xsize
            ])
        
        w0 = 0
        for wi in range(wn):
            z0 = 0
            for zi in range(zn):
                y0 = 0
                for yi in range(yn):
                    x0 = 0
                    for xi in range(xn):
                        out[w0:w0+wsize, z0:z0+zsize, y0:y0+ysize, x0:x0+xsize] \
                            += source[wi,zi,yi,xi,:,:,:,:]
                        x0 += xstep
                    y0 += ystep
                z0 += zstep
            w0 += wstep
        return out
    
    else:
        raise(ValueError("unshred: dimension N must be 1, 2, 3, or 4"))
    
                        
        
            
            

    