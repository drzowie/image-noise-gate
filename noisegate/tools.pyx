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
from warnings import warn

# Set the compile-time type of the working arrays for noise_gate_batch.
# (This is a Cython optimization - types are used below)
CDTYPE = np.complex128
ctypedef np.complex128_t CDTYPE_t

RDTYPE = np.float64
ctypedef np.float64_t RDTYPE_t

IDTYPE = np.int64
ctypedef np.int64_t IDTYPE_t




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
    # single general-purpose case unless I drop all the way into C.  But 
    # the API is unified in case I (or you) later think of a general-purpose 
    # way of doing this that is more elegant.
    #
    # About the only real optimizations here are:
    #  (1) compilation of the hotspot variables through cdef
    #  (2) ordering of the loop nesting to reduce L2/L3 cache violations
    
    # Detect dimensionality
    N = len(source.shape)
    
    # Promote to a 1-D numpy array (size N) if scalar
    if( (not isinstance(size,np.ndarray)) or len(size)==1 ):
        size = size + np.zeros(N)
    if( (not isinstance(step,np.ndarray)) or len(step)==1 ):
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
        chunk_x_count = np.floor((source.shape[0]-xsize)/xstep).astype(int)+1
        
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
        
        chunk_x_count = 1+np.floor((source.shape[2]-xsize)/xstep).astype(int)
        chunk_y_count = 1+np.floor((source.shape[1]-ysize)/ystep).astype(int)
        chunk_z_count = 1+np.floor((source.shape[0]-zsize)/zstep).astype(int)
    
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
    
    if(not isinstance(step,np.ndarray) or len(step)==1):
        step = np.zeros(N)+step
    
    if(len(step) != N):
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
            (xn-1) * xstep + xsize
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
            (yn-1) * ystep + ysize,
            (xn-1) * xstep + xsize
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
            (zn-1) * zstep + zsize,
            (yn-1) * ystep + ysize,
            (xn-1) * xstep + xsize
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
            (wn-1) * wstep + wsize,
            (zn-1) * zstep + zsize,
            (yn-1) * ystep + ysize,
            (xn-1) * xstep + xsize
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
    
        
def hannify(source, order=2, axis=None, copy=False):
    '''
    hannify - Apply Hann windows to a broadcast collection of data "cubes"
    
    The Hann window is sin^2, scaled to the size of the window itself.  The 
    sin function is scaled so that the center of the window is unity and the 
    wavelength is the same as the width of the window.
    
    This is used because Hann windows and variants thereof have nice 
    summing properties, so if you choose the right offset between adjacent 
    Hann windows you don't have to do a scaled sum.  This kills two 
    birds with one stone, for the noisegate algorithm:  it provides good 
    apodization and also makes recombining the data better.

    Parameters
    ----------
    source :  Numpy array 
    axis : None or int or list of ints
        This is the axis to which the Hann window should be applied.  if it is
        None, then the last N axes are used.
    copy : Boolean (default False)
        Indicates whether the data should be copied before treatment.  if False
        (the default), then they are modified in place (and returned).  If True,
        then they are copied and the copy gets modified and returned.

    Returns
    -------
    None.

    '''
    cdef int axis_i
    cdef int N = len( source.shape )
    
    default_axis = (axis==None)
    if(default_axis):
        if( N % 2):
            raise ValueError("hannify: if axis is not specified, source must have 2N axes")
        axis = list( range( N/2, N ) )
    
        
    # Check for dups in the axis list and throw an error.  This is the st00pid
    # way to work but axis is generally a short list so who cares
    for axis1 in range( len(axis) ):
        for axis2 in range(axis1+1, len(axis) ):
            if(axis[axis1]==axis[axis2]):
                raise ValueError("hannify: axis specifier must not contain dups")
    
         
    if(copy):
        source = np.copy(source)
        
    ### To minimize passes through the data, we assemble a filter array that
    ### has as many axes as are called out in axis.  Then we transpose the 
    ### original data to put the relevant axes at the back (this should be 
    ### cheap in all cases and is a no-op in the default case).  Then we can
    ### apply the window with a single broadcast multiply.
    
    axis_sizes = [ source.shape[axis[i]] for i in range(len(axis)) ]
    filt = np.ones(axis_sizes)

    for axis_i in range( len(axis) ):
        axes = list( range( len(axis) ) )
        axes[ axis_i ]  = len(axis) - 1
        axes[ -1 ]      = axis_i
        fs = np.transpose( filt, axes=tuple( axes ) )
        
        # construct a sin^2 window.  Use cos^2 centered on the window,
        # because it's a little cleaner analytically.
        dex = np.arange(fs.shape[-1]).astype(float)
        dex -= fs.shape[ -1 ] / 2 - 0.5
        dex *= np.pi / fs.shape[ -1 ]
        window = np.cos(dex)
        window *= window

        # Broadcast multiply acts on the last dim of fs, which flows to filt
        fs *= window
        
    # All the loop above was to generate a filter function acting on all the 
    # axes called out in the "axis" list.  Now we have to make a window on source
    # with all those axes at the end, so broadcasting works.
    #
    # We only bother with all that if the axis field was specified since
    # in the default case they're in the right order already.
    #
    # Strategy: start with all axes in order.  Walk through and do explicit
    # exchanges until we have the right broadcast setup.
    if default_axis:
        
        ss = source
        
    else:
        
        axis_xpose_order = list( range( N ) )
        for axis_i in range( len(axis) ):
            if( axis_xpose_order[ - axis_i - 1 ] !=  axis[ - axis_i - 1 ] ):

                for axis_j in range( N ):
                    if( axis_xpose_order [ axis_j ] == axis[ - axis_i - 1 ] ):
                        # Exchange the two elements
                        a = axis_xpose_order[ axis_j ]
                        axis_xpose_order[ axis_j ] = axis_xpose_order [ -axis_i - 1 ]
                        axis_xpose_order[ -axis_i - 1 ] = a
                        break
                    else:
                        # None identified
                        raise ValueError("hannify: something went wrong with the axis transposition step")
        ss = np.transpose( source, axes = axis_xpose_order )
    
    # Do the transpose to put all the active axes at the end, then 
    # broadcast-multiply the filter.
    ss *= filt
               
    return source

######################################

# Make a Hann window of a given size.  Not heavily optimized 
cpdef hann_window(size):
    '''
    hann_window - make a 3D Hann window, and return it
    
    Not heavily optimized since it gets called only occasionally.

    Parameters
    ----------
    size : 3-element array-like
        The size of the window to create

    Returns
    -------
    a NumPy array containing the window

    '''
    out = np.zeros(size) + 1
    dex = np.mgrid[0:size[0],0:size[1],0:size[2]]

    out *= np.sin( (dex[0,...] + 0.5) * np.pi / size[0] )
    out *= np.sin( (dex[1,...] + 0.5) * np.pi / size[1] ) 
    out *= np.sin( (dex[2,...] + 0.5) * np.pi / size[2] ) 
    out *= out
    
    
    return out

    


cpdef get_noise_spectrum(np.ndarray[RDTYPE_t,ndim=3] source,
                              float pct=50,
                              float dkpct=5,
                              str model='shot',
                              cubesize=18,
                              cubediv=6,
                              subsamp=4,
                              verbose=False
                              ):
    '''
    get_noise_spectrum -- extract a noise spectrum from a 3D data set,
    without explicit cubification.  Returns a dictionary containing the 
    spectrum itself.
    
    The noise spectrum calculation follows the analysis in the DeForest 2017
    paper.

    Parameters
    ----------
    source : numpy array (3D)
        Data set to use.  Any vignetting function must have been removed already.
    float pct : float, optional
        The percentile value of Fourier magnitudes to keep for the noise spectrum. 
        The default is 50.
    float dkpct : float, optional
        The percentile value of Fourier magnitudes to keep for the dark portion
        of a hybrid noise spectrum (if relevant). The default is 5.
    str model : TYPE, optional
        The noise model to use -- must be "constant", shot", "hybrid", or .
        "multiplicative".  The default is "shot".
    int cubesize : int, optional
        The size of a neighborhood to treat.  This can be either a scalar or a 
        3-vector. It must be an integer multiple of cubediv.  The default is 18.
    int cubediv : int, optional
        This is the fraction of the cubesize to step between adjacent neighborhoods.
        It must divide evenly into cubesize. The default is 6.
    int subsamp : int, optional
        This is the fraction along each axis by which the available neighborhoods
        are to be subsampled. The default is 2, which samples 1/8 of the available
        neighborhoods.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # Converting model to a cdeffed "imodel" lets us use interger comparisons
    # in the hotspot loop below
    cdef int imodel
    if(model[0]=='c'):
        imodel = 1
    elif(model[0]=='s'):
        imodel = 2
    elif(model[0]=='h'):
        imodel = 3
    elif(model[0]=='m'):
        imodel = 4
    else:
        raise ValueError('get_noise_spectrum: model must be const, shot, hybrid, or multiplicative')

    # Make sure that the cubesize and subsamp are cdeffed ndarrays with integer type
    cdef np.ndarray[IDTYPE_t,ndim=1] cd_cubesize = np.zeros((3),dtype=IDTYPE) + np.array(cubesize,dtype=IDTYPE)
    cdef np.ndarray[IDTYPE_t,ndim=1] cd_cubediv  = np.zeros((3),dtype=IDTYPE) + np.array(cubediv, dtype=IDTYPE)
    cdef np.ndarray[IDTYPE_t,ndim=1] cd_subsamp  = np.zeros((3),dtype=IDTYPE) + np.array(subsamp, dtype=IDTYPE)
       
    if( np.any( cd_cubesize % cd_cubediv != 0 ) ):
        raise ValueError('get_noise_spectrum: cubesize and subsamp must be compatible.')

    if( np.any( (cd_cubediv != 3) * (cd_cubediv != 6) ) ):
        raise ValueError(f"get_noise_spectrum: only cubedivs of 3 and 6 are supported. ({cd_cubediv})")
             
    # Pull cube size out into individual ints for later use    
    cdef int cd_cube_zsize = cd_cubesize[0]
    cdef int cd_cube_ysize = cd_cubesize[1]
    cdef int cd_cube_xsize = cd_cubesize[2]
    
    # cdef loop ends
    cdef int source_zstop = source.shape[0] - cd_cube_zsize
    cdef int source_ystop = source.shape[1] - cd_cube_ysize
    cdef int source_xstop = source.shape[2] - cd_cube_xsize
    
    # Define a windowing function array
    cdef np.ndarray[RDTYPE_t,ndim=3] cd_hann = hann_window(cd_cubesize)
    
    # Define windows to hold the cubie before and after
    # the cubie gets pre-allocated.  The mcubie is calculated on the fly 
    # by np.abs.
    cdef np.ndarray[RDTYPE_t,ndim=3] cubie  = np.empty(cd_cubesize,dtype=RDTYPE)
    cdef np.ndarray[RDTYPE_t,ndim=3] mcubie 
  
    # Define strides through the source array for each cube.  This is how 
    # far we step for each comparison spectrum.
    cdef int source_zstride = cd_cube_zsize * cd_subsamp[0] // cd_cubediv[0]
    cdef int source_ystride = cd_cube_ysize * cd_subsamp[1] // cd_cubediv[1]
    cdef int source_xstride = cd_cube_xsize * cd_subsamp[2] // cd_cubediv[2]
    
    # Calculate how many spectra we are examining
    cdef int n_spectra = (
            int( source_zstop / source_zstride + 1 ) *
            int( source_ystop / source_ystride + 1) *
            int( source_xstop / source_xstride + 1)
            )

    # Define spectral arrays to hold the ensemble.  Allocate only those that
    # are needed.  Three types of arrays, so three variables.
    cdef np.ndarray[RDTYPE_t,ndim=4] const_spectra
    cdef np.ndarray[RDTYPE_t,ndim=4] shot_spectra
    cdef np.ndarray[RDTYPE_t,ndim=4] mult_spectra

    if( imodel == 1 or imodel==3):
        const_spectra = np.empty( ( cd_cube_zsize, cd_cube_ysize, cd_cube_xsize, n_spectra ) )
    if( imodel == 2 or imodel==3):
        shot_spectra = np.empty( ( cd_cube_zsize, cd_cube_ysize, cd_cube_xsize, n_spectra ) )
    if( imodel == 4):
        mult_spectra = np.empty( ( cd_cube_zsize, cd_cube_ysize, cd_cube_xsize, n_spectra ) )
    
    # Define some loop variables.  cdex points to which spectrum we are storing in the array.
    cdef int ix,iy,iz,icx,icy,icz
    cdef int cdex = 0
    cdef float acc
    
    # Main accumulation loop
    # Outer 3: across cubies
    for icz in range(0, source_zstop, source_zstride):
        for icy in range(0, source_ystop, source_ystride):
            for icx in range(0, source_xstop, source_xstride):
                
                # Copy a cubie into the holding tank
                for iz in range(0,cd_cube_zsize):
                    for iy in range(0,cd_cube_ysize):
                        for ix in range(0,cd_cube_xsize):
                            # I hate having so much pointer arithmetic here -- 
                            # dropping into C would go faster,
                            # with direct pointer increments.
                            cubie[iz,iy,ix] = cd_hann[iz,iy,ix] * source[ icz+iz, icy+iy, icx+ix]

                # Now Fourier transform.  Replace this with an FFTW call one day.
                # Might be slower because of the Python call to FFT; is deinitely 
                # slower because it constructs a new numpy array in np.abs (not
                # to mention the unnecessary square roots in np.abs -- abs^2 would
                # be sufficient)
                mcubie = np.abs(np.fft.fftn( cubie ))

                # Accumulate spectra as needed. Hopefully the branches (A) go at
                # C speed and (B) benefit from brach prediction.  
                # It would be better to bring them outside the outer-3
                # loops, but not as much better as using FFTW instead of np.fft.
                
                # Dark spectrum for constant or hybrid model
                if( imodel==1 or imodel==3 ):
                    for iz in range(0,cd_cube_zsize):
                        for iy in range(0,cd_cube_ysize):
                            for ix in range(0,cd_cube_xsize):
                                const_spectra[iz,iy,ix,cdex] = mcubie[iz,iy,ix]
                                
                # Shot spectrum for shot or hybrid model
                if( imodel==2 or imodel==3):
                    # Accumulate sum-of-sqrts
                    acc = 0
                    for iz in range(0,cd_cube_zsize):
                        for iy in range(0,cd_cube_ysize):
                            for ix in range(0,cd_cube_xsize):
                                acc += sqrt(cubie[iz,iy,ix])
                    if(acc==0):
                        acc=1
                    for iz in range(0,cd_cube_zsize):
                        for iy in range(0,cd_cube_ysize):
                            for ix in range(0,cd_cube_xsize):
                                shot_spectra[iz,iy,ix,cdex] = mcubie[iz,iy,ix] / acc
                                
                # Proportional spectrum for mult model
                if( imodel==4):
                    # Accumulate sum
                    acc = 0
                    for iz in range(0,cd_cube_zsize):
                        for iy in range(0,cd_cube_ysize):
                            for ix in range(0,cd_cube_xsize):
                                acc += cubie[iz,iy,ix]
                    if(acc==0):
                        acc=1
                    for iz in range(0,cd_cube_zsize):
                        for iy in range(0,cd_cube_ysize):
                            for ix in range(0,cd_cube_xsize):
                                mult_spectra[iz,iy,ix,cdex] = mcubie[iz,iy,ix]/acc
                
                # Move to the next spectrum
                cdex+=1
                
    # Now we've accumulated a collection of spectra.  
    # Define the output dictionary (we're out of the hotspot) and then
    # do the sorting in numpy.
    
    out = {'model':model}
    
    assert(cdex==n_spectra)
    
    if( imodel==1 or imodel==3):
        const_spectra.sort(axis=-1)
        if(imodel==1):
            dex = int( (pct / 100.0) * cdex )
        else:
            dex = int( (dkpct / 100.0) * cdex )
        # Make sure we always keep the 0 and 1 components  
        for iz in (-1,0,1):
            for iy in (-1,0,1):
                for ix in (-1,0,1):
                    const_spectra[iz,iy,ix,dex] = 0                    
        out['const_spectrum'] = const_spectra[:,:,:,dex]+0
           
        
    if( imodel==2 or imodel==3):
        shot_spectra.sort(axis=-1)
        dex = int( pct/100 * cdex )
        
        # Make sure we always keep the 0 and 1 components 
        for iz in (-1,0,1):
            for iy in (-1,0,1):
                for ix in (-1,0,1):
                    shot_spectra[iz,iy,ix,dex] = 0
        out['shot_spectrum'] = shot_spectra[:,:,:,dex]+0
             
    
    if( imodel==4):
        mult_spectra.sort(axis=-1)
        dex = int( pct/100 * cdex )
        # Make sure we always keep the 0 and 1 components 
        for iz in (-1,0,1):
            for iy in (-1,0,1):
                for ix in (-1,0,1):
                    mult_spectra[iz,iy,ix,dex] = 0
        out['mult_spectrum'] = mult_spectra[:,:,:,dex]+0
            
    return out


cpdef noise_gate_batch(
        np.ndarray[RDTYPE_t,ndim=3] source,
        float pct=30,
        float dkpct=5,
        float factor=1.5,
        float dkfactor=1.5,
        str method='gate',
        str model='hybrid',
        cubesize=18,
        cubediv=3,
        subsamp=4,
        vignette=None,
        spectrum=None,
        ):
    '''
    noise_gate_batch - carry out noise gatng on an image sequence contained
    in a 3D numpy array.  Returns the modified array.
    
    The noise gating follows the analysis in the DeForest 2017 paper.
    
    This version tries to lean in to the Cython approach and eschews 
    cubification completely.
    
    Specific algorithmic adjustments are described in the parameters section 
    below
    
    The neighborhood processing strategy uses the identity that 
        sum(x=0,2π/3,4π/3) sin^4 (x + z) = 1.125
    to merge apodization and recombination.  Neighborhoods are subsampled by
    a factor of 3, and offset-summed to reconstitute the original data.  this
    means that a margin near each boundary of the data set remains apodized.
    The sin^4 comes from *double* apodization: each cube is apodized with a
    normal sin^2 window before the initial Fourier transform; and then apodized 
    again after it is inverse-transformed.  The second apodization minimizes 
    boundary effects from the (potentially radical) adaptive filtering, and 
    results in an overall windowing function of sin^4.
    

    Parameters
    ----------
    source : Numpy array with 3 dimensions
        This is a Numpy array containing the source data.  Only 3 dimensions
        are supported at the moment:  t, y, x.
    pct : float, optional
        The neighborhoods (cubies) in the data are treated as an ensemble, and
        scaled Fourier components are used to estimate noise level.  This is the
        percentile value (across neighborhoods) of each Fourier component, that
        is considered to be a noise level threshold.  The default value of 50
        assumes that most Fourier components are noise dominated in most
        locations of the image.  Lower values should be used for processing less 
        noisy image sets.
    dkpct : float, optional
        This works as pct, but for estimating dark noise.  The default value 
        of 5 assumes that the fifth-percentile value (across neighborhoods) of 
        any given Fourier component can be treated as the dark noise level.
        Dark noise is relevant for the "hybrid" noise model only.
    factor : float, optional
        This is the factor by which a Fourier component magnitude must exceed 
        its corresponding noise-model value, to be retained in the output 
        data.  The default value of 2 keeps approximately 3% of the background
        noise spuriously (2 sigma); this seems to be a good compromise between
        retaining noise and rejecting weak image features, for many applications.
    float dkfactor : TYPE, optional
        This is similar to the 'factor' parameter, but applies to dark noise
        in the hybrid noise model.
    method : string, optional
        This describes the filtering method used to handle neighborhood Fourier
        components that do not rise significantly above the noise model.  
        Accepted values are:
            gate: causes simple gating:  Fourier components that do not 
               significantly exceed the noise floor are explicitly set to 0
            wiener: causes a Wiener filter to be applied, with rollover at the
               significance threshold ('factor' parameter times the noise model).
        The default is 'gate'.
    model : string, optional
        This describes the noise model to be used for the source data. The noise
        model describes how noise is expected to vary as a function of image
        characteristics.  Accepted values are:
            constant: the dominant noise source is considered to be independent
               of image value.  This is suitable for data, such as solar 
               magnetograms or dark-noise-limited images in general, that are
               dominated by an additive noise source independent of image value.
            shot: the dominant noise source is considered to scale like the 
               square root of image value, like Poisson "shot" noise from 
               photon counting
            hybrid: two noise sources are considered:  shot noise, and also a
               dark noise "floor" in dark portions of the image.  This is the
               most commonly applicable model for direct scientific images 
               that are shot noise limited with an APS or CCD detector.
            multiplicative: the dominant noise term is considered to be a 
               constant multiplicative noise source or, equivalently, an 
               additive noise source whose RMS value scales with image value.
        The default is "hybrid".
    cubesize : int, optional
        This is the number of pixels in a neighborhood ('cubie') to be considered
        for a local adaptive filter.  The number must be divisible by 3.  You 
        can specify either a single integer, in which case the neighborhoods are
        cubical, or an array of three values in (t, y, x) order, in which case
        the neighborhoods can have separate size along each axis. For maximum
        noise reduction the neighborhood should be comparable to the maximum
        coherence length in the data.  For data that include rapidly moving
        features which do not, themselves, evolve rapidly, you should consider
        matching the t size to the x and y sizes, based on the expected speed
        of the features.
    cubediv int, optional
        This is the fraction of a neighborhood size to step between neighborhoods
        when processing.  3 gives 2/3 overlap between neighborhoods in all 
        axes, 6 gives 5/6 overlap.
    subsamp : int, optional
        This allows the noise-spectrum estimator to subsample the cubies data 
        by the specified factor along each axis, saving computing time. The 
        default is 2, which reduces the number of Fourier transforms for this 
        step by a factor of 8.  
    spectrum : dictionary, optional
        If present, this should be a dictionary returned by get_noise_spectrum.
        If not present, a new noise spectrum is calculated for this batch.  
        Since the noise spectrum model is constant for each instrument 
        configuration, there is no nead to recalculate the spectrum for 
        different batches processed as part of a larger data set; this
        parameter lets you explicitly calculate a spectrum first, then call
        noise_gate_batch multiple times to process a data set too large to 
        fit in memory simultaneously.
    vignette : numpy 2-D array or none
        If supplied, this is a vignetting function of the instrument. The data
        are assumed to be already corrected for vignetting, so they are scaled 
        by the vignetting function before processing, then it is removed after
        processing.  This produces better uniformity of noise characeristics in,
        e.g., coronagraphs that use vignetting to reduce detector dynamic range.

        
    Returns
    -------
    A numpy array matching the source array, containing the cleaned-up data.

    '''
    
    # Converting model to a cdeffed "imodel" lets us use integer comparisons later
    cdef int imodel
    if(model[0] =='c'):
        imodel=1
    elif(model[0]=='s'):
        imodel=2
    elif(model[0]=='h'):
        imodel=3
    elif(model[0]=='m'):
        imodel=4
    else:
        raise ValueError("noise_gate_batch: Mode must be constant, hybrid, shot, or multiplicative")
        
    # Converting model to a cdeffed "imethod" lets us use integer comparisons later
    cdef int imethod
    if(method[0]=='g'):
        imethod = 1
    elif(method[0]=='w'):
        imethod = 2
        raise ValueError("noise_gate_batch: wiener filter not implemented (yet)")
    else: 
        raise ValueError("noise_gate_batch: method must be gate or wiener")
  
    
    # Make sure that the cubesize and subsamp are cdeffed ndarrays with integer type
    cdef np.ndarray[IDTYPE_t,ndim=1] cd_cubesize = np.zeros((3),dtype=IDTYPE) + np.array(cubesize,dtype=IDTYPE)
    cdef np.ndarray[IDTYPE_t,ndim=1] cd_cubediv  = np.zeros((3),dtype=IDTYPE) + np.array(cubediv, dtype=IDTYPE)
    cdef np.ndarray[IDTYPE_t,ndim=1] cd_subsamp  = np.zeros((3),dtype=IDTYPE) + np.array(subsamp, dtype=IDTYPE)
       
    
    if( np.any( cd_cubesize % cd_cubediv != 0 ) ):
        raise ValueError('noise_gate_batch: cubesize and subsamp must be compatible.')

    if( spectrum is None):
       spectrum = get_noise_spectrum(source, 
                                            pct=pct,
                                            dkpct=dkpct,
                                            model=model,
                                            cubesize=cubesize,
                                            cubediv=cubediv,
                                            subsamp=subsamp
                                            )
       
    # Pull cube size out into individual ints for later use    
    cdef int cd_cube_zsize = cd_cubesize[0]
    cdef int cd_cube_ysize = cd_cubesize[1]
    cdef int cd_cube_xsize = cd_cubesize[2]
    
    # cdef loop ends
    cdef int source_zstop = source.shape[0] - cd_cube_zsize
    cdef int source_ystop = source.shape[1] - cd_cube_ysize
    cdef int source_xstop = source.shape[2] - cd_cube_xsize
    
    # Define strides through the source array for each cube.  This is how 
    # far we step for each cubie.
    cdef int source_zstride = cd_cube_zsize // cd_cubediv[0]
    cdef int source_ystride = cd_cube_ysize // cd_cubediv[1]
    cdef int source_xstride = cd_cube_xsize // cd_cubediv[2]

    # Grab a Hann window function (sin**2)
    cdef np.ndarray[RDTYPE_t,ndim=3] cd_hann = hann_window(cd_cubesize)
    cdef np.ndarray[RDTYPE_t,ndim=3] cd_hann2
    
    # Set the Hann scaling function (trig on sin**4 sums)
    cdef float hannscale = 1
    cdef int i
    for i in range(3):
        if(cd_cubediv[i] == 3):
            hannscale *= 8.0/9.0
        elif(cd_cubediv[i] == 6):
            hannscale *= 4.0/9.0
        else:
            raise ValueError(f"noise_gate_batch: only cubedivs of 3 and 6 are supported. ({cd_cubediv})")
    cd_hann2 = cd_hann * hannscale
    
    # Allocate the output array.  Has to be set to zero since we accumulate on the fly
    cdef np.ndarray[RDTYPE_t,ndim=3] dest = np.zeros([source.shape[0],source.shape[1],source.shape[2]])
    
    # Allocate the "holding tank" and the Fourier transform for each cubie
    cdef np.ndarray[RDTYPE_t,ndim=3] cubie = np.empty(cd_cubesize,dtype=RDTYPE)
    cdef np.ndarray[CDTYPE_t,ndim=3] fcubie
    cdef np.ndarray[RDTYPE_t,ndim=3] mcubie 
    
 
    # Extract the spectra we need
    cdef np.ndarray[RDTYPE_t,ndim=3] const_spectrum
    cdef np.ndarray[RDTYPE_t,ndim=3] var_spectrum
    
    if(imodel==2 or imodel==3):
        shot_spectrum = spectrum['shot_spectrum'] * factor
    if(imodel==4):
        mult_spectrum = spectrum['mult_spectrum'] * factor
        
    # Define some loop variables.
    cdef int ix, iy, iz, icx, icy, icz
    cdef float acc
    
    # Figure relevant spectrum reference
    if(imodel==1):
        const_spectrum = spectrum['const_spectrum'] * factor
    elif(imodel==2):
        var_spectrum = spectrum['shot_spectrum'] * factor
    elif(imodel==3):
        const_spectrum = spectrum['const_spectrum'] * dkfactor
        var_spectrum = spectrum['shot_spectrum'] * factor
    elif(imodel==4):
        var_spectrum = spectrum['mult_spectrum'] * factor
    
    
    for icz in range(0, source_zstop, source_zstride):
        for icy in range(0, source_ystop, source_ystride):
            for icx in range(0, source_xstop, source_xstride):
                    
                # Copy the current cubie into the holding tank
                for iz in range(0,cd_cube_zsize):
                    for iy in range(0, cd_cube_ysize):
                        for ix in range(0, cd_cube_xsize):
                            cubie[iz, iy, ix] = cd_hann[iz, iy, ix] * source[ icz+iz, icy+iy, icx+ix]
                
                # Fourier transform
                fcubie = np.fft.fftn( cubie )
                mcubie = np.abs(fcubie)
                
                # Accumulate a local scaling factor if necessary
                if( imodel ==2 or imodel==3 ):
                    # Shot noise needs the sum-of-sqrts
                    acc = 0
                    for iz in range(0,cd_cube_zsize):
                        for iy in range(0,cd_cube_ysize):
                            for ix in range(0,cd_cube_xsize):
                                acc += sqrt(cubie[iz,iy,ix])
                elif( imodel==4 ):
                    # Multiplicative noise needs the sum
                    acc = 0
                    for iz in range(0,cd_cube_zsize):
                        for iy in range(0,cd_cube_ysize):
                            for ix in range(0,cd_cube_xsize):
                                acc += cubie[iz,iy,ix]

                # Do the actual filtering: constant spectrum, variable spectrum, or hybrid
                if( imodel==1):
                    for iz in range(0,cd_cube_zsize):
                        for iy in range(0,cd_cube_ysize):
                            for ix in range(0,cd_cube_xsize):
                                if(mcubie[iz,iy,ix] < const_spectrum[iz,iy,ix]):
                                    fcubie[iz,iy,ix]=0
                elif( imodel==2 or imodel==4):
                    for iz in range(0,cd_cube_zsize):
                        for iy in range(0,cd_cube_ysize):
                            for ix in range(0,cd_cube_xsize):
                                if(mcubie[iz,iy,ix] < var_spectrum[iz,iy,ix] * acc):
                                    fcubie[iz,iy,ix]=0
                elif( imodel==3):
                    for iz in range(0,cd_cube_zsize):
                        for iy in range(0,cd_cube_ysize):
                            for ix in range(0,cd_cube_xsize):
                                if(mcubie[iz,iy,ix] < var_spectrum[iz,iy,ix] * acc or
                                   mcubie[iz,iy,ix] < const_spectrum[iz,iy,ix]):
                                    fcubie[iz,iy,ix]=0

                # Inverse Fourier transform the cubie and accumulate it into the destination
                mcubie = np.real(np.fft.ifftn(fcubie))
                for iz in range(0,cd_cube_zsize):
                    for iy in range(0,cd_cube_ysize):
                        for ix in range(0,cd_cube_xsize):
                            dest[ icz+iz, icy+iy, icx+ix] += mcubie[iz, iy, ix] * cd_hann2[iz, iy, ix]
                           
    return dest

                        

                    
                                
                
                
                
                
    

 
