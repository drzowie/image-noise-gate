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
        print(f"hannify: xpose order is {axis_xpose_order}")
    
    # Do the transpose to put all the active axes at the end, then 
    # broadcast-multiply the filter.
    ss *= filt
               
    return source


# find_ng_spectrum is a hotspot so it should be fully cdeffed.  It isn't, since
# the sorting means that vectorization cache-breakage doesn't cost all that much.

def get_noise_spectrum( source, 
                    float pct=50, 
                    float dkpct=5, 
                    int   subsamp=2, 
                    str   model='shot'):
    '''
    get_noise_spectrum - extract a noise spectrum from an already-cubified
    3D data set.  Returns a dictionary containing the spectrum itself.  
    
    The noise spectrum calculation follows the analysis in the DeForest 2017
    paper.
    
    You should already have cubified and hannified the data for this call.
    
    find_spectrum is specifically for 3D work and therefore requires a 6D 
    (cubified) input.

    Parameters
    ----------
    source : numpy array (6D)
        Cubified, hannified data set to use.  First 3 axes run across cubies,
        last 3 axes are within each cubie.
    pct : floating-point, optional
        This is the percentile value of each Fourier component magnitude 
        (considered as an ensemble across cubes and modified, if relevant,
        with the shot noise estimator) to treat as the noise floor. The 
        default is 50, which uses the median and is suitable for images that
        are noise dominated in most Fourier components.
    dkpct : For the hybrid mode, this specifies the percentile of each Fourier
        component magnitude that is used for the constant term in the hybrid
        model.  The default is 5, which gets the darkest portion of the image
        sequence and is suitable for data sets in which a noticeable amount
        of the data are dominated by dark noise effects.
    model : string, optional
        This is the noise model to use when finding the spectrum.  The default
        is 'shot'.  Valid values are 'constant', 'multiplicative', 'shot', and
        'hybrid [shot and constant]'.  Only the first character is used. 
        The default is 'shot'.
    subsamp : integer, optional
        This is the stride to take through each cube axis.  Most data sets
        have a large enough number of subcubes that subsampling can greatly
        increase speed at no real cost in precision.  The default value of 2
        speeds up spectrum extraction by a factor of 8

    Returns
    -------
    A dictionary containing the spectrum, the dark spectrum if relevant, and
    metadata about the cube size and mode.  The dictionary is suitable for
    passing in to ng_batch or ng_sequence.

    '''
    if(not isinstance(source,np.ndarray)):
        raise ValueError("get_noise_spectrum: source data must be a NumPy array")
    if(len(source.shape) != 6):
        raise ValueError("get_noise_spectrum: source data must have 6 dimensions")
    if(dkpct >= 100 or dkpct < 0):
        raise ValueError("get_noise_spectrum: dkpct is out of range [0,100)")
    if(pct >= 100 or dkpct < 0):
        raise ValueError("get_noise_spectrum: pct is out of rante [0,100)")
        
    # Output is a dictionary.  Populate as needed.
    out = {'model':model}


    # Subsample the cubies array and Fourier transform the subset.  We don't
    # care about the phase so we just grab magnitude ('abs').
    # The "cubies" array is ( N, z, y, x ), where N is the number of cubies
    # we're going to look at, and the other are dimensions of each cubie.
    cubies = source[0::subsamp,0::subsamp,0::subsamp,...]. \
        reshape([-1, source.shape[3], source.shape[4], source.shape[5]])

    # Fourier transform is *after* the sum-of-square-roots in case we ever 
    # decide to do it in-place.  This is really wasteful since the fft3
    # converts everything to complex and also copies all the data.
    fcubies = np.abs(np.fft.fftn(cubies,axes=(-3,-2,-1)))
    
    # use m as a shorthand
    m = model

    if( m[0] not in {'c','h','s','m','n'}):
        raise ValueError("get_noise_spectrum: Mode must be constant, hybrid, shot, or multiplicative")

    # Both constant and hybrid mode use a constant spectrum with no scaling.
    # Accumulate that now, so we can scale (aka screw up) the fcubies spectra
    # for the shot noise calculation later.
    if(m[0] == 'c' or m[0] == 'h'):
        if(m[0]=='h'):
            frac = dkpct/100
            out['const_pct'] = dkpct
        else:
            frac = pct/100
            out['const_pct'] = pct
        dex = np.floor(frac * fcubies.shape[0]).astype(int)
         
        fcubies.sort(axis=0)
        out['const_spectrum'] = fcubies[dex,:,:,:]
        
    # Both shot and hybrid mode use sum-sqrt scaled spectra.
    if(m[0] == 's' or m[0] == 'h'):
        # Take the square root of each pixel and then sum over each cubie. 
        # The cubies get clipped at 1e-20 since shot noise treats each value
        # as positive-definite.
        cubies_sumsqrt = np.sum(np.sqrt(cubies.clip(1e-20)),axis=(-1,-2,-3))

        # reshape back to be broadcastable with fcubies, then scale the fcubies
        fcubies /= cubies_sumsqrt.reshape([-1,1,1,1])
        fcubies.sort(axis=0)
        dex = np.floor( pct / 100 * fcubies.shape[0] ).astype(int)
        
        out['shot_spectrum'] = fcubies[dex,:,:,:]
    
    # multiplicative mode just scales by the average value..
    if(m[0]=='m'):
        # Since the Fourier transform found the average value already, 
        # we just divide by that instead of calculating it again.
        # divide by that. 
        fcubies /= fcubies[:,0,0,0].reshape([-1,1,1,1])
        fcubies.sort(axis=0)
        
        dex = np.floor( pct/100 * fcubies.shape[0] )
        
        out['mult_spectrum'] = fcubies[dex,:,:,:]
 
    return out



def noise_gate_batch(
        source,
        float pct=50,
        float dkpct=5,
        float factor=2.0,
        float dkfactor=2.0,
        str method='gate',
        str model='hybrid',
        int cubesize=18,
        int subsamp=2,
        vignette=None,
        spectrum=None,
        ):
    '''
    noise_gate_batch - carry out noise gatng on an image sequence contained
    in a 3D numpy array.  Returns the modified array.
    
    The noise gating follows the analysis in the DeForest 2017 paper.
    
    the data are cubified nto neighborhoods and subjected to Hann windowing,
    then (if necessary) sent to the noise spectrum estimator 
    (get_noise_spectrum).  Then the cubified data are compared to the noise 
    spectrum and filtered with a locally constructed adaptive filter that keeps 
    the significant terms in the local neighborhood Fourier transform around 
    each point.  Finally the cubies are reassembled to match the original data.
    
    Specific algorithmic adjustments are described in the parameters section 
    below
    
    The cubification strategy uses the identity that 
        sum(x=0,2π/3,4π/3) sin^4 (x + z) = 1.125
    to merge apodization and recombination.  The neighborhoods are subsampled by
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
 
    # use m as a shorthand for the model
    m = model
    
    if( m[0] not in {"c","h","s","m","n"}):
       raise ValueError("noise_gate_batch: Mode must be constant, hybrid, shot, or multiplicative")
  
    cdef int m2
    if(method[0]=='g'):
        m2 = 0
    elif(method[0]=='w'):
        m2 = 1
    else:
        raise ValueError("noise_gate_batch: method must be gate or wiener")
        
        
    if( (spectrum is not None) and (spectrum['model'] != m)):
        raise ValueError("noise_gate_batch: spectrum mode must match gating mode")
    
    # Check for  vignetting function if present, and decorrect
    if( vignette is not None ):
        source *= vignette
    
    # Cut up the data -- have to do this first, in case we call get_noise_spectrum.
    cubies = shred(source, cubesize, cubesize/3)
    hannify(cubies)

    # Call get_noise_spectrum if necessary
    if(spectrum is None):
        spectrum = get_noise_spectrum(cubies, pct=pct, dkpct=dkpct, 
                                      subsamp=subsamp, model=model)
        
    # Now drop into the optimized central loop to do the gating.  These are
    # separate routines to provide all-cdef variables that Cython needs for 
    # best optimization.
    if(m[0] =='c'):
        noise_gate_const(cubies, spectrum['const_spectrum'], factor, m2)
        
    elif(m[0]=='h'):
        noise_gate_hybrid(cubies, spectrum['const_spectrum'],spectrum['shot_spectrum'], 
                          factor, dkfactor, m2)
    elif(m[0]=='s'):
        noise_gate_shot(cubies, spectrum['shot_spectrum'], factor, m2)
        
    elif(m[0]=='m'):
        noise_gate_mult(cubies, spectrum['mult_spectrum'], factor, m2)
        
    elif(m[0]=='n'):
        warn("noise_gate_batch: model 'none' selected; ignoring gating step")
    
    # Do the second apodization and reconstitute.  The 1.125 comes from the
    # trig identity sum{i=0,1,2}{sin(x+2*i*PI/3)**4} = 1.125
    # The cube is because we're working in three axes.
    hannify(cubies);
    out = unshred(cubies,cubesize/3) / (1.125**3)
    
    if( vignette is not None ):
        source /= (vignette + (vignette==0))
    
    return out


#####################
# These noise_gate_<method> routines are broken out so that Cython can fully
# compile them.  But they're not as fully optimized as they could be.  In 
# particular, the fastest algorithm would dip into C and use pointer increment
# operations instead of indexing calculations for every array index.
#
# Maybe fast enough for now; maybe needs to be refined later.
#
# Note the dimensionality spec -- Cythonic optimization that allows direct
# indexing instead of object access.
#
# method is 0 for gating, 1 for wiener
#
cdef noise_gate_const(np.ndarray[RDTYPE_t, ndim=6] cubies, 
                      np.ndarray[RDTYPE_t, ndim=3] const_spec, 
                      float factor, 
                      int method):
    assert cubies.dtype == RDTYPE
    assert const_spec.dtype == RDTYPE
    
    assert(method==0 or method==1)
    
    cdef int ch_z, ch_y, ch_x
    cdef int max_z = cubies.shape[0]  # max_{xyz} indexes cubes
    cdef int max_y = cubies.shape[1]
    cdef int max_x = cubies.shape[2]
    cdef int siz_z = cubies.shape[3]  # siz_{xyz} indexes within each cube
    cdef int siz_y = cubies.shape[4]
    cdef int siz_x = cubies.shape[5]
    cdef int z,y,x
    cdef float snr, wf
    
    
    # We prepare the spectrum by prescaling with factor ot avoid a mult in the
    # hotspot.  Also allocate a single complex spectrum and scalar cubie workspace.
    cdef np.ndarray[RDTYPE_t,ndim=3] c_spec = const_spec * factor
    cdef np.ndarray[RDTYPE_t,ndim=3] tmp_scale = np.zeros([siz_z,siz_y,siz_x],dtype=RDTYPE)
    cdef np.ndarray[CDTYPE_t,ndim=3] tmp_spec
    
  
    for ch_z in range(max_z):
        for ch_y in range(max_y):
            for ch_x in range(max_x):
                # fftn call drops into Python.  Not the very hottest of hot
                # spots (that's inside the zyx loops) but not great either.  
                # This is where dropping into C and calling fftw3 might
                # help.
                tmp_spec = np.fft.fftn(cubies[ch_z,ch_y,ch_x],axes=(0,1,2))
                tmp_scale = np.abs(tmp_spec)
                
                # Explicit full Cythonized loop over interior -- is this 
                # better than implicit?  Dunno.  Docs say it is.
                if(method==0):
                    for z in range(siz_z):
                        for y in range(siz_y):
                            for x in range(siz_x):
                                if(tmp_scale[z,y,x] < c_spec[z,y,x]):
                                    tmp_spec[z,y,x] = 0
                elif(method==1):
                    for z in range(siz_z):
                        for y in range(siz_y):
                            for x in range(siz_x):
                                snr = tmp_scale[z,y,x] / c_spec[z,y,x]
                                wf = snr/(snr+1)
                                tmp_spec[z,y,x] = tmp_spec[z,y,x]*wf
                                
                # Dropping into Python to de-fft and stuff the value back into
                # the original cubies array.
                cubies[ch_z,ch_y,ch_x] = np.fft.ifftn(tmp_spec,axes=(0,1,2)).astype(float)
    
cdef noise_gate_hybrid(np.ndarray cubies, 
                      np.ndarray const_spec, 
                      np.ndarray shot_spec,
                      float dkfactor,
                      float factor, 
                      char method ):  
    assert cubies.dtype == RDTYPE
    assert const_spec.dtype == RDTYPE
    assert shot_spec.dtype==RDTYPE
    
    assert(method==0 or method==1)
    
    cdef int ch_z, ch_y, ch_x
    cdef int max_z = cubies.shape[0]  # max_{xyz} indexes cubes
    cdef int max_y = cubies.shape[1]
    cdef int max_x = cubies.shape[2]
    cdef int siz_z = cubies.shape[3]  # siz_{xyz} indexes within each cube
    cdef int siz_y = cubies.shape[4]
    cdef int siz_x = cubies.shape[5]
    cdef int z,y,x
    cdef float snr, wf
    cdef float sumsqrt
    
    # We prepare the spectrum by prescaling with factor to avoid a mult in the
    # hotspot.  Also allocate a single complex spectrum and scalar cubie workspace.
    cdef np.ndarray[RDTYPE_t,ndim=3] c_spec = const_spec * dkfactor
    cdef np.ndarray[RDTYPE_t,ndim=3] s_spec = shot_spec * factor
    cdef np.ndarray[RDTYPE_t,ndim=3] tmp_scale = np.zeros([siz_z,siz_y,siz_x],dtype=RDTYPE)
    cdef np.ndarray[CDTYPE_t,ndim=3] tmp_spec, tmp_cube

    for ch_z in range(max_z):
        for ch_y in range(max_y):
            for ch_x in range(max_x):
                # fftn call drops into Python.  Not the very hottest of hot
                # spots (that's inside the zyx loops) but not great either.  
                # This is where dropping into C and calling fftw3 might
                # help.
                tmp_spec = np.fft.fftn(cubies[ch_z,ch_y,ch_x],axes=(0,1,2))
                tmp_scale = np.abs(tmp_spec)
                
                # Get the sum-of-square-roots.  Probably needs to be 
                # done in C (like the above) for maximum speed.
                sumsqrt = np.sum(np.sqrt(np.abs(cubies[ch_z,ch_y,ch_x])))
                
                # Explicit full Cythonized loop over interior -- is this 
                # better than implicit?  Dunno.  Docs say it is.

                if(method==0):
                    for z in range(siz_z):
                        for y in range(siz_y):
                            for x in range((z==0 and y==0),siz_x):
                                if(tmp_scale[z,y,x] < c_spec[z,y,x] or 
                                   tmp_scale[z,y,x] < sumsqrt * s_spec[z,y,x]
                                   ):
                                    tmp_spec[z,y,x] = 0

                elif(method==1):
                    for z in range(siz_z):
                        for y in range(siz_y):
                            for x in range((z==0 and y==0),siz_x):
                                snr = tmp_scale[z,y,x] / (c_spec[z,y,x]**2 + (s_spec[z,y,x]*sumsqrt)**2)**0.5
                                wf = snr/(snr+1)
                                tmp_spec[z,y,x] = tmp_spec[z,y,x]*wf
                                
                # Dropping into Python again to de-fft and stuff the value back into
                # the original cubies array.
                cubies[ch_z,ch_y,ch_x] = np.fft.ifftn(tmp_spec,axes=(0,1,2)).astype(float)

    

cdef noise_gate_shot(np.ndarray cubies, 
                      np.ndarray shot_spec, 
                      float factor, 
                      char method ):
    assert cubies.dtype == RDTYPE
    assert shot_spec.dtype == RDTYPE
    
    assert(method==0 or method==1)
    
    cdef int ch_z, ch_y, ch_x
    cdef int max_z = cubies.shape[0]  # max_{xyz} indexes cubes
    cdef int max_y = cubies.shape[1]
    cdef int max_x = cubies.shape[2]
    cdef int siz_z = cubies.shape[3]  # siz_{xyz} indexes within each cube
    cdef int siz_y = cubies.shape[4]
    cdef int siz_x = cubies.shape[5]
    cdef int z,y,x
    cdef float snr, wf
    cdef float sumsqrt
    
    # We prepare the spectrum by prescaling with factor to avoid a mult in the
    # hotspot.  Also allocate a single complex spectrum and scalar cubie workspace.
    cdef np.ndarray[RDTYPE_t,ndim=3] s_spec = shot_spec * factor
    cdef np.ndarray[RDTYPE_t,ndim=3] tmp_scale = np.zeros([siz_z,siz_y,siz_x],dtype=RDTYPE)
    cdef np.ndarray[CDTYPE_t,ndim=3] tmp_spec, tmp_cube

    for ch_z in range(max_z):
        for ch_y in range(max_y):
            for ch_x in range(max_x):
                # fftn call drops into Python.  Not the very hottest of hot
                # spots (that's inside the zyx loops) but not great either.  
                # This is where dropping into C and calling fftw3 might
                # help.
                tmp_spec = np.fft.fftn(cubies[ch_z,ch_y,ch_x],axes=(0,1,2))
                tmp_scale = np.abs(tmp_spec)
                
                # Get the sum-of-square-roots.  Probably needs to be 
                # done in C (like the above) for maximum speed.
                sumsqrt = np.sum(np.sqrt(np.abs(cubies[ch_z,ch_y,ch_x])))
                
                # Explicit full Cythonized loop over interior -- is this 
                # better than implicit?  Dunno.  Docs say it is.

                if(method==0):
                    for z in range(siz_z):
                        for y in range(siz_y):
                            for x in range((z==0 and y==0),siz_x):
                                if(tmp_scale[z,y,x] < sumsqrt * s_spec[z,y,x]
                                   ):
                                    tmp_spec[z,y,x] = 0

                elif(method==1):
                    for z in range(siz_z):
                        for y in range(siz_y):
                            for x in range((z==0 and y==0),siz_x):
                                snr = tmp_scale[z,y,x] / (s_spec[z,y,x]*sumsqrt)
                                wf = snr/(snr+1)
                                tmp_spec[z,y,x] = tmp_spec[z,y,x]*wf
                                
                # Dropping into Python again to de-fft and stuff the value back into
                # the original cubies array.
                cubies[ch_z,ch_y,ch_x] = np.fft.ifftn(tmp_spec,axes=(0,1,2)).astype(float)


cdef noise_gate_mult(np.ndarray cubies, 
                      np.ndarray mult_spec, 
                      float factor, 
                      char method ):     
    assert cubies.dtype == RDTYPE
    assert mult_spec.dtype == RDTYPE
    
    assert(method==0 or method==1)
    
    cdef int ch_z, ch_y, ch_x
    cdef int max_z = cubies.shape[0]  # max_{xyz} indexes cubes
    cdef int max_y = cubies.shape[1]
    cdef int max_x = cubies.shape[2]
    cdef int siz_z = cubies.shape[3]  # siz_{xyz} indexes within each cube
    cdef int siz_y = cubies.shape[4]
    cdef int siz_x = cubies.shape[5]
    cdef int z,y,x
    cdef float snr, wf
    
    # We prepare the spectrum by prescaling with factor to avoid a mult in the
    # hotspot.  Also allocate a single complex spectrum and scalar cubie workspace.
    cdef np.ndarray[RDTYPE_t,ndim=3] m_spec = mult_spec * factor
    cdef np.ndarray[RDTYPE_t,ndim=3] tmp_scale = np.zeros([siz_z,siz_y,siz_x],dtype=RDTYPE)
    cdef np.ndarray[CDTYPE_t,ndim=3] tmp_spec, tmp_cube

    for ch_z in range(max_z):
        for ch_y in range(max_y):
            for ch_x in range(max_x):
                # fftn call drops into Python.  Not the very hottest of hot
                # spots (that's inside the zyx loops) but not great either.  
                # This is where dropping into C and calling fftw3 might
                # help.
                tmp_spec = np.fft.fftn(cubies[ch_z,ch_y,ch_x],axes=(0,1,2))
                tmp_scale = np.abs(tmp_spec)
                
                # Explicit full Cythonized loop over interior -- is this 
                # better than implicit?  Dunno.  Docs say it is.
                # The fillip in the x loop prevents scrozzling the origin component.

                if(method==0):
                    for z in range(siz_z):
                        for y in range(siz_y):
                            for x in range((z==0 and y==0),siz_x):
                                if(tmp_scale[z,y,x] < tmp_scale[0,0,0] * m_spec[z,y,x]
                                   ):
                                    tmp_spec[z,y,x] = 0

                elif(method==1):
                    for z in range(siz_z):
                        for y in range(siz_y):
                            for x in range((z==0 and y==0),siz_x):
                                snr = tmp_scale[z,y,x] / (tmp_scale[0,0,0] * m_spec[z,y,x])
                                wf = snr/(snr+1)
                                tmp_spec[z,y,x] = tmp_spec[z,y,x]*wf
                                
                # Dropping into Python again to de-fft and stuff the value back into
                # the original cubies array.
                cubies[ch_z,ch_y,ch_x] = np.fft.ifftn(tmp_spec,axes=(0,1,2)).astype(float)

