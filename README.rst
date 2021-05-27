================
Image-noise-gate
================

Noise gating removes noise from image sequences using spatiotemporal coherence
to identify statistically significant features of the images. In noise-limited
image sequences (i.e. those with visible dark noise or photon shot noise) with
sufficient spatial and temporal resolution to capture their subject, noise 
gating can improve SNR by roughly an order of magnitude (i.e. increase effective
exposure time by 100x).  

Image-noise-gate packages some helper routines and a single main entry point:
noise_gate_batch, which accepts a collection of images as a 3-D NumPy array. 
The image batch is cut into smaller "cubies" that represent neighborhoods
in the original data set.  The cubies are apodized, Fourier transformed, and
compared to a noise model (which may be constant noise, such as detector 
read noise; shot noise, for photon limited images; a hybrid model using
shot noise and a constant noise floor; or multiplicative noise).  The noise
model is specific to the instrument used to collect the data; if you do not 
supply one, it is assembled from the data a posteriori.  Fourier components
that do not exceed the noise model by a specified factor are zeroed ("gate" 
method) or smoothly attenuated with a Wiener filter ("wiener" method).  The
cleaned data are then inverse Fourier transformed and finally reassembled 
into the original shape.

The method leaves an apodized border 2/3 the size of the "cubies", along each 
edge of the data -- so if you specify a neighborhood size of 12x12x12, a
swath 8 pixels wide is invalidated at each border of the data set.

Typical usage
=============

The basic operation is to noise-gate a batch of images that can all fit in
RAM.  That operation is implemented with noise_gate_batch().  Large-batch 
and stream processing are also possible and are implemented with wrapper 
functions around noise_gate_batch.


Batch processing
----------------

noise_gate_batch(data_cube, cubesize=12, model='hybrid', factor=2.0)


Large-batch processing
----------------------

Not implemented yet


Stream processing
-----------------

Not implemented yet


Theory
======

The code works on two key principles: (1) the Fourier transform concentrates
recognizable image features, but not noise -- easing discrimination between
noise and features, without direct reference to the position of a feature
within a neighborhood; (2) in image sequences that are strongly affected by 
noise, it is possible to estimate both the overall noise level and how it 
varies across the dataset, from the dataset itself.  

The noise model generation involves treating the Fourier spectra of all the 
"cubies" in the data set, as a statistical ensemble.  From this ensemble, the
code generates a noise spectrum specific to the data source, using the 
particular model specified ( constant, shot, hybrid, or multiplicative ).

The noise gating part involves estimating a location-specific noise model 
using the contents of each cubie and the supplied noise spectrum.  Finally,
Fourier components of each cubie that do not exceed the corresponding noise 
model are either zeroed or attenuated.  

For more details, see DeForest 2017 (ApJ 838, 155).

History
=======

This package was ported and adapted from a collection of Perl/PDL routines
"noise_gate") first written in 2017.  

Commercial use of the method is subject to U.S. Patent No. 10181181.

Notes
=====

This version of the method makes use of NumPy operations (such as fft)
in the per-neighborhood hot spot, so it is ~2x slower than the PDL code. 
Dropping into C and using fftw and other per-pixel primitive operations
might bring it up to parity with PDL.

Author
======

Craig DeForest
Southwest Research Institute
deforest@boulder.swri.edu
