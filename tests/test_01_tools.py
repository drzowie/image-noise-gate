#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:59:31 2021

@author: zowie
"""

import noisegate as ng
import numpy as np

def test_001_shred():
    # 1-D: 10-grid, 0 to 9 
    a = np.mgrid[0:10].T.astype(float)
    assert(a.shape==(10,))
    try:
        b = ng.shred(a,[2,4],[3])
        raise AssertionError("shred should bomb with bad dims")
    except: 
        pass
    try:
        b = ng.shred(a,[2],[3,2])
        raise AssertionError("shred should bomb with bad dims")
    except: 
        pass
    
    b = ng.shred(a,[2],[3])
    assert(b.shape==(3,2))
    
    
    # 2-D: 10x10 grid, 0 to 99 (Y,X indexing in final axis)
    a = (np.mgrid[0:10,0:10].T.astype(float) * np.array([1,10])).sum(axis=2)
    assert(a.shape==(10,10))
    # Extract 2x4 regions (Y,X), with steps of 2 in X and 3 in Y
    b = ng.shred(a,[2,4],[3,2])
    assert(b.shape==(3,3,2,4))
    assert( np.all( b[0,0] == 
                   np.array([[0,1,2,3],[10,11,12,13]])))
    assert( np.all( b[0,1] == 
                   np.array([[2,3,4,5],[12,13,14,15]])))
    assert( np.all( b[1,0] == 
                   np.array([[30,31,32,33],[40,41,42,43]])))
                             
    # 3-D: 10x10x10 grid, 0 to 999
    a = (np.mgrid[0:10,0:10,0:10].T.astype(float) * np.array([1,10,100])).sum(axis=3)
    assert(a.shape==(10,10,10))
    b = ng.shred(a,[2,4,6],[3,2,1])
    bb = np.array(
            [[[  0,  1,  2,  3,  4,  5],[ 10, 11, 12, 13, 14, 15],
              [ 20, 21, 22, 23, 24, 25],[ 30, 31, 32, 33, 34, 35]],
             [[100,101,102,103,104,105],[110,111,112,113,114,115],
              [120,121,122,123,124,125],[130,131,132,133,134,135]]]
            )
                  
    assert(np.all(b[0,0,0] == bb))
    assert(np.all(b[1,0,0] == bb+300))
    assert(np.all(b[0,1,0] == bb+20))
    assert(np.all(b[0,0,1] == bb+1))
                
    
    # 4-D: 10x10x10x10