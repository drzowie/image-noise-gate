#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:59:31 2021

@author: zowie
"""

import noisegate as ng
import numpy as np

def test_001_shred():
    # 2-D: 10x10 grid, 0 to 99 (Y,X indexing in final axis)
    a = (np.mgrid[0:10,0:10].T.astype(float) * np.array([1,10])).sum(axis=2)
    print(f"a shape is {a.shape}")
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
                             
