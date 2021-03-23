#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:59:31 2021

@author: zowie
"""

import noisegate as ng
import numpy as np

def test_001_shred2d():
    # 10x10 grid, 0 to 99 (Y,X indexing in final axis)
    a = (np.mgrid[0:10,0:10].T.astype(float) * np.array([10,1])).sum(axis=2)
    print(f"a shape is {a.shape}")
    assert(a.shape==(10,10))
    # Extract 2x4 regions, with steps of 2 in X and 3 in Y
    b = ng.shred2d(a,[2,4],[3,2])
    
