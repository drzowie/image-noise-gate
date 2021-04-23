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
    a = (np.mgrid[0:10,0:10,0:10,0:10].T.astype(float)*np.array([1,10,100,1000])).sum(axis=4)
    assert(a.shape==(10,10,10,10))
    b = ng.shred(a,[2,3,4,6],[4,3,2,1])
    bb = np.array(
            [[[[   0,   1,   2,   3,   4,   5],[  10,  11,  12,  13,  14,  15],
               [  20,  21,  22,  23,  24,  25],[  30,  31,  32,  33,  34,  35]],
              [[ 100, 101, 102, 103, 104, 105],[ 110, 111, 112, 113, 114, 115],
               [ 120, 121, 122, 123, 124, 125],[ 130, 131, 132, 133, 134, 135]],
              [[ 200, 201, 202, 203, 204, 205],[ 210, 211, 212, 213, 214, 215],
               [ 220, 221, 222, 223, 224, 225],[ 230, 231, 232, 233, 234, 235]]],
             [[[1000,1001,1002,1003,1004,1005],[1010,1011,1012,1013,1014,1015],
               [1020,1021,1022,1023,1024,1025],[1030,1031,1032,1033,1034,1035]],
              [[1100,1101,1102,1103,1104,1105],[1110,1111,1112,1113,1114,1115],
               [1120,1121,1122,1123,1124,1125],[1130,1131,1132,1133,1134,1135]],
              [[1200,1201,1202,1203,1204,1205],[1210,1211,1212,1213,1214,1215],
               [1220,1221,1222,1223,1224,1225],[1230,1231,1232,1233,1234,1235]]]]
        )
    assert(np.all(b[0,0,0,0]==bb))
    assert(np.all(b[1,0,0,0]==bb+4000))
    assert(np.all(b[0,1,0,0]==bb+300))
    assert(np.all(b[0,0,1,0]==bb+20))
    assert(np.all(b[0,0,0,1]==bb+1))
    
    a = (np.mgrid[0:10,0:10,0:10,0:10,0:10].T.astype(float)*np.array([1,10,100,1000,10000])).sum(axis=5)
    try:
        b=ng.shred(a,[2,2,3,4,6],[5,4,3,2,1])
        assert(False)
    except:
        pass
    
def test_002_unshred():
    # 1-D case
    a = np.ones([4,5])
    b = ng.unshred(a,[2,])
    assert(b.size == 3*2+5)
    assert(all(b==np.array([1,1,2,2,3,2,3,2,2,1,1])))
    
    # 2-D: a 4x3 array of 5x2 chunks
    a = np.ones([3,4,2,5])
    b = ng.unshred(a,[1,2])
    assert(b.shape == (2*1+2,3*2+5))
    assert(all(b[0,:]==np.array([1,1,2,2,3,2,3,2,2,1,1])))
    assert(all(b[1,:]==np.array([2,2,4,4,6,4,6,4,4,2,2])))
    assert(all(b[2,:]==np.array([2,2,4,4,6,4,6,4,4,2,2])))
    assert(all(b[3,:]==np.array([1,1,2,2,3,2,3,2,2,1,1])))
    
    # 2-D: check scalar step spec
    b = ng.unshred(a,[1])
    
def test_003_hannify():
    # N=1 (2-D chunk) case:  three 4-element chunks
    data = np.ones([3,4])
    data0 = np.copy(data)
    ng.hannify(data)
    assert( data.shape == data0.shape )
    assert( not np.all( data0 == data ) )
    assert( np.all(np.isclose(data,data[:,range(data.shape[1]-1,-1,-1)])))
    assert( np.sum(data[0,:])==2 )
    
    # N=2 (4-D chunk) case: four 3x4-element chunks
    data = np.ones([2,2,3,4])
    data0 = np.copy(data)
    ng.hannify(data)
    assert(data.shape==data0.shape)
    assert( not np.all(data0==data) )
    assert(np.all(np.isclose(
        data,
        data[
            :,
            :,
            range(data.shape[2]-1,-1,-1),
            :
            ],
        atol=1e-9
        ) ) )
    assert(np.all(np.isclose(
        data,
        data[
            :,
            :,
            :,
            range(data.shape[3]-1,-1,-1),
            ],
        atol=1e-9
        ) ) )
    assert( np.isclose( np.sum(data[0,0,:,:]), 3, atol=1e-9 ) )
    
            
            
    
        









    
    
    