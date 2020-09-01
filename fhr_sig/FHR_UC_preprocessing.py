#!/usr/bin/env python
# coding: utf-8

# # FHR UC signal Processing
# 
# ## Python Version of https://github.com/utsb-fmm/FHR
#         
# The FHR and UC are both sampled in 4 Hz.



from os.path import dirname, join as pjoin
import scipy.io as sio
import os

import numpy as np
import pandas as pd

import time
import cv2
import os

import matplotlib.pyplot as plt

np.set_printoptions(edgeitems=9)
np.core.arrayprint._line_width = 180
pd.options.display.max_columns = 100

def read_matlab_fhr(fhr_name, train=1, mdir = '.\\octave\\FHR-master\\FHR-master\\'):
    
    f = mdir +'train_csv\\'+fhr_name+'.csv'
    if train==0:
        f = mdir +'test_csv\\'+fhr_name+'.csv'
    f = pd.read_csv(f, names=['UC', 'FHR', 'rel'], header=None)
    return f.FHR.values
    


# 


def read_matlab_fhr_df(fhr_name, train=1, mdir = '.\\octave\\FHR-master\\FHR-master\\'):
    
    f = mdir +'train_csv\\'+fhr_name+'.csv'
    if train==0:
        f = mdir +'test_csv\\'+fhr_name+'.csv'
    f = pd.read_csv(f, names=['UC', 'FHR', 'rel'], header=None)
    return f
    


# 

DEBUG = False 
def set_debug(deb):
    DEBUG = deb


# 


def get_first_last_non_zero(fhr):
    s = 0
    e = len(fhr)-1
    
    if (fhr[0]==0): 
        s = (ft!=0).argmax(axis=0)
    if (fhr[-1]==0):
        e = np.max(np.nonzero(fhr))
    return s, e
        


# 


def get_non_zero_seqs(fhr):
        
    p = np.where(fhr>0, 1, 0).astype(int)
    s = np.array([1]+ list(np.diff(p)))
    sind = np.where(s==1)[0][1:]
    eind = np.where(s==-1)[0]-1
        
    if DEBUG: print(len(sind))
    
    if DEBUG: print(len(eind))
    
    return sind, eind


# 


def remove_small_particles(fhr, n_sec = 5, Hz = 4):
    
    sind, eind = get_non_zero_seqs(fhr)
    
    if (fhr[-1]>0):
        sind = sind[:-1]
    if (fhr[0]>0):
        eind = eind[1:]

        
    for i in range(len(sind) - 1):
        s1 = sind[i]
        e1 = sind[i+1]
        
        subfhr = np.where(fhr[s1:e1] > 0, 1, 0).sum()
        
        if (subfhr < n_sec * Hz):
            
            if DEBUG: print(' zeroing from: ', s1, ' to: ', e1, subfhr)
            fhr[s1:e1] = 0
    return fhr


# In[15]:


def remove_small_pikes(fhr, gap_sec = 30, Hz = 4, gap=25):

    ft = fhr.copy()
    ft = np.where(fhr > 220, 0, fhr)
    ft = np.where(fhr < 50, 0, fhr)
    
    #snz,enz = get_first_last_non_zero(fhr)
    ifirst = np.argmax(ft>0)
    ft[0:ifirst] = ft[ifirst]
    ilast = np.max(np.nonzero(ft))
    ft[ilast:] = ft[ilast]
    
    if DEBUG: print(ifirst,ilast)
        
    #ft = ft[ifirst:ilast+1].copy()
        
    sind, eind = get_non_zero_seqs(ft)   
    
    if DEBUG:print(len(sind))    
    for i in range(len(sind)-1):
        
        e_pre = eind[i]
        s = sind[i]
        e = eind[i+1]
        s_post = sind[i+1]

        subfhr = e-s + 1
        if DEBUG:print(subfhr, gap_sec * Hz)
        
        if (subfhr < gap_sec * Hz):
            
            if DEBUG:print('diff', ft[s] - ft[e_pre], ft[e] - ft[s_post])
            if (ft[s] - ft[e_pre] < -gap and ft[e] - ft[s_post] < -gap):
                if DEBUG: print(' subfhr < gap_sec * Hz ',e_pre, 'zeroing from: ', s, 'to: ', e+1, s_post, subfhr)
                
                ft[s:e+1] = 0
            if (ft[s] - ft[e_pre] > gap and ft[e] - ft[s_post] > gap):
                if DEBUG: print(' subfhr < gap_sec * Hz ',e_pre, 'zeroing from: ', s, 'to: ', e+1, s_post, subfhr)
                ft[s:e+1] = 0
            
    #fhr[ifirst:ilast+1] = ft
    return ft


# ### Linerar interpolation
# * linear_interpolate_nan - close nan sequences with linear interpolation
# * make_small_gaps_nan - make small gaps we want to interpolate to nan

def linear_interpolate_nan(x):
    not_nan = np.logical_not(np.isnan(x))
    indices = np.arange(len(x))
    x = np.interp(indices, indices[not_nan], x[not_nan])
    return x


# 


def make_small_gaps_nan(fhr, gap_sec = 30, Hz = 4):
    
    #snz,enz = get_first_last_non_zero(fhr)
    #ft = fhr[snz:enz+1].copy()
    
    ft = fhr.copy()
    sind, eind = get_non_zero_seqs(ft)

    for i in range(len(sind)):
        
        sz = eind[i]
        ez = sind[i]
        
        subfhr = ez - sz
        
        if (subfhr < gap_sec * Hz):
            
            if DEBUG: print(sz, ez, subfhr)
            ft[sz+1:ez] = np.nan
            
    
    return ft


# ## Summerize preprocessing 


def preprocess(fhr, uc, remove_padding_zeros = True):

    fhr = np.where((fhr > 220) | (fhr < 50), 0, fhr)
    
    ifirst = 0;
    ilast = len(fhr)

    ifirst = np.argmax(fhr>0)
    ilast = np.max(np.nonzero(fhr))
    
    if (remove_padding_zeros):
        ifirst = np.argmax(fhr>0)
        ilast = np.max(np.nonzero(fhr))
        print(ifirst, ilast)
    else:
        if (fhr[0] == 0):
            fhr[0] = 120
        if (fhr[ilast-1] == 0):
            fhr[ilast-1] = 120

    
    fhr = remove_small_particles(fhr)
    fhr = remove_small_pikes(fhr)
    fhr = fhr[ifirst:ilast]
    
    fhr = remove_small_particles(fhr)
    fhr = remove_small_pikes(fhr)

    # fhr = make_small_gaps_nan(fhr) in matlab toolbox they close any gap
    
    fhr[fhr==0] = np.nan
    fhr_nan = fhr.copy()
    fhr = linear_interpolate_nan(fhr_nan)
    
        
    uc = uc[ifirst:ilast]
    uc = remove_small_particles(uc)
    uc = remove_small_pikes(uc, gap=10)
    uc[uc==0] = np.nan
    uc_nan = uc.copy()
    uc = linear_interpolate_nan(uc_nan)

    return fhr, fhr_nan, uc, ifirst, ilast


# 


def preprocess_df_fhr(df, col_good_signal = None):
    
    fhr = df['FHR'].values
    uc = df.UC.values
    
    if (col_good_signal != None):
    
        df['rel_fhr']= df['FHR']*df[col_good_signal]
        fhr = df['rel_fhr'].values

    fhr, fhr_nan, uc, ifirst, ilast = preprocess(fhr, uc)
    return fhr, fhr_nan, uc, ifirst, ilast


# 


def load_matlab_fhr_preprocess(fname, train=1):
    
    f = read_matlab_fhr_df(fname, train)

    fhr, fhr_nan, uc, ifirst, ilast = preprocess_df_fhr(f, col_good_signal = 'rel')
    return fhr, fhr_nan, uc, ifirst, ilast


def get_ctu_chb_sig(rec_id=1004, CSV_DIR = '.\\ctu_csv_data\\'):
    
    
    fname = 'df_sig_'+str(rec_id)+'.csv'
    print('loading ', CSV_DIR + 'csv_recs\\'+fname)
    df = pd.read_csv(CSV_DIR + 'csv_recs\\'+fname)
    
    fhr, fhr_nan, uc, ifirst, ilast = preprocess_df_fhr(f)
    return fhr, fhr_nan, uc, ifirst, ilast
    

