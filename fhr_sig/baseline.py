from os.path import dirname, join as pjoin
import scipy.io as sio
import os
import re

import numpy as np
import matplotlib.pyplot as plt


from scipy.ndimage import filters
from IPython.display import display, Markdown, Latex
import warnings
import cv2
import os

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
np.set_printoptions(edgeitems=9)
np.core.arrayprint._line_width = 180
import scipy
DEBUG = False

#*************************************************************************
# Matlab functions
#*************************************************************************
## Matlab Fir2
def mat_fix(x):
    if (np.isscalar(x)):
        if (x >= 0):
            return np.floor(x)
        else:
            return (np.rint(x)+1)
  
    return np.where(x>=0, np.floor(x), np.rint(x)+1)
#*************************************************************************        
def mat_rem(x,y):
    if (y == 0):
        if (type(x)==int):
            return 0
        else: return np.nan
    
    return x % y
#*************************************************************************    
#function b = fir2(n, f, m, grid_n, ramp_n, window)
# grid_n and ramp_n must be integers
def mat_fir2(n, f_in, m_in, grid_n = np.nan, ramp_n= np.nan, window = np.nan):

    f = f_in.copy()
    m = m_in.copy()
    
    ## verify frequency and magnitude vectors are reasonable
    t = len(f);
    if (t<2 or f[0]!=0 or f[-1]!=1 or any(np.diff(f)<0)):
        print ("Error fir2: frequency must be nondecreasing starting from 0 and ending at 1");
        return None
    elif t != len(m):
        print ("Error fir2: frequency and magnitude vectors must be the same length");
        return None

    if not np.isnan(grid_n):
        w=grid_n; 
    if not np.isnan(ramp_n):
        w=ramp_n;
    
    if (np.isnan(window)): 
        window = np.hamming(n+1)

    ## Default grid size is 512... unless n+1 >= 1024
    if np.isnan(grid_n):
        if n+1 < 1024:
            grid_n = 512;
        else:
            grid_n = n+1;
    
    ## ML behavior appears to always round the grid size up to a power of 2
    # nextpow2 (MATLAB Functions) p = nextpow2(A) returns the smallest power of two 
    # that is greater than or equal to the absolute value of A.
    # (That is, p that satisfies 2^p >= abs(A) ).
    # This function is useful for optimizing FFT operations, 
    # which are most efficient when sequence length is an exact power of two.
    grid_n =  np.power(2, int(np.ceil(np.log2(abs(grid_n)))))


    ## Error out if the grid size is not big enough for the window
    if (2*grid_n < n+1):
        print ("Error fir2: grid size must be greater than half the filter order");
        return None
  
    
    if np.isnan(ramp_n): ramp_n = mat_fix (grid_n / 25.)
    
    if DEBUG: print('mat_fir2: grid_n', grid_n, 'ramp_n', ramp_n)
    ## Apply ramps to discontinuities
    if (ramp_n > 0):
        ## remember original frequency points prior to applying ramps
        basef = f.copy()
        basem = m.copy()

        ## separate identical frequencies, but keep the midpoint

        idx = np.argwhere(np.diff(f)==0).flatten()
        if DEBUG: print('mat_fir2: idx', idx)
        for i in idx:
            f[i] = f[i] - ramp_n/grid_n/2.;
        if DEBUG: print('mat_fir2: f after idx ', f)
         
        for i in idx:            
            f[i+1] = f[i+1] + ramp_n/grid_n/2.;

        f = np.concatenate([f, [basef[i] for i in idx]])
        ## make sure the grid points stay monotonic in [0,1]
        f[f<0] = 0;
        f[f>1] = 1;
        f = np.unique(np.concatenate([f, [basef[i] for i in idx]]))

        ## preserve window shape even though f may have changed
        m = scipy.interpolate.interp1d(basef, basem)(f)

    
    ## interpolate between grid points
    grid = scipy.interpolate.interp1d(f,m)(np.linspace(0,1,grid_n+1))
    
    ## Transform frequency response into time response and
    ## center the response about n/2, truncating the excess

    if (mat_rem(n,2) == 0):
        x = np.concatenate([grid , grid[grid_n-1:0:-1]])
        b = np.fft.ifft(np.concatenate([grid , grid[grid_n-1:0:-1]]))
        mid = (n+1)/2.;
        b = np.real (np.concatenate([ b[len(b) - int(np.floor(mid)) :] , b[0:int(np.ceil(mid))] ]));
    else:
        ## Add zeros to interpolate by 2, then pick the odd values below.
        b = np.fft.ifft(np.concatenate([grid , np.zeros((grid_n*2)) , grid[grid_n-1:0:-1]]));
        b = 2 * np.real(np.concatenate([ b[len(b)-n::2] , b[1:n+2:2]]));  
    
    ## Multiplication in the time domain is convolution in frequency,
    ## so multiply by our window now to smooth the frequency response.
    ## Also, for matlab compatibility, we return return values in 1 row
    b = np.multiply(b , window)
    return b
#*************************************************************************
def mat_decimate(x, q, n = 0): #function y = decimate(x, q, n, ftype)

    if (n==0):
        n = 8


    [b, a] = signal.cheby1 (n, 0.05, 0.8/q);
    y = signal.filtfilt (b, a, x,  padtype = 'odd', padlen=3*(max(len(b),len(a))-1))

    y = y[0:len(x):q];
    return y
#*************************************************************************    
def mat_interp1(xp,yp,xf):
    lin = scipy.interpolate.interp1d(xp, yp, 'linear')(xf)
    return lin
    
#*************************************************************************
def mat_fir1(n, w):

    if isinstance(w, (list, tuple, set, np.ndarray)):
        print('mat_fir1: w is an array: ', w)
        return None
    
    ## Assign default window, filter type and scale.
    ## If single band edge, the first band defaults to a pass band to
    ## create a lowpass filter.  If multiple band edges, the first band
    # defaults to a stop band so that the two band case defaults to a
    ## band pass filter.  Ick.
    window  = [];
    scale   = 1;
    ftype   = 1

    ## build response function according to fir2 requirements
    bands = 2
    f = np.zeros((2*bands));
    f[0] = 0; 
    f[-1]=1;
    if DEBUG: print('mat_fir1:  f', f)
    
    f[1:2*bands-1:2] = w #f(2:2:2*bands-1) = w;
    if DEBUG: print('mat_fir1:  f', f)
    f[2:2*bands-1:2] = w #f(3:2:2*bands-1) = w;
    if DEBUG: print('mat_fir1:  f', f)
    
    m = np.zeros((2*bands));
    m[0:2*bands:2] = mat_rem(np.arange(1, bands+1) , 2)   #m(1:2:2*bands) = rem([1:bands]-(1-ftype),2);
    m[1:2*bands:2] = m[0:2*bands:2]   #m(2:2:2*bands) = m(1:2:2*bands);
    if DEBUG: print('mat_fir1:  m', m)
    
    ## Increment the order if the final band is a pass band.  Something
    ## about having a nyquist frequency of zero causing problems.
    if mat_rem(n,2)==1 and m[-1]==1:
        print("mat_fir1: n must be even for highpass and bandstop filters. Incrementing.");
        n = n+1;
        #M = 1
        #window = [window; window];
    if DEBUG: print('mat_fir1:  n', n)
    #print('mat_fir1:  M', M)

    # mat_fir2(n, f_in, m_in, grid_n = np.nan, ramp_n= np.nan, window = np.nan):
    b = mat_fir2(n, f, m, ramp_n = 2)
    if DEBUG: print('mat_fir1:  b', b)
    
      ## normalize filter magnitude
    if scale == 1:
        ## find the middle of the first band edge
        ## find the frequency of the normalizing gain
        if m[0] == 1:
            ## if the first band is a passband, use DC gain
            w_o = 0;
        elif f[3] == 1:
            ## for a highpass filter,
            ## use the gain at half the sample frequency
            w_o = 1;
        else:
            ## otherwise, use the gain at the center
            ## frequency of the first passband
            w_o = f[2] + (f[3]-f[2])/2;
    if DEBUG: print('mat_fir1:  w_o', w_o)

    renorm = 1./np.abs(np.polyval(b, np.exp( complex(0,-1)*np.pi*w_o)));
    if DEBUG: print('mat_fir1:  np.polyval(b, np.exp( complex(0,-1)*np.pi*w_o))', np.polyval(b, np.exp( complex(0,-1)*np.pi*w_o)))
    if DEBUG: print('mat_fir1:  renorm', renorm)
    
    b = renorm*b;
    return b
#*************************************************************************    
def mat_fftfilt (b, x, n = np.nan):

    ## If N is not specified explicitly, we do not use the overlap-add
    ## method at all because loops are really slow.  Otherwise, we only
    ## ensure that the number of points in the FFT is the smallest power
    ## of two larger than N and length(b).  This could result in length
    ## one blocks, but if the user knows better ...

    # make sure shape is (len,1)
    #x = x.reshape((len(x),1))
    #b = b.reshape((len(b),1))
    r_x = len(x)
    c_x = 1
    r_b = 1
    c_b = len(b)

    l_b = r_b * c_b;
    print('b', b.shape,b)
    #b = b.reshape ((b, l_b, 1));

    if (np.isnan(n)):
        ## Use FFT with the smallest power of 2 which is >= length (x) +
        ## length (b) - 1 as number of points ...
        n = int(2 ** np.ceil(np.log2(abs(r_x + l_b - 1))))
        print('n', n)
        print('b', b)
        B = np.fft.fft (b, n)
        #B = B.reshape((len(B), 1))
        #B = np.repeat(B, c_x, axis=1)
        print('B', B.shape, B)
        print('np.fft.fft (x, n)', np.fft.fft (x, n), np.fft.fft (x, n).shape)
        y = np.fft.ifft (np.multiply(np.fft.fft (x, n) ,B))#B(:, ones (1, c_x)));
        print('y', y.shape, y)
  
    else:
        #n = 2 ^ nextpow2 (max ([n, l_b]));
        n = int(2 ** np.ceil(np.log2(abs(max ([n, l_b])))))
        L = n - l_b + 1;
        B = np.fft.fft (b, n)#.reshape((len(B), 1))
        #B = np.repeat(B, c_x, axis=1)

        #B = B(:, ones (c_x,1));

        R = int(np.ceil (r_x / L))
        y = np.zeros ((r_x, c_x));
        print('R', R)
        for r in range(R): #r = 1:R
            lo = (r - 1) * L + 1;
            hi = min (r * L, r_x);
            tmp = np.zeros ((n, c_x));
            print('tmp', tmp.shape)
            print('x[lo:hi+1,:]', x[lo:hi+1,:], x[lo:hi+1,:].shape)
            tmp[0:(hi-lo+1+1),:] = x[lo:hi+1,:];
            tmp = np.fft.ifft (np.multiply(np.fft.fft (tmp) , B));
            hi  = min (lo+n-1, r_x);
            y[lo:hi+1,:] = y[lo:hi+1,:] + tmp[0:(hi-lo+1+1),:];

    print('**y', y.shape,y)
    y = y[0:r_x]
    print('y', y.shape,y)

    ## Final cleanups:

    ## - If both b and x are real, y should be real.
    ## - If b is real and x is imaginary, y should be imaginary.
    ## - If b is imaginary and x is real, y should be imaginary.
    ## - If both b and x are imaginary, y should be real.
    xisreal = np.all ( np.isreal(x) == 1)
    xisimag = np.all ( np.isreal(x) == 0)
    print('xisreal', xisreal)
    print('xisimag', xisimag)

    if (np.all ( np.isreal(b) == 1)):
        if (xisreal):
            #xisreal = 1 xisimag = 0
            y = np.real (y );
        else:

            #xisreal = 0 xisimag = 1
            y = np.complex (np.real (y ) * 0, np.image (y ));

    elif (np.all ( np.isreal(x) == 0)):
        if (xisimag):
            #xisreal = 0 xisimag = 1
            y  = np.real (y);
        else:
            #xisreal = 1 xisimag = 0
            y  = np.complex (np.real (y ) * 0, np.image (y ));
            

    ## - If both x and b are integer in both real and imaginary
    ##   components, y should be integer.
    if (not np.any (b - mat_fix (b))):
        idx = np.argwhere(not np.any (x - mat_fix (x))).flatten()
        
        y [idx] = np.round (y [idx]);
        
    return y
#*************************************************************************
def mat_fftfilt (b, x, n = np.nan):

    ## If N is not specified explicitly, we do not use the overlap-add
    ## method at all because loops are really slow.  Otherwise, we only
    ## ensure that the number of points in the FFT is the smallest power
    ## of two larger than N and length(b).  This could result in length
    ## one blocks, but if the user knows better ...

    # make sure shape is (len,1)
    #x = x.reshape((len(x),1))
    #b = b.reshape((len(b),1))
    r_x = len(x)
    c_x = 1
    r_b = 1
    c_b = len(b)

    l_b = r_b * c_b;
    if DEBUG: print('mat_fftfilt: b', b.shape,b)
    #b = b.reshape ((b, l_b, 1));

    if (np.isnan(n)):
        ## Use FFT with the smallest power of 2 which is >= length (x) +
        ## length (b) - 1 as number of points ...
        n = int(2 ** np.ceil(np.log2(abs(r_x + l_b - 1))))
        B = np.fft.fft (b, n)
        y = np.fft.ifft (np.multiply(np.fft.fft (x, n) ,B))#B(:, ones (1, c_x)));
  
    else:
        #n = 2 ^ nextpow2 (max ([n, l_b]));
        n = int(2 ** np.ceil(np.log2(abs(max ([n, l_b])))))
        L = n - l_b + 1;
        B = np.fft.fft (b, n)#.reshape((len(B), 1))
        R = int(np.ceil (r_x / L))
        y = np.zeros ((r_x, c_x));
        for r in range(R): #r = 1:R
            lo = (r - 1) * L + 1;
            hi = min (r * L, r_x);
            tmp = np.zeros ((n, c_x));
            tmp[0:(hi-lo+1+1),:] = x[lo:hi+1,:];
            tmp = np.fft.ifft (np.multiply(np.fft.fft (tmp) , B));
            hi  = min (lo+n-1, r_x);
            y[lo:hi+1,:] = y[lo:hi+1,:] + tmp[0:(hi-lo+1+1),:];

    y = y[0:r_x]

    ## Final cleanups:

    ## - If both b and x are real, y should be real.
    ## - If b is real and x is imaginary, y should be imaginary.
    ## - If b is imaginary and x is real, y should be imaginary.
    ## - If both b and x are imaginary, y should be real.
    xisreal = np.all ( np.isreal(x) == 1)
    xisimag = np.all ( np.isreal(x) == 0)


    if (np.all ( np.isreal(b) == 1)):
        if (xisreal):
            #xisreal = 1 xisimag = 0
            y = np.real (y );
        else:

            #xisreal = 0 xisimag = 1
            y = np.complex (np.real (y ) * 0, np.image (y ));

    elif (np.all ( np.isreal(x) == 0)):
        if (xisimag):
            #xisreal = 0 xisimag = 1
            y  = np.real (y);
        else:
            #xisreal = 1 xisimag = 0
            y  = np.complex (np.real (y ) * 0, np.image (y ));
            

    ## - If both x and b are integer in both real and imaginary
    ##   components, y should be integer.
    if (not np.any (b - mat_fix (b))):
        idx = np.argwhere(not np.any (x - mat_fix (x))).flatten()
        
        y [idx] = np.round (y [idx]);
        
    return y
#*************************************************************************
def mat_interp1(x, y, xnew, kind='linear'):
    
    if kind == 'spline':
        kind='cubic'
    if not kind in ['linear', 'cubic']:
        print('mat_interp1: ', kind, ' Not supported')
        return None
    f = scipy.interpolate.interp1d(x, y, kind)
    return f(xnew)
#*************************************************************************
def  mat_interp(x, q, n = 4, Wc = 0.5):

    if DEBUG: print('mat_interp:\n', '** mat_interp **')
    if q != mat_fix(q):
        print("decimate only works with integer q.")
        return np.nan

    y = np.zeros(len(x)*q+q*n+1)
    if DEBUG: print('mat_interp:len(x)*q+q*n+1', len(x)*q+q*n+1)

    y[0:len(x)*q:q] = x;
    if DEBUG: print('mat_interp:------------xxxxxxxxxxxxxxxxxxx------  y', y)

    b = mat_fir1(2*q*n+1, Wc/q);
    y=q*mat_fftfilt(b, y);
  
    # ? y(1:q*n+1) = [];  # adjust for zero filter delay

    return y[q*n+1:]
#*************************************************************************    
#*************************************************************************    
# Fhr functionality for FHR baseline calculation
#*************************************************************************
def fhr_avgsubsamp(x,factor):
    
    y=np.zeros((int(np.floor(len(x)/factor))));
    for i in range(len(y)):
        y[i]=np.mean(x[i*factor+1:(i+1)*factor]);
    return y
    
from scipy import signal
#*************************************************************************
def fhr_butterfilt(data, srate, f1, f2, order = 6, zeroPhase = 1):
    
    if DEBUG: print('fhr_butterfilt: order', order)
    assert(zeroPhase==1)
    if ((f1>0) and (f2==0)):
        b1,a1 = signal.butter(order, 2*f1/srate,'highpass', output='ba')

    elif (f1==0 and f2>0):
        b1,a1 = signal.butter(order, 2*f2/srate,'lowpass', output='ba')
        
    elif (f1>0 and f2>0 and f2<f1):
        b1,a1 = signal.butter(order, [2.*f2/srate ,2.*f1/srate],btype='band', output='ba')
            
    elif (f1>0 and f2>0 and f2>f1):
                
        z, p, k = signal.butter(1, [2.*f1/srate, 2.*f2/srate], btype='bandpass', output='zpk')
        b1, a1 = signal.zpk2tf(z, p, k)

    if DEBUG: print('fhr_butterfilt:a1', a1)
    if DEBUG: print('fhr_butterfilt:b1', b1)

    if zeroPhase:
        
        nfilt = max(len(b1),len(a1))

        a1 = np.pad(a1, (0,nfilt-len(a1)), mode='constant',)
        b1 = np.pad(b1, (0,nfilt-len(b1)), mode='constant',)
        data=signal.filtfilt(b1,a1,data, padtype = 'odd', padlen=3*(max(len(b1),len(a1))-1))
        
    #else:
    #    
    #    data=signal.filter(b1,a1,data)
    #    print('data',data)
        
    return data
#*************************************************************************
def fhr_enveloppe(x,srate,f0,f1, deb = False): #function [y,x]=

    fftx=np.fft.fft(x);
    if (deb):
        print('fhr_enveloppe: fftx', fftx.shape, fftx)#[:10])
        
    siglen=len(x)/srate;
    if (deb):
        print('*********** np.round(f0*siglen+1)', np.round(f0*siglen+1))
        print('*********** (f0*siglen+1)', (f0*siglen+1))
    firstsamp=int(np.round(f0*siglen+1))
    lastsamp=int(np.round(f1*siglen+1))
    if (deb):
        print('fhr_enveloppe: firstsamp, lastsamp', firstsamp, lastsamp)
        print('(lastsamp-firstsamp) %2', (lastsamp-firstsamp) %2)
    ffty=np.zeros(x.shape,dtype=complex);

    iend = int((lastsamp-firstsamp-1)/2)
    if ((lastsamp-firstsamp) % 2 ==0): iend += 1
    if (deb):
        print('fhr_enveloppe: iend, from', iend, fftx[firstsamp-1:firstsamp-1+iend])
        print('fhr_enveloppe: iend, to', ffty[-iend:])
        print('fhr_enveloppe: iend', iend)
    
    #if(mod(lastsamp-firstsamp,2)==1) 
    #    ffty([end-(lastsamp-firstsamp-3)/2:end 1:(3+lastsamp-firstsamp)/2])=fftx(firstsamp:lastsamp);
    # else
    #   ffty([end-(lastsamp-firstsamp-2)/2:end 1:(2+lastsamp-firstsamp)/2])=fftx(firstsamp:lastsamp);
    if (lastsamp-firstsamp) %2 ==1:

        if (firstsamp-1+int((lastsamp-firstsamp-1)/2) >= firstsamp-1):

            ffty[len(ffty) - 1 - int((lastsamp-firstsamp-3)/2):] = fftx[firstsamp-1:iend]

        if (lastsamp > firstsamp-1+int((lastsamp-firstsamp-1)/2)):
            #ffty[:int((3+lastsamp-firstsamp)/2)] = fftx[firstsamp-1+len(ffty) - int((lastsamp-firstsamp-3)/2):lastsamp]
            ffty[:int((3+lastsamp-firstsamp)/2)] = fftx[firstsamp-1+iend:lastsamp]

        
    else:
        
        ffty[len(ffty)-1 - int((lastsamp-firstsamp-2)/2):] = fftx[firstsamp-1:firstsamp-1+len(ffty[len(ffty)-1 - int((lastsamp-firstsamp-2)/2):])]
        ffty[:int((2+lastsamp-firstsamp)/2)] = fftx[lastsamp-int((2+lastsamp-firstsamp)/2):lastsamp]


        

    fftx[:(firstsamp-1)] = 0.     #fftx[:(firstsamp-1)] = 0.

    fftx[lastsamp:len(fftx)-lastsamp+1] = 0

    fftx[len(fftx)-firstsamp+1:] = 0

    
    if (deb):
        print('fhr_enveloppe: before ifft fftx[0:10]', fftx.size, fftx)#[:10])
    x=np.fft.ifft(fftx);
    y=2*abs(np.fft.ifft(ffty));
    
    return y, x

#*************************************************************************
def fhr_medgliss(X,win,coef,decim,X2 = np.nan,p2 = np.nan,c = np.nan, nargs=4): # [Y,mp]=medgliss(X,win,coef,decim,X2,p2,c)

    if DEBUG: print('+++medgliss',X,win,coef,decim,X2,p2,c )
    
    Xd=fhr_butterfilt(X,240,0,240/2.2/decim,8,1);
    
    Xd=Xd[0::decim]
    if (nargs>4):
        X2=X2[0::decim]

    coefd= mat_decimate(coef,decim);
    coefd=np.where(coefd > 0, coefd, 0)#coefd.*(coefd>0);
    midwin=(len(win)-1)/2;
    Yd=np.zeros((Xd.shape));

    mintolerated=np.zeros(len(Xd));
    maxtolerated=255*np.ones((len(Xd)));
    
    if DEBUG: print('fhr_medgliss: 10*240/decim', 10*240/decim, '240/2/decim', 240/2/decim, 'len(Xd)-10*240/decim', len(Xd)-10*240/decim)
    if DEBUG: print('fhr_medgliss: len(Xd)', len(Xd), '-10*240/decim', -10*240/decim)
    for i in range(0,int(240/2/decim), len(Xd)-int(10*240/decim)):
        twin= np.arange(i,i+10*240./decim).astype(int)
        
        mi=min(Xd[twin.astype(int)]);
        ma=max(Xd[twin.astype(int)]);

        mintolerated[twin[mintolerated[twin]<=mi]]=mi;
        maxtolerated[twin[maxtolerated[twin]<=ma]]=ma;
    
    mp = np.zeros((len(Xd)))
    
    for i in range(len(Xd)):
        if(i<midwin):
            mwm=i;
            if DEBUG: 
                print('fhr_medgliss: np.floor((midwin-mwm)/2)', np.floor((midwin-mwm)/2),'len(Xd)-(i+1)',len(Xd)-(i+1) )
                print('fhr_medgliss: max of', mwm, np.floor((midwin-mwm)/2))
                print('fhr_medgliss: np.max(mwm, np.floor((midwin-mwm)/2))',max(mwm, np.floor((midwin-mwm)/2)))
            mwp=min(max(mwm, np.floor((midwin-mwm)/2)), len(Xd)-(i+1));
        elif(len(Xd)-(i+1)<midwin):#??? -i + 1
            mwp=len(Xd)-(i+1);
            mwm=min(max(mwp, np.floor((midwin-mwp)/2)), i);
        else:
            mwm=midwin;
            mwp=midwin;
        
        if DEBUG: print('fhr_medgliss:i', i, 'mwm', mwm, 'mwp', mwp)
        points=np.arange(i-mwm,i+mwp+1).astype(int)
        if DEBUG: print('fhr_medgliss: ----------------8888888888888---------------points\n',points)
        if DEBUG: print('fhr_medgliss: ----------------8888888888888---------------Xd\n',Xd)
        Xpoints=Xd[points];
        if DEBUG: print('fhr_medgliss: ----------------8888888888888---------------Xpoints\n',Xpoints)
        if DEBUG: 
            print('fhr_medgliss: win[209:210]',win.shape, win[209:210])
            print('fhr_medgliss: coefd[points]', coefd[points])
            print('fhr_medgliss: midwin-mwm+1:midwin+mwp+1', midwin-mwm+1,midwin+mwp+1)
            print('fhr_medgliss: win[midwin-mwm+1:midwin+mwp+1]',win[int(midwin-mwm+1):int(midwin+mwp+1+1)])

        tmp = np.where((Xpoints>=mintolerated[i]) & (Xpoints<=maxtolerated[i]), 1, 0)
        if DEBUG: print('fhr_medgliss: coefd[points].shape', coefd[points].shape, 'win[int(midwin-mwm+1):int(midwin+mwp+1+1)].shape', 
              win[int(midwin-mwm):int(midwin+mwp+1)].shape)
        coefwin=np.multiply(coefd[points],win[int(midwin-mwm):int(midwin+mwp+1)])
        if DEBUG: print('fhr_medgliss: coefwin', coefwin)
        coefwin=np.multiply(tmp,coefwin)
        if DEBUG: print('fhr_medgliss: coefwin', coefwin)
                             
        s=coefwin.sum()
        if DEBUG: print('fhr_medgliss: ---------------------s', s)
        scoef=win[int(midwin-mwm):int(midwin+mwp+1)].sum()
        if DEBUG: print( 'fhr_medgliss: scoef', scoef)

        if nargs>4:
            if DEBUG: print('fhr_medgliss: i, Xpoints, X2[i]',i, Xpoints, X2[i])
            Xpoints= np.concatenate([Xpoints , np.array([X2[i]])])
            if DEBUG: print('fhr_medgliss: coefwin, np.array([np.max(0,c*p2[i]*scoef-s)])',coefwin,c*p2[i]*scoef-s)
            if DEBUG: print('fhr_medgliss:', max(0,c*p2[i]*scoef-s) )
            coefwin= np.concatenate([coefwin, np.array([max(0,c*p2[i]*scoef-s)])]);
            s=coefwin.sum();
        
        #[p,order]=sort(Xpoints);
        
        order = np.argsort(Xpoints)
        if DEBUG: print('fhr_medgliss:order', order)
        p = np.sort(Xpoints)
        if DEBUG: print('fhr_medgliss: p', p)
        if DEBUG: print('fhr_medgliss: coefwin[order]', coefwin[order])
        if DEBUG: print('fhr_medgliss: coefwin', coefwin)
        # find(X,1,'first') finds the first  index corresponding to nonzero elements.
        #Yd[i]=p(find(cumsum(coefwin(order))>=s/2,1,'first'));
        #t = array([1, 1, 1, 2, 2, 3, 8, 3, 8, 8])
        #nonzero(t == 8)[0][0]
        if DEBUG: print('fhr_medgliss: np.nonzero(np.cumsum(coefwin[order])>=s/2)', np.nonzero(np.cumsum(coefwin[order])>=s/2))
        Yd[i]  = p[np.nonzero(np.cumsum(coefwin[order])>=s/2)[0][0]]
        if DEBUG: print('===============i', i,'Yd', Yd)
        if DEBUG: print('fhr_medgliss: i', i, 'Yd[i]',Yd[i])
        if DEBUG: print('fhr_medgliss: ************** s/scoef', s/scoef)
        mp[i]=s/scoef;

    if DEBUG: 
        print('fhr_medgliss: mp', mp)
        print('fhr_medgliss: Yd', Yd)
        print('fhr_medgliss: decim', decim)
    Y=mat_interp(Yd,decim);
    Y=Y[:len(X)];

    mp=mat_interp(mp,decim);
    mp=mp[:len(X)];

    return Y, mp
#*************************************************************************    
# strate = 240

def fhr_aamwmfb(FHRi, deb = DEBUG): #function [baseline,accelerations,decelerations,falseAcc,falseDec]=aamwmfb(FHRi)

    st = 240
    
    FHR1=fhr_butterfilt(FHRi,240,0,1,1);
    
    FHR2=fhr_butterfilt(FHRi,240,0,2,1);
    
    FHR4=fhr_butterfilt(FHRi,240,0,4,1);
    
    FHR8=fhr_butterfilt(FHRi,240,0,8,1);
    
    FHR16=fhr_butterfilt(FHRi,240,0,16,1);
    

    fcut=[0 ,1 ,3 ,7]
    fdat=np.zeros((3,2, len(FHRi)), dtype=np.float64)

    for j in range(3):#j=1:3
        if DEBUG: print('aamwmfb: ',fcut[j],fcut[j+1])
        fdat[j,0]=fhr_butterfilt(FHRi,240,fcut[j],fcut[j+1],1);
        if DEBUG: print('aamwmfb: --- j',j)
        if DEBUG: print('aamwmfb: fdat[j,0]=fhr_butterfilt(FHRi,240,',fcut[j],',',fcut[j+1],',',1,')')
        if DEBUG: print('aamwmfb: ',fdat[j,0])
        
        if DEBUG: print('aamwmfb: ',fdat[j,1][1:].shape)
        fdat[j,1]=np.concatenate([np.zeros((1)), fdat[j,0][1:]-fdat[j,0][:-1]])*240.;

    # 47529       4 size of t
    t = np.zeros((len(FHRi), 4), dtype=np.float64)
    if DEBUG: print('aamwmfb: ',fdat[0,1].size)
    t[:,0] = np.abs(fdat[0,1])
    if DEBUG: print('aamwmfb: t[:,0]', t[:,0])
    x, y = fhr_enveloppe(fdat[0,1],st,0,2*fcut[1])
    t[:,1] = x
    x, y = fhr_enveloppe(fdat[1,1],st,0,2*fcut[2])
    t[:,2] = x
    x, y = fhr_enveloppe(fdat[2,1],st,0,2*fcut[3])
    t[:,3] = x
    
    if DEBUG: print('aamwmfb: t',t.shape, '\n***',t)
    Q= np.array([-2.4744 ,   0.0266,    0.0413  ,  0.0105   , 0.0036])
    P=1-np.divide(np.exp(Q[0]+t.dot(Q[1:])),(1+np.exp(Q[0]+t.dot(Q[1:]))));
    

    distancecoef=np.concatenate([np.linspace(0,1, num=200), np.ones((1)), np.linspace(1,0, num=200)])
    if DEBUG: print('aamwmfb: +++distancecoef', distancecoef.shape)
    if DEBUG: print('aamwmfb: +++P', distancecoef.shape)
    [bl1,mp1]=fhr_medgliss(FHR2,distancecoef,P,24);
                       
    P2=np.multiply(P, np.divide(np.exp(3.21-0.19*np.abs(FHR1-bl1)),(1+np.exp(3.21-0.19*np.abs(FHR1-bl1)) )))
    [bl2, tmpo] = fhr_medgliss(FHR2,distancecoef**2, P2, 24, bl1, mp1, 0.1, nargs=7);
    P3=np.multiply(P,np.divide(np.exp(2.5-0.19*np.abs(FHR4-bl2)),(1+np.exp(2.5-0.19*np.abs(FHR4-bl2)) )));
    [bl3, tmpo]=fhr_medgliss(FHR4,np.power(distancecoef,4) ,P3,24,bl2,mp1,0.1, nargs=7);
    P4=np.multiply(P,np.divide(np.exp(2-0.19*np.abs(FHR8-bl3)),(1+np.exp(2-0.19*np.abs(FHR8-bl3)) )))
    [bl4, tmpo]=fhr_medgliss(FHR8,np.power(distancecoef,8),P4,24,bl3,mp1,0.1, nargs=7);
    P5=np.multiply(P,np.divide( np.exp(1.5-0.19*np.abs(FHR16-bl4)),(1+np.exp(1.5-0.19*np.abs(FHR16-bl4)) )));
    [bl5, tmpo]=fhr_medgliss(FHR16,np.power(distancecoef,16),P5,24,bl4,mp1,0.1, nargs=7);
    P6=np.multiply(P, np.divide(np.exp(1-0.19*np.abs(FHR16-bl5)),(1+np.exp(1-0.19*np.abs(FHR16-bl5)) )))
    [bl6, tmpo]=fhr_medgliss(FHR16,np.power(distancecoef,16),P6,24,bl5,mp1,0.1, nargs=7);
    baseline=bl6;


    return baseline
        
