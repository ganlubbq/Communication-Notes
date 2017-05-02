
"""
ECE 4625/5625 Project 1 Support Module

Mark Wickert February 2015
"""
from __future__ import division #provides float div as x/y and int div as x//y
from matplotlib import pylab
#from matplotlib import mlab 
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import ssd


class rate_change(object):
    """
    A simple class for encapsulating the upsample/filter and
    filter/downsample operations used in modeling a comm
    system. Objects of this class will hold the required filter
    coefficients once an object is instantiated.
    
    Mark Wickert February 2015
    """
    def __init__(self,M_change = 12,fcutoff=0.9,N_filt_order=8,ftype='butter'):
        """
        Object constructor method
        """
        self.M = M_change # Rate change factor M or L
        self.fc = fcutoff*.5 # must be fs/(2*M), but scale by fcutoff
        self.N_forder = N_filt_order
        if ftype.lower() == 'butter':
            self.b, self.a = signal.butter(self.N_forder,2/self.M*self.fc)
        elif ftype.lower() == 'cheby1':
            # Set the ripple to 0.05 dB
            self.b, self.a = signal.cheby1(self.N_forder,0.05,2/self.M*self.fc)
        else:
            print('ftype must be "butter" or "cheby1"')
        
    def up(self,x):
        """
        Upsample and filter the signal
        """
        y = self.M*ssd.upsample(x,self.M)
        y = signal.lfilter(self.b,self.a,y)
        return y
    
    def dn(self,x):
        """
        Downsample and filter the signal
        """
        y = signal.lfilter(self.b,self.a,x)
        y = ssd.downsample(y,self.M)
        return y


def freqz_resp(b,a=[1],mode = 'dB',fs=1.0,Npts = 1024,fsize=(6,4)):
    """
    A method for displaying digital filter frequency response magnitude,
    phase, and group delay. A plot is produced using matplotlib

    freq_resp(self,mode = 'dB',Npts = 1024)

    A method for displaying the filter frequency response magnitude,
    phase, and group delay. A plot is produced using matplotlib

    freqz_resp(b,a=[1],mode = 'dB',Npts = 1024,fsize=(6,4))

        b = ndarray of numerator coefficients
        a = ndarray of denominator coefficents
     mode = display mode: 'dB' magnitude, 'phase' in radians, or 
            'groupdelay_s' in samples and 'groupdelay_t' in sec, 
            all versus frequency in Hz
     Npts = number of points to plot; defult is 1024
    fsize = figure size; defult is (6,4) inches
    
    Mark Wickert, January 2015
    """
    f = np.arange(0,Npts)/(2.0*Npts)
    w,H = signal.freqz(b,a,2*np.pi*f)
    plt.figure(figsize=fsize)
    if mode.lower() == 'db':
        plt.plot(f*fs,20*np.log10(np.abs(H)))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain (dB)')
        plt.title('Frequency Response - Magnitude')

    elif mode.lower() == 'phase':
        plt.plot(f*fs,np.angle(H))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (rad)')
        plt.title('Frequency Response - Phase')

    elif (mode.lower() == 'groupdelay_s') or (mode.lower() == 'groupdelay_t'):
        """
        Notes
        -----

        Since this calculation involves finding the derivative of the
        phase response, care must be taken at phase wrapping points 
        and when the phase jumps by +/-pi, which occurs when the 
        amplitude response changes sign. Since the amplitude response
        is zero when the sign changes, the jumps do not alter the group 
        delay results.
        """
        theta = np.unwrap(np.angle(H))
        # Since theta for an FIR filter is likely to have many pi phase
        # jumps too, we unwrap a second time 2*theta and divide by 2
        theta2 = np.unwrap(2*theta)/2.
        theta_dif = np.diff(theta2)
        f_diff = np.diff(f)
        Tg = -np.diff(theta2)/np.diff(w)
        # For gain almost zero set groupdelay = 0
        idx = pylab.find(20*np.log10(H[:-1]) < -400)
        Tg[idx] = np.zeros(len(idx))
        max_Tg = np.max(Tg)
        #print(max_Tg)
        if mode.lower() == 'groupdelay_t':
            max_Tg /= fs
            plt.plot(f[:-1]*fs,Tg/fs)
            plt.ylim([0,1.2*max_Tg])
        else:
            plt.plot(f[:-1]*fs,Tg)
            plt.ylim([0,1.2*max_Tg])
        plt.xlabel('Frequency (Hz)')
        if mode.lower() == 'groupdelay_t':
            plt.ylabel('Group Delay (s)')
        else:
            plt.ylabel('Group Delay (samples)')
        plt.title('Frequency Response - Group Delay')
    else:
        s1 = 'Error, mode must be "dB", "phase, '
        s2 = '"groupdelay_s", or "groupdelay_t"'
        print(s1 + s2)
