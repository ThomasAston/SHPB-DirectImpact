########################################################################
#1. MAIN SCRIPT FOR OBTAINING DIRECT IMPACT SHPB STRESS-STRAIN CURVE 
########################################################################

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.widgets import Cursor, Button, Slider
from scipy.optimize import curve_fit
from scipy import interpolate
import pylab
from scipy import signal
import tkinter as tk
from tkinter import filedialog
import os
from strain import engStrain, strainTime

########################################################################
# INITIALISE
########################################################################
class initData():
    
    ### PROJECTILE ###
    def proj(self):
        L_proj = 0.50    # Projectile length (m)
        E_bar = 210e9    # Bar Young's modulus (Pa or N/m^2)
        rho_bar = 7850   # Bar density (kg/m^3)
        vel = np.sqrt(E_bar/rho_bar)   # Wave speed in bar material (m/s)
        duration_loading = 2*L_proj/vel     # Loading duration from striker (s)
        D_bar = (0.022022+0.022116)/2.0     # Bar diameter, from calipers??? (m) ***
        freq_acqui = 1e6    # Frequency acquired? (Hz?) ***
        L_inci_bar = 2.6    # Inicident bar length (m)
        print('Pulse duration: ', duration_loading, 's')
        barData = [L_proj,E_bar,rho_bar,vel,duration_loading,D_bar,freq_acqui,L_inci_bar]
        return barData

    ### SPECIMEN ###
    def spec(self):
        L_spec = 0.01   # Specimen length (m)
        D_spec = 0.02   # Specimen diameter (m)
        specData = [L_spec,D_spec]
        return specData

########################################################################
# PROCESS EXPERIMENTAL DATA
########################################################################
class procExpData():
    ### READ DATA FROM FILE AND PROCESS###
    def PROCESS(self, fileName):
        t,strain_incid,strain_trans,trig=np.loadtxt(fileName,unpack=True,skiprows=21)

        ### Average the first 100 points and calculate noise ###
        strain_incid_cumulative = 0 # Initialise cumulative sum of first 100 strains
        for i in range(100):
            strain_incid_cumulative += strain_incid[i] # Sum first 100 strains
        strain_incid = strain_incid - strain_incid_cumulative/100 # Strain - average strain gives noise
        
        strainData = [t,strain_incid,strain_trans,trig] # Compile all strain data

        ### Filtering ###
        timestep = t[3] - t[2]      # Calculate timestep between measurements
        datasecond = 1/timestep     # Calculate timesteps per second
        cutoff_freq = 10000         # Choose cutoff frequency for signal filtering
        n_pass=cutoff_freq/(0.5*datasecond)
        n_pass_a=cutoff_freq/(0.5*datasecond)-0.05
        n_pass_b=cutoff_freq/(0.5*datasecond)+0.05       
        (bbut, abut) = signal.butter(4, n_pass, btype='low', analog=0, output='ba')
        strain_incid_f= signal.filtfilt(bbut, abut, strain_incid)   # Filter incident strain
        strain_trans_f= signal.filtfilt(bbut, abut, strain_trans)   # Filter transmitted strain
        
        ### Second order numerical derivatives of strain signals ###
        strainRate_trans=np.zeros(len(t))       # Initialise array for raw transmitted strain rate
        strainRate_incid=np.zeros(len(t))       # Initialise array for raw incident strain rate       
        strainRate_trans_f=np.zeros(len(t))     # Initialise array for filtered transmitted strain rate
        strainRate_incid_f=np.zeros(len(t))     # Initialise array for filtered incident strain rate

        for i in range(len(t)-1-10):
            i = 10+i
            strainRate_trans[i]=(strain_trans[i+1]-strain_trans[i-1])/(2*timestep)
            strainRate_incid[i]=(strain_incid[i+1]-strain_incid[i-1])/(2*timestep)
            strainRate_trans_f[i]=(strain_trans_f[i+1]-strain_trans_f[i-1])/(2*timestep)
            strainRate_incid_f[i]=(strain_incid_f[i+1]-strain_incid_f[i-1])/(2*timestep)

        strainRateData = [strainRate_incid,strainRate_incid_f,strainRate_trans, strainRate_trans_f] # Compile all strain rate data

        ############################################################
        ### Plot incident strain pulse to select start and end times
        ############################################################
        x1 = 1 # Arbitrary value for starting time
        x2 = 2 # Arbitrary value for end time
        
        fig1, ax=plt.subplots(facecolor='white', figsize=(10,8))
        plt.subplots_adjust(left=0.1, bottom=0.35)
        plt.plot(t*1e6,strain_incid_f,'r-',label='Incid filt',markersize=7, markerfacecolor='w',markeredgecolor='r',markeredgewidth=2,linewidth=2)
        plt.plot(t*1e6,strain_incid,'b-',label='Incid',markersize=7, markerfacecolor='w',markeredgecolor='r',markeredgewidth=2,linewidth=2)
        p1 = plt.axvline(x=x1,color='green')    # Plot vertical line to be controlled by slider 1
        p2 = plt.axvline(x=x2,color='red')      # Plot vertical line to be controlled by slider 2
        plt.legend(loc='best',numpoints=1)
        plt.xlim(t[0]*1e6,t[-1]*1e6)
        plt.xlabel(r'$Time\ [microseconds]$',fontsize=24)
        plt.ylabel(r'$Strain\ [-]$',fontsize=24)
        plt.grid( b=None,which='major', axis='both',linewidth=1)
        plt.grid( b=None,which='minor', axis='both',linewidth=0.2)

        ### Setup sliders 1 and 2 for setting start and end time
        axSlider1 = plt.axes([0.1,0.2,0.8,0.05])
        slider1 = Slider(ax=axSlider1,label='Start time', valmin=t[0]*1e6,valmax=t[-1]*1e6)
        axSlider2 = plt.axes([0.1,0.1,0.8,0.05])
        slider2 = Slider(axSlider2,'End time', valmin=t[0]*1e6,valmax=t[-1]*1e6, slidermin=slider1)
       
        def val_update(val):
            xval1 = slider1.val
            p1.set_xdata(xval1)
            xval2 = slider2.val
            p2.set_xdata(xval2)
            plt.draw()
            return xval1, xval2
        cid1 = slider1.on_changed(val_update)
        cid2 = slider2.on_changed(val_update)

        ### Button for accepting start and end times set by sliders 1 and 2
        axButton1 = plt.axes([0.1, 0.9, 0.1, 0.1])
        btn1 = Button(axButton1, 'Accept')
        chosen_times = []
        def applySliders(event):
            xval1 = slider1.val
            xval2 = slider2.val
            chosen_times.append(xval1)
            chosen_times.append(xval2)
            plt.close(1)
            return chosen_times
        cid = btn1.on_clicked(applySliders)

        plt.show()
        ############################################################

        ### Choose start and end times from chosen_times and print
        t_cut_start = round(chosen_times[0])
        t_cut_end = round(chosen_times[1])
        
        print('Chosen start of search at: ', t_cut_start, 'microseconds')
        print('Chosen end of search at: ', t_cut_end, 'microseconds')

        #######################################
        ### DETERMINE PULSE INITIATION TIME ###
        #######################################
        
        ### Method 1 ###
        # Choose limit value of strain, which when exceeded, is chosen as...
        # initiation time of pulse.
        def Method1(strainData):
            signalPulse = 0      # Setup signal to determine start of pulse
            lim_strain = 14 # Chosen strain limit

            for i in range(len(t)-1-10):
                i = 10+i
                if (abs(strain_incid[i])>lim_strain):   # If strain exceeds limit
                    if signalPulse == 0:                # And pulse has not started
                        signalPulse = 1                 # Signal that pulse has started
                        t_ini = t[i]                    # Set t_ini to current loop time
                        print('(Method 1) Pulse initiation time is: ', t_ini, 's')
            return t_ini
                        
        ### Method 2 ###
        # Choose limit value of FILTERED strain RATE, which when exceeded, is chosen...
        # as initiation time of pulse.
        def Method2(strainData,strainRateData):
            signalPulse = 0         # Signal to determine start of pulse
            lim_strainRate = 5e5    # Chosen strain rate limit 
            
            for i in range(len(t)-1-10):
                i = 10+i
                if (abs(strainRate_incid_f[i])>lim_strainRate):     # If incident strain rate exceeds limit
                    if signalPulse==0:                              # And pulse has not started
                        signalPulse = 1                             # Signal that pulse has started
                        t_ini = t[i]                                # Set t_ini to current loop time
                        print('(Method 2) Pulse initiation time is: ', t_ini, 's')
            return t_ini

        ### Method 3 ###
        # Identify time at which filtered strain no longer oscillates about zero...
        # and choose as initiation time of pulse.
        def Method3(strainData, strainRateData, chosen_times):
            signalPulse = 0         # Signal to determine start of pulse
            t_cut_start = int(round(chosen_times[0]))     # Chosen starting time of strain data
            t_cut_end = int(round(chosen_times[1]))       # Chosen ending point of strain data  
            lim_strain = -4          # Chosen strain limit

            for i in range(t_cut_start, t_cut_end,1):
                strain_incid_cut = strain_incid_f[i:t_cut_end]      # Chop filtered strain to start at current start time
                meas_over_lim = len([num for num in strain_incid_cut[1:t_cut_end] if num<=lim_strain]) # Count number of strain measurements over limit
                meas_total = len(strain_incid_cut[1:t_cut_end])    # Count total number of strain measurements 
                if meas_over_lim == meas_total:     # If all measurements over limit
                    if signalPulse == 0:            # And pulse has not started
                        signalPulse = 1             # Signal that pulse has started
                        t_ini = t[i]                # Set t_ini to current time
                        print('(Method 3) Pulse initiation time is: ', t_ini*1e6, 'microseconds')
            return t_ini
    
        t_ini = Method3(strainData, strainRateData, chosen_times)

        #######################################
        ### CALCULATE SPECIMEN STRESS ###
        #######################################
        init = initData()
        L_proj,E_bar,rho_bar,vel,duration_loading,D_bar,freq_acqui,L_inci_bar = init.proj()
        L_spec,D_spec = init.spec()
        stress_spec = ((D_bar**2)/(D_spec**2))*(strain_incid)*0.21 # MPa

        #######################################
        ### SPECIMEN STRAIN ###
        #######################################
        
        t_cut_end = int(np.amax(strainTime))
        
        stressTime = np.arange(0,t_cut_end, 1)
        f = interpolate.interp1d(strainTime, engStrain)
        new_engStrain = f(stressTime)
        t_ini = int(t_ini*1e6)

        #######################################
        ### PLOT AND EXPORT ###
        #######################################
        print("Plotting Stress-Strain curve...")
        fig2=plt.figure(facecolor='white', figsize=(10,8))
        plt.plot(new_engStrain,stress_spec[t_ini:(t_ini+t_cut_end)]*-1,'b--',label='Stress-strain',markersize=7, markerfacecolor='w',markeredgecolor='r',markeredgewidth=2)
        plt.xlim(0, np.amax(new_engStrain))
        plt.legend(loc='best',numpoints=1)
        plt.xlabel(r'$Strain\ [-]$',fontsize=24)
        plt.ylabel(r'$Stress\ [MPa]$',fontsize=24)
        plt.grid( b=None,which='major', axis='both',linewidth=1)
        plt.grid( b=None,which='minor', axis='both',linewidth=0.2)
        plt.show()

def main():   
    root = tk.Tk()
    root.withdraw()
    print("Select the .txt file corresponding with the chosen video")
    fileName = filedialog.askopenfilename()
    fileChosen = ('%s' % os.path.basename(fileName))
    print("File chosen: ", fileChosen)
    root.destroy()

    proc = procExpData()
    proc.PROCESS(fileName)

if __name__ == "__main__":
    main()