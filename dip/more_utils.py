#!/usr/bin/env python

#Importing the necessary modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import os
from tqdm import tqdm
from astropy.timeseries import BoxLeastSquares
from astropy import units as u


#Defining the locating lightcurve function
def LocateLC(folder):
    ''' 
    scan dir tree and extract folders that look like lc*_cal
    '''
    fnames = os.listdir(folder)
    
    splitfiles = []
    split2files = []
    split3files = []
    goodfiles = []

    for i in fnames:
        splitfiles.append(i.split('_'))

    for s in splitfiles:
        if len(s) >1:
            if s[1] == 'cal':
                split2files.append(s[0]+'_'+s[1])
            
    for t in split2files:
        split3files.append(t.split('lc'))

    for u in split3files:
        if u[0] == '':
            goodfiles.append('lc' + u[1])

    return goodfiles
#End of LocateLC function

#Defining the search function
def search(asas_sn_id, folder):
    '''
    Input:
        asas_sn_id: the asassn id of the desired star
        folder: this is the path to the folder containing the folders of data

    Output:
        desired: this is the folder that contains the desired light curve data

    This function takes an asassnid and finds the folder with the correct light curve data. This function uses the LocateLC
    function to to find folders containing light curve data. It then reads those folders looking for the file that contains
    the data for the given asassnid. It does this by attempting to match the file name with the asassnid.
    '''
    goodfiles = LocateLC(folder)

    for i in tqdm(range(len(goodfiles))):
        gfnames = os.listdir(folder+'/'+goodfiles[i])
        for j in gfnames:
            if str(asas_sn_id)+'.dat' == j:
                return str(goodfiles[i])
#End of the search function

#Defining the plotparams function. It changes the parameters for plots to custom settings
def plotparams(ax, labelsize=15):
    '''
    Basic plot params

    :param ax: axes to modify

    :type ax: matplotlib axes object

    :returns: modified matplotlib axes object
    '''

    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=labelsize)
    ax.tick_params('both', length=8, width=1.8, which='major')
    ax.tick_params('both', length=4, width=1, which='minor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    return ax
#End of plotparams function

#Defining the find_dip function. This function uses scipy.signal.find_peaks to find dipping events
def find_dip(asas_sn_id, guide='known_dipper_lightcurves/', p=0.17, d=25, h=0.3, w=2):
    """ 
    Input:
        asas_sn_id: the asassn id of the desired star
        guide: the path to the data file of the desired star
        p: this is the parameter of prominence in the scipy.signal.find_peaks() function. Defined as prominence of peaks
        d: this is the distance parameter in the scipy.signal.find_peaks() function. This is the required minimum horizontal
            distance between neighboring peaks
        h: this is the parameter of height in the scipy.signal.find_peaks() function. This is defined as the required height
            of the peaks
        w: this is the width parameter in the scipy.signal.find_peaks() function. This is defined as the required width of 
            the peaks in the sample

    Output:
        dfg: this is the data frame of the stars g-band data
        peak: these are the indicies of the peaks
        prop: these are the properties of the peaks
        length: this is the number of peaks found

    This function uses the read_lightcurve function to create a dataframe with the data of both bands. This function uses the
    g band. It takes the mean and uses the mag - avg to evaluate. There are several different parameters that can be chosen for
    analysis. The function returns the dataframe and different information about the peaks
    """
    dfv, dfg = read_lightcurve(asas_sn_id, guide)

    meanmag = dfg["Mag"].mean()

    dfg["Mag-Avg"] = dfg["Mag"] - meanmag

    peaks = scipy.signal.find_peaks(dfg["Mag-Avg"], prominence=p, distance=d, height=h, width=w) 

    peak = peaks[0]

    prop = peaks[1]

    length = len(peak)

    return dfg, peak, prop, length
#End of find_dip function

#Defining the find_dipDF function. This function is similar to the find_dip function, but returns a dataframe
def find_dipDF(asas_sn_id, guide='known_dipper_lightcurves/', p=0.17, d=25, h=0.3, w=2):
    """
    Input:
        asas_sn_id: the asassn id of the desired star
        guide: the path to the data file of the desired star
        p: this is the parameter of prominence in the scipy.signal.find_peaks() function. Defined as prominence of peaks
        d: this is the distance parameter in the scipy.signal.find_peaks() function. This is the required minimum horizontal
            distance between neighboring peaks
        h: this is the parameter of height in the scipy.signal.find_peaks() function. This is defined as the required height
            of the peaks
        w: this is the width parameter in the scipy.signal.find_peaks() function. This is defined as the required width of 
            the peaks in the sample

    Output:
        PeakNum: this is the resulting dataframe from everything calculated by the function

    This function uses the read_lightcurve function to create a dataframe with the data of both bands. This function uses the
    g band. It takes the mean and uses the mag - avg to evaluate. There are several different parameters that can be chosen for
    analysis. The function returns a dataframe full of information about the peaks.
    """
    dfv, dfg = read_lightcurve(asas_sn_id, guide)

    meanmag = dfg["Mag"].mean()

    dfg["Mag-Avg"] = dfg["Mag"] - meanmag

    peaks = scipy.signal.find_peaks(dfg["Mag-Avg"], prominence=p, distance=d, height=h, width=w)

    MaxDepth = dfg["Mag"].max()

    max = dfg.loc[dfg["Mag"] == MaxDepth].reset_index(drop=True)

    End = dfg["JD"].max()

    Begin = dfg["JD"].min()

    Time = End - Begin

    Peakperday = len(peaks[0]) / Time

    Magstd = np.std(dfg["Mag"])

    d = {'asas_sn_id':asas_sn_id, 'Number of dips':len(peaks[0]), 'Average Magnitude':meanmag, 'Standard Deviation of Magnitude': Magstd,
         'Lowest Dip':dfg["Mag"].max(), 'Difference of average and lowest dip': dfg["Mag-Avg"].max(), 'Time of Lowest Dip':max["JD"], 
         'Peaks per time':Peakperday} #This the information put into the dataframe

    PeakNum = pd.DataFrame(data=d, index=[0])

    return PeakNum
#End find_dipDf function

#Defining the camparam function. This function takes the data and separates it based on the camera used.
def camparam(asas_sn_id, guide='known_dipper_lightcurves/'):
    ''' 
    Input:
        asas_sn_id: the asassn id of the desired star
        guide: the path to the data file of the desired star

    Output:
        result: this is a dataframe containing the parameters for each camera of a given light curve

    This function uses the find_dip function to give the dataframe. The function the takes a list of every unique
    camera and provides the parameters of each camera. Then the data of each camera is compiled into one dataframe.
    '''
    dfg, peak, prop, length = find_dip(asas_sn_id, guide)

    camarray = np.unique(dfg["Camera"].values)

    campar = pd.DataFrame()

    for i in camarray:
        camdf = dfg[dfg["Camera"] == i].reset_index(drop=True)
        d = {'asas_sn_id':asas_sn_id, 'Camera':camdf["Camera"], 'Average': camdf["Mag"].mean(), 'Median':camdf["Mag"].median(),
             'ST Deviation': np.std(camdf["Mag"])} #These are the set paramters of each camera
        adddf = pd.DataFrame(data=d)
        campar = pd.concat([campar,adddf]).reset_index(drop=True)

    result = campar.drop_duplicates('Camera').reset_index(drop=True)

    return result
#End camparam function

#Defining the box function. This function is the first filter applyed to the data in order to remove false positives.
def box(asas_sn_id, guide='known_dipper_lightcurves/'):
    ''' 
    Input:
        asas_sn_id: the asassn id of the desired star
        guide: the path to the data file of the desired star

    Output:
        filter3: This contains the data that has passed through the filter

    The parameters of the data was plotted and it was shown that most of the data including all of the known dippers lie 
    in or near these boundaries. Numbers of dips less than 10. Standard Deviation less than or equal to 0.15. Peaks in the 
    length of the curve less than or equal to 0.015. This function returns a dataframe with values that have peaks in the
    box described above.
    '''
    Peaknum = find_dipDF(asas_sn_id, guide)
    filter0 = Peaknum.loc[Peaknum["Number of dips"] > 0].reset_index(drop=True)
    filter1 = filter0.loc[filter0["Number of dips"] < 10].reset_index(drop=True)
    filter2 = filter1.loc[filter1["Peaks per time"] <= 0.015].reset_index(drop=True)
    filter3 = filter2.loc[filter2["Standard Deviation of Magnitude"] <= 0.15].reset_index(drop=True)

    return filter3
#End of box function

#Define the first camera filter
def camfilter(asas_sn_id, guide='known_dipper_lightcurves/', p=0.17, d=25, h=0.3, w=2):
    ''' 
    Input:
        asas_sn_id: the asassn id of the desired star
        guide: the path to the data file of the desired star
        p: this is the parameter of prominence in the scipy.signal.find_peaks() function. Defined as prominence of peaks
        d: this is the distance parameter in the scipy.signal.find_peaks() function. This is the required minimum horizontal
            distance between neighboring peaks
        h: this is the parameter of height in the scipy.signal.find_peaks() function. This is defined as the required height
            of the peaks
        w: this is the width parameter in the scipy.signal.find_peaks() function. This is defined as the required width of 
            the peaks in the sample

    Output:
        avgfilt: the dataframe that has filtered out the lightcurves by comparing the averages of the cameras

    This function uses the camparam and read_lightcurve functions to filter out bad cameras. The function looks for 
    cameras far from the average and removes them. The function takes the filtered data for the star and then looks
    for peaks in a similar manner to the find_dip function. It then returns a dataframe describing the filtered data.
    '''
    cams = camparam(asas_sn_id, guide)
    dfv, dfg = read_lightcurve(asas_sn_id, guide)
    filtereddf = pd.DataFrame()
    filtPeakNum1 = pd.DataFrame()

    cams["Difference"] = cams["Average"] - dfg["Mag"].mean()

    newdf1 = cams.loc[cams["Difference"] <= 0.1].reset_index(drop=True)

    newdf2 = newdf1.loc[newdf1["Difference"] >= -0.1].reset_index(drop=True)

    goodcam = newdf2["Camera"]

    if len(goodcam) == 0:
        badcam = {'asas_sn_id':asas_sn_id, 'Number of dips':np.nan, 'Average Magnitude':np.nan, 'Standard Deviation of Magnitude': np.nan,
           'Lowest Dip':np.nan, 'Difference of average and lowest dip': np.nan, 'Time of Lowest Dip':np.nan, 'Peaks per time':np.nan}
            #Parameters of the light curves that only have bad cameras
        filtPeakNum1 = pd.DataFrame(data=badcam, index=[0])

    else:
        for i in goodcam:
            alter = dfg.loc[dfg["Camera"] == i].reset_index(drop=True)
            filtereddf = pd.concat([alter,filtereddf]).reset_index(drop=True)

            filtmeanmag = filtereddf["Mag"].mean()

            filtereddf["Mag-Avg"] = filtereddf["Mag"] - filtmeanmag

            peaks = scipy.signal.find_peaks(filtereddf["Mag-Avg"], prominence=p, distance=d, height=h, width=w)

            filtMaxDepth = filtereddf["Mag"].max()

            filtmax = filtereddf.loc[filtereddf["Mag"] == filtMaxDepth].reset_index(drop=True)

            filtEnd = filtereddf["JD"].max()

            filtBegin = filtereddf["JD"].min()

            filtTime = filtEnd - filtBegin

            try:
                filtPeakperday = len(peaks[0]) / filtTime

            except:
                filtPeakperday = 0 

            filtMagstd = np.std(filtereddf["Mag"])

            filtd = {'asas_sn_id':asas_sn_id, 'Number of dips':len(peaks[0]), 'Average Magnitude':filtmeanmag, 'Standard Deviation of Magnitude': filtMagstd,
                'Lowest Dip':filtereddf["Mag"].max(), 'Difference of average and lowest dip': filtereddf["Mag-Avg"].max(), 'Time of Lowest Dip':filtmax["JD"], 
               'Peaks per time':filtPeakperday} #Parameters of the rerun light curves 

            filtPeakNum2 = pd.DataFrame(data=filtd, index=[0])

    filtPeakNum = pd.concat([filtPeakNum1,filtPeakNum2]).reset_index(drop=True)
    
    avgfilt = filtPeakNum.dropna()

    return avgfilt
#End camfilter function

# Defining the second camera filter.
def camfilter2(asas_sn_id, guide='known_dipper_lightcurves/', p=0.17, d=25, h=0.3, w=2):
    ''' 
    Input:
        asas_sn_id: the asassn id of the desired star
        guide: the path to the data file of the desired star
        p: this is the parameter of prominence in the scipy.signal.find_peaks() function. Defined as prominence of peaks
        d: this is the distance parameter in the scipy.signal.find_peaks() function. This is the required minimum horizontal
            distance between neighboring peaks
        h: this is the parameter of height in the scipy.signal.find_peaks() function. This is defined as the required height
            of the peaks
        w: this is the width parameter in the scipy.signal.find_peaks() function. This is defined as the required width of 
            the peaks in the sample

    Output:
        stdfilt: the dataframe that has filtered out the lightcurves by comparing the standard deviations of the cameras

    This function acts in a very similar way to the first one. The only difference is that this function compares the
    standard deviation of each camera with the average standard deviation. This is in the hopes of removing stars with
    bad camera points all over the lightcurve.
    '''
    cams = camparam(asas_sn_id, guide)
    dfv, dfg = read_lightcurve(asas_sn_id, guide)
    filtereddf = pd.DataFrame()
    filtPeakNum1 = pd.DataFrame()

    cams["Filter 2"] = cams["ST Deviation"] - np.std(dfg["Mag"])

    goodcams = cams.loc[cams["Filter 2"] < 0.1].reset_index(drop=True)

    onlycam = goodcams["Camera"]

    for i in onlycam:
        alter2 = dfg.loc[dfg["Camera"] == i].reset_index(drop=True)
        filtereddf = pd.concat([alter2,filtereddf]).reset_index(drop=True)

        filtmeanmag = filtereddf["Mag"].mean()

        filtereddf["Mag-Avg"] = filtereddf["Mag"] - filtmeanmag

        peaks = scipy.signal.find_peaks(filtereddf["Mag-Avg"], prominence=p, distance=d, height=h, width=w)

        filtMaxDepth = filtereddf["Mag"].max()

        filtmax = filtereddf.loc[filtereddf["Mag"] == filtMaxDepth].reset_index(drop=True)

        filtEnd = filtereddf["JD"].max()

        filtBegin = filtereddf["JD"].min()

        filtTime = filtEnd - filtBegin

        try:
            filtPeakperday = len(peaks[0]) / filtTime

        except:
            filtPeakperday = 0

        filtMagstd = np.std(filtereddf["Mag"])

        filtd = {'asas_sn_id':asas_sn_id, 'Number of dips':len(peaks[0]), 'Average Magnitude':filtmeanmag, 'Standard Deviation of Magnitude': filtMagstd,
                'Lowest Dip':filtereddf["Mag"].max(), 'Difference of average and lowest dip': filtereddf["Mag-Avg"].max(), 'Time of Lowest Dip':filtmax["JD"], 
               'Peaks per time':filtPeakperday} #These are the parameters of the resulting dataframe

        filtPeakNum2 = pd.DataFrame(data=filtd, index=[0])

    filtPeakNum = pd.concat([filtPeakNum1,filtPeakNum2]).reset_index(drop=True)
    
    stdfilt = filtPeakNum.dropna()

    return stdfilt
#End of camfilter2 function

#Defining the ploting function
def plot_dip(asas_sn_id, guide='known_dipper_lightcurves/'):
    """
    Input:
        asas_sn_id: the asassn id of the desired star
        guide: the path to the data file of the desired star

    Output: 
        this function returns a plot of the light curve of the data

    This function plots the light curve with the data of the given asassnid. This function sorts identifies unique cameras
    and then establishes the number of data points for each camera. The function then plots each different point for each
    different camera. It then marks the peaks that were found in the function and is organized to the desired specifications
    """
    dfg, peak, prop, length = find_dip(asas_sn_id, guide)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax = plotparams(ax)
    ax.set_title(asas_sn_id)

    cams = dfg["Camera"]
    camtype = np.unique(cams)
    camnum = len(camtype)

    colors = ["#6b8bcd", "#b3b540", "#8f62ca", "#5eb550", "#c75d9c", "#4bb092", "#c5562f", "#6c7f39", 
              "#ce5761", "#c68c45"]
    
    camdf = pd.DataFrame()

    for i in range(0,camnum):
        camdf = dfg.loc[dfg["Camera"] == camtype[i]].reset_index(drop=True)
        for j in range(0,len(camdf)):
            ax.scatter((camdf["JD"][j] - (2.458 * 10 **6)), camdf["Mag"][j], color = colors[i], alpha = 0.6)

    ax.vlines((dfg["JD"][peak] - (2.458 * 10**6)), dfg["Mag"].min(), dfg["Mag"].max(), "k", alpha=0.3)
    ax.invert_yaxis() 
    ax.set_xlabel('Julian Date - 2.458e6', fontsize=20)
    ax.set_ylabel('Magnitude', fontsize=20)
    plt.show() 
    #plt.savefig(str(asas_sn_id)+'.jpg') #Can be uncommented to save the plot
    #plt.close(fig)
#End of plotting function

#Defining the plot zoom function
def plot_zoom(asas_sn_id, guide='known_dipper_lightcurves/'):
    '''
    Input:
        asas_sn_id: the asassn id of the desired star
        guide: the path to the data file of the desired star

    Output: 
        this function returns a plot of the light curve of the data as well as plots zoomed in on the location of the peaks 

    This function plots the light curve with the data of the given asassnid. This function sorts identifies unique cameras
    and then establishes the number of data points for each camera. The function then plots each different point for each
    different camera. It then marks the peaks that were found in the function and is organized to the desired specifications
    This function also makes plots that are zoomed in on the peaks that have been found. All plots are included in the same
    image.
    '''
    file = search(asas_sn_id, folder=guide)
    dfg, peak, prop, length = find_dip(asas_sn_id, guide + file)
    fig, ax = plt.subplots((length+1),1, figsize=(12,6))
    

    cams = dfg["Camera"]
    camtype = np.unique(cams)
    camnum = len(camtype)

    colors = ["#6b8bcd", "#b3b540", "#8f62ca", "#5eb550", "#c75d9c", "#4bb092", "#c5562f", "#6c7f39", 
              "#ce5761", "#c68c45"]
    
    camdf = pd.DataFrame()    

    for h in range(0,len(ax.flat)):
        for i in range(0,camnum):
            camdf = dfg.loc[dfg["Camera"] == camtype[i]].reset_index(drop=True)
            for j in range(0,len(camdf)):
                ax[h].scatter((camdf["JD"][j] - (2.458 * 10 **6)), camdf["Mag"][j], color = colors[i], alpha = 0.6)
        ax[h].vlines((dfg["JD"][peak] - (2.458 * 10**6)), dfg["Mag"].min(), dfg["Mag"].max(), "k", alpha=0.3)
        ax[h].invert_yaxis()
        
    ax[0].set(title=asas_sn_id)
    
    for t in range(1,len(ax.flat)):
        ax[t].set_xlim(left= ((dfg["JD"][peak[t-1]]-(2.458*10**6)) - (0.05 * (dfg["JD"].mean()-(2.458*10**6)))), 
                       right = ((dfg["JD"][peak[t-1]]-(2.458*10**6)) + (0.05 * (dfg["JD"].mean()-(2.458*10**6)))))

    for g in ax.flat:
        g.set(xlabel='Julian Date - 2.458e6', ylabel = 'Magnitude')        
#End of the peak zoom function

#Defining of the periodogram function
def period(asas_sn_id, guess, guide='known_dipper_lightcurves/'):
    '''
    Input:
        asas_sn_id: the asassn id of the desired star
        guess: guess of the period
        guide: the path to the data file of the desired star

    Output:
        t: this is the period
        dfg: this is the updated dataframe

    This function takes the magnitude and converts it to the flux of the star. It then takes the flux and puts it in a 
    Box Least Squares periodogram that is modeled using the guess period. The periodogram then gives the best period
    along with a dataframe that has been updated with the orbit number and the phase
    '''
    file = search(asas_sn_id, folder=guide)
    dfv, dfg = read_lightcurve(asas_sn_id, guide + '/' +file)
    x = dfg["JD"]
    m = dfg["Mag"]
    y = 10 **(-m/2.5)
    y = y /np.median(y)

    periodgrid = np.linspace(guess-1, guess+1, 10000)
    durations = np.linspace(0.1, 0.9, 10)
    model = BoxLeastSquares(x,y)
    periodogram = model.power(periodgrid, durations)
    t = periodogram.period[np.argmax(periodogram.power)]

    dfg["OrbNum"] = dfg["JD"] / t
    dfg["Phase"] = dfg["OrbNum"] % 1

    return t, dfg
#End of the periodogram function

#Define the phase curve function
def phase(asas_sn_id, guess, guide='known_dipper_lightcurves/'):
    '''
    Input:
        asas_sn_id: the asassn id of the desired star
        guess: guess of the period
        guide: the path to the data file of the desired star

    Output: 
        this function returns a plot of the phase curve of the data

    This function plots the phase curve with the data of the given asassnid. This function uses the period function to do so
    and is similar in style to the plotting function. It also requires the guess used in the period function
    '''
    t, dfg = period(asas_sn_id, guess, guide)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax = plotparams(ax)
    ax.set_title(asas_sn_id)

    cams = dfg["Camera"]
    camtype = np.unique(cams)
    camnum = len(camtype)

    colors = ["#6b8bcd", "#b3b540", "#8f62ca", "#5eb550", "#c75d9c", "#4bb092", "#c5562f", "#6c7f39", 
              "#ce5761", "#c68c45"]
    
    camdf = pd.DataFrame()

    for i in range(0,camnum):
        camdf = dfg.loc[dfg["Camera"] == camtype[i]].reset_index(drop=True)
        for j in range(0,len(camdf)):
            ax.scatter(camdf["Phase"][j], camdf["Mag"][j], color = colors[i], alpha = 0.6)

    ax.invert_yaxis() 
    ax.set_xlabel('Phase', fontsize=20)
    ax.set_ylabel('Magnitude', fontsize=20)
    plt.show() 
    #plt.savefig(str(asas_sn_id)+'.jpg') #Can be uncommented to save the plot
    #plt.close(fig)
    #End of the phase curve function