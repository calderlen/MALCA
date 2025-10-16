



# all of this code is adapted from Brayden JoHantgen's code



def read_lightcurve_dat(asas_sn_id, guide = 'known_dipper_lightcurves/'):
    """
    Input: 
        asas_sn_id: the asassn id of the desired star
        guide: the path to the data file of the desired star

    Output: 
        dfv: This is the dataframe for the V-band data of the star
        dfg: This is the dataframe for the g-band data of the star
    
    This function reads the data of the desired star by going to the corresponding file and copying the data of that file onto 
    a data frame. This data frame is then sorted into two data frames by comparing the value in the Photo filter column. If the
    Photo filter column data has a value of one, its row is sorted into the data frame corresponding to the V-band. If the Photo
    filter column data has a value of zero, it gets sorted into the data frame corresponding to the g-band.
    """
    fname = os.path.join(guide, str(asas_sn_id)+'.dat')

    dfv = pd.DataFrame()
    dfg = pd.DataFrame()

    fdata = pd.read_fwf(fname, header=None)
    fdata.columns = ["JD", "Mag", "Mag_err", "Quality", "Cam_number", "Phot_filter", "Camera"] #These are the columns of data

    dfv = fdata.loc[fdata["Phot_filter"] == 1].reset_index(drop=True) #This sorts the data into the V-band
    dfg = fdata.loc[fdata["Phot_filter"] == 0].reset_index(drop=True) #This sorts the data into the g-band

    dfv['Mag'].astype(float)
    dfg['Mag'].astype(float)

    dfv['JD'].astype(float)
    dfg['JD'].astype(float)

    return dfv, dfg


def read_lightcurve_csv(asas_sn_id, guide = 'known_dipper_lightcurves/'):
    """
    Input: 
        asas_sn_id: the asassn id of the desired star
        guide: the path to the data file of the desired star

    Output: 
        dfv: This is the dataframe for the V-band data of the star
        dfg: This is the dataframe for the g-band data of the star
    
    This function reads the data of the desired star by going to the corresponding file and copying the data of that file onto 
    a data frame. This data frame is then sorted into two data frames by comparing the value in the Photo filter column. If the
    Photo filter column data has a value of one, its row is sorted into the data frame corresponding to the V-band. If the Photo
    filter column data has a value of zero, it gets sorted into the data frame corresponding to the g-band.
    """
    fname = os.path.join(guide, str(asas_sn_id)+'.csv')

    df = pd.read_csv(fname)

    df['Mag'] = pd.to_numeric(df['mag'],errors='coerce')
    df = df.dropna()
    
    df['Mag'].astype(float)
    df['JD'] = df.HJD.astype(float)

    dfg = df.loc[df['Filter'] == 'g'].reset_index(drop=True)
    dfv = df.loc[df['Filter'] == 'V'].reset_index(drop=True)

    return dfv, dfg

# This function finds the peaks 
def find_peak(df, prominence=0.17, distance=25, height=0.3, width=2):
	"""
	Inputs:
		df: dataframe of the data, requires columns of 'Mag' and 'JD'
		prominence: same parameter of scipy.signal.find_peaks()
		distance: same parameter of scipy.signal.find_peaks()
		height: same parameter of scipy.signal.find_peaks()
		width: same parameter of scipy.signal.find_peaks()

	Outputs:
		peak: a series of the peaks found
		meanmag: the average magnitude of the light curve
		length: the number of peaks found

	Description:
	"""
	df['Mag'] = [float(i) for i in df['Mag']]
	df['JD'] = [float(i) for i in df['JD']]
	mag = df['Mag']
	jd = df['JD']

	meanmag = sum(mag) / len(mag)
	df_mag_avg = [i - meanmag for i in mag]
	
    peaks = scipy.signal.find_peaks(df_mag_avg, prominence=prominence, distance=distance, height=height, width=width) 
	
    peak = peaks[0]
	prop = peaks[1]
	
    length = len(peak)
	
    peak = [int(i) for i in peak]
	peak = pd.Series(peak)
	
    return peak, meanmag, length	
# End of the find_peak

# This function creates a custom id using the position of the star

# End of custom_id


# This function plots the light curve

# End of plot_light_curve

#

#

#

#

#
#def peak_params(df):
#