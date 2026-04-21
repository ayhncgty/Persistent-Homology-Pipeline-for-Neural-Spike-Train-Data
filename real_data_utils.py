import numpy as np
import pandas as pd
import pickle

path_neuron_df = '/Users/cgty/Library/Mobile Documents/com~apple~CloudDocs/CLASSES/FSU/MyResearch/GITHUB/Spike-Train-Data-Analysis/Data/Temperature Data/NeuronDF.pickle' # fill this in yourself
pairs = pd.read_csv('pairs.csv',dtype= {'Date':str}) # pairs is a csv file that helps keep a fixed order for Mouse-Date pairs
with open(path_neuron_df,'rb') as file:
    Df = pickle.load(file)


print('Experimental data is loaded successfully.')
print('Following table displays all Mouse-Date pairs.')
display(pairs)


# helper functions to work with Df
def get_raster(Mouse,Date,Taste,Trial,time_interval = [0,4000]):
    """
    INPUT:
    1) Mouse = str
    2) Date = str
    3) Taste = int
    4) Trial = int
    5) time_interval = list with two entries. It will tell the min and max time to look at in the raster. Set to 0ms - 4000ms by default.
    OUTPUT: numpy array. Will extract from Df the specified raster plot
    """
    frame = Df[(Df['MouseID'] == Mouse) & (Df['Date'] == Date) & (Df['Taste'] == Taste) & (Df['Trial'] == Trial)].iloc[:,7:] # Column 7 and on is spike train
    frame_array = np.array(frame) # turn it into a numpy array. This represents the whole 4000ms
    
    #### Extract the time interval specified ####
    time_begin = time_interval[0]
    time_end = time_interval[1]
    
    frame_array = frame_array[:,time_begin:time_end] # update the frame_array to extract the specified time interval

    return frame_array

def get_dates(Mouse):
    """
    INPUT:
    Mouse = str
    OUTPUT: numpy array consisting of dates the given mouse had trials
    """
     # get dates
    out = Df[(Df['MouseID'] == Mouse) & (Df['Taste'] == 0)]['Date'].unique() # This is taste independent. So, we will pick taste 0 as a representative.
    return out

def get_trial(Mouse, Date, Taste):
    """
    INPUT: 
    Mouse
    Date
    Taste
    OUTPUT:
    Number of trials
    """
    out = Df[(Df['MouseID'] == Mouse) & (Df['Date'] == Date) &(Df['Taste'] == Taste)]['Trial'].unique()
    return(len(out))  


def get_neurons(Mouse, Date):
    """
    Input
    Mouse: str
    Date: str 
    Returns
    Neurons that belong to the Mouse-Date pair 
    """
    row_mask = (Df["MouseID"] == Mouse) & (Df["Date"] == Date)
    neurons = np.unique(np.array(Df[row_mask]["Neuron"]))
    return neurons

def get_trial_from_neuron(neuron,taste,time_interval = [2000,4000]):
    """
    Input
    neuron: int -- neuron id index in Df
    taste: int -- 0, 1 or 2.
    time_interval = [2000,4000] by default
    Returns:
    array -- Returns all trials from the neuron-taste pair. array has shape (# of trials, end_time - begin_time)
    """
    row_mask = (Df["Neuron"] == neuron) & (Df["Taste"] == taste)
    trials = np.array(Df[row_mask].iloc[:,7:])
    begin_time = time_interval[0]
    end_time = time_interval[1]
    array = np.array(trials)[:,begin_time:end_time]

    return array

# Map Mouse-Date-Taste (MDT) to all of its corresponding rasters/trials
MDT_to_trials = {}             
for _, row in pairs.iterrows():
    Mouse = row["Mouse"]
    Date = row["Date"]

    for Taste in (0, 1, 2):
        # Assuming get_trial returns an integer count of trials
        n_trials = get_trial(Mouse=Mouse, Date=Date, Taste=Taste)

        raster_list = []
      

        for j in range(n_trials):
            # Fetch raster: Shape is likely (N_neurons, N_timepoints)
            raster = get_raster(Mouse=Mouse, Date=Date, Taste=Taste, Trial=j, time_interval=[2000, 4000])
            raster_list.append(raster)

        # Store
        MDT_to_trials[(Mouse, Date, Taste)] = raster_list