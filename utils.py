
## General packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch # needed for plot_barcode function

## Victor-Purpura related packages
import quantities as pq
from elephant.spike_train_dissimilarity import victor_purpura_distance
from neo.core import SpikeTrain

## TDA related packages
from ripser import ripser



## helper function to extract spike times
def spike_times(spike_train):
    return np.where(spike_train)[0]

# ====== Plotters ======================
def plot_raster(raster, title='Raster Plot', xlabel='Time (ms)', ylabel='Neuron ID',axes = None):

    # check if axes is provided, if not create a new figure
    if axes is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        ax = axes
    
    for i in range(raster.shape[0]):
        spike_times = np.where(raster[i] == 1)[0]
        ax.scatter(spike_times, np.full_like(spike_times, i), marker='o', color='black', s=10)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticks(ticks=np.arange(0,raster.shape[0]), labels=np.arange(1, raster.shape[0]+1))
    ax.set_xlim(-0.5, raster.shape[1] +10)
    ax.set_ylim(-0.5, raster.shape[0] + 0.5)

def plot_spike_train(spike_train, axes=None, **vline_kwargs):
    """
    Plot a spike train as vertical lines.
    spike_train: np.ndarray of shape (T,), binary (0/1)
    axes: matplotlib axes object, if None, creates a new figure
    vline_kwargs: passed to axes.vlines (e.g., color, linewidth)
    Returns:
        axes: the matplotlib axes object used
    """
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(figsize=(10, 2))

    spike_times = np.where(spike_train)[0]
    axes.vlines(spike_times, ymin=0, ymax=1, **vline_kwargs)
    axes.set_yticks([])

    # Remove grid and spines for cleaner look
    axes.grid(False)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_visible(False)

    return axes

def plot_barcode(barcode_dim_0,
                        barcode_dim_1=None,
                        r=None,
                        figsize=(8, 4),
                        bar_height=0.25,
                        pad=0.05,
                        ax=None,
                        colors=None,
                        linewidth=8):
    """
    Plot persistence barcodes given separate arrays for H0 and H1.

    Parameters
    ----------
    barcode_dim_0 : (n0, 2) array
        Birth–death pairs for H0.
    barcode_dim_1 : (n1, 2) array or None
        Birth–death pairs for H1 (optional).
    r : float or None
        Filtration value at which to truncate bars.
        If None, use the max finite death across all bars.
    figsize : tuple
        Figure size if ax is None.
    bar_height : float
        Vertical thickness of each bar.
    pad : float
        Vertical spacing between bars.
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, a new figure/axis is created.
    colors : dict or None
        Mapping {0: color_for_H0, 1: color_for_H1}.
    """
    if colors is None:
        colors = {0: 'tomato', 1: 'cornflowerblue'}

    # Collect all bars in one list with dim labels
    bar_data = []

    if barcode_dim_0 is not None:
        for b, d in np.asarray(barcode_dim_0):
            bar_data.append((0, float(b), float(d)))

    if barcode_dim_1 is not None:
        for b, d in np.asarray(barcode_dim_1):
            bar_data.append((1, float(b), float(d)))

    if len(bar_data) == 0:
        return

    # Determine truncation level r
    finite_deaths = [d for (_, _, d) in bar_data if np.isfinite(d)]
    if r is None:
        r = max(finite_deaths) if finite_deaths else 1.0

    # Clip deaths at r
    bar_data = [(dim, b, min(d, r)) for (dim, b, d) in bar_data if b < r]

    # Compute y-positions: stack H0, then a gap, then H1
    y_positions = []
    current_y = 0.0

    # H0 bars
    for dim, b, d in bar_data:
        if dim == 0:
            y_positions.append((dim, current_y, b, d))
            current_y += bar_height + pad

    # Small extra gap between H0 and H1
    if any(dim == 1 for dim, _, _ in bar_data):
        current_y += bar_height

    # H1 bars
    for dim, b, d in bar_data:
        if dim == 1:
            y_positions.append((dim, current_y, b, d))
            current_y += bar_height + pad

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot bars
    xmax = 0.0
    for dim, y, b, d in y_positions:
        ax.plot([b, d], [y, y], color=colors.get(dim, 'black'), linewidth=linewidth)
        xmax = max(xmax, d)

    # Axis formatting
    ax.set_yticks([])
    ax.set_ylim(-bar_height, current_y + bar_height)
    ax.set_xlim(0, max(r, xmax) * 1.05)
    ax.set_xlabel("Filtration value")
    ax.set_title("Persistence barcode")

    # Legend
    legend_elements = []
    if any(dim == 0 for dim, _, _, _ in y_positions):
        legend_elements.append(Patch(facecolor=colors[0], edgecolor='none', label="H0"))
    if any(dim == 1 for dim, _, _, _ in y_positions):
        legend_elements.append(Patch(facecolor=colors[1], edgecolor='none', label="H1"))
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left')

 # ==========================================================

# ===== Functions to Generate Spike Trains and Rasters & Adding Noise======
def gen_spike_train(T=1000, prob_map=None, random_state=None):
    """
    Generate a random spike train of length T.
    prob_map: list of tuples (start, end, prob)
        Each tuple defines a time window [start, end] and the probability of spiking in that window.
    random_state: int, np.random.Generator, or None
        Seed or random generator for reproducibility.
    Returns:
        spike_train: np.ndarray of shape (T,), binary (0/1)
    """
    rng = np.random.default_rng(random_state)
    spike_train = np.zeros(T, dtype=int)
    if prob_map is None:
        prob_map = [(0, T-1, 0.1)]  # default: uniform low probability

    for start, end, prob in prob_map:
        start = max(0, start)
        end = min(T-1, end)
        spike_train[start:end+1] = rng.random(end-start+1) < prob

    return spike_train





# Define a function to introduce noise to the raster with a specified noise level---that is either randomly shift some spikes by a small random amount and/or randomly add or remove spikes in the raster.
def add_noise_to_raster(raster, noise_level_shift=0.1,shift_strength = 5, noise_level_add =0.001, noise_level_remove = 0.001,  random_state=None):
    """
    Add noise to the raster data by randomly shifting spikes and adding/removing spikes.

    Parameters:
    raster (np.ndarray): The original raster data (2D array: neurons x time).
    noise_level_shift (float): Probability of shifting each spike.
    shift_strength (int): Maximum amount (in bins) to shift a spike.
    noise_level_add (float): Probability of adding a spike at an empty bin.
    noise_level_remove (float): Probability of removing an existing spike.
    random_state: int, np.random.Generator, or None for reproducibility.

    Returns:
    np.ndarray: The noisy raster data (same shape as input).
    """
    rng = np.random.default_rng(random_state)
    noisy_raster = raster.copy()

    # 1. Randomly remove spikes. Only applicable if there is a spike at that time point.
    for i in range(noisy_raster.shape[0]):
        for j in range(noisy_raster.shape[1]):
            if noisy_raster[i, j] > 0 and rng.random() < noise_level_remove:
                noisy_raster[i, j] = 0
    
    # 2. Randomly shift some spikes in time
    for i in range(noisy_raster.shape[0]):
        for j in range(noisy_raster.shape[1]):
            if noisy_raster[i, j] > 0 and rng.random() < noise_level_shift:
                shift = rng.integers(-shift_strength, shift_strength + 1)  # Shift by -shift_strength to +shift_strength ms
                new_j = j + shift
                # only shift if there is no spike at the new time point
                if 0 <= new_j < noisy_raster.shape[1] and noisy_raster[i, new_j] == 0:
                    noisy_raster[i, j] = 0  # Remove original spike
                    noisy_raster[i, new_j] = 1  # Add shifted spike
             
    # 3. Randomly add spikes. Only applicable if there is no spike at that time point.
    for i in range(noisy_raster.shape[0]):
        for j in range(noisy_raster.shape[1]):
            if noisy_raster[i, j] == 0 and rng.random() < noise_level_add:
                noisy_raster[i, j] = 1

    


    return noisy_raster


def add_noise_to_spike_train(spike_train, insert_spike_prob=0.001, remove_spike_prob=0.001, shift_spike_prob=0.01, shift_strength=1, random_state=None):
    """
    Adds noise to a spike train by inserting, removing, or shifting (jiggling) spikes.
    - spike_train: 1D input array of 0s and 1s
    - insert_spike_prob: probability of inserting a spike at a given index (if no spike)
    - remove_spike_prob: probability of removing a spike at a given index (if spike exists)
    - shift_spike_prob: probability of shifting a spike left or right by up to shift_strength units (if spike exists)
    - shift_strength: maximum number of indices to shift a spike (left or right)
    - random_state: int, np.random.Generator, or None for reproducibility

    Returns:
        noisy_train: 1D array after noise added
    """
    rng = np.random.default_rng(random_state)
    noisy_train = spike_train.copy()
    N = len(spike_train)

    # loop for inserting or removing spikes
    for i in range(N):
        if noisy_train[i] == 0: # if there is no spike at index i
            # Insert a spike with a certain probability
            if rng.random() < insert_spike_prob:
                noisy_train[i] = 1
        else: # if there is a spike at index i
            # Remove the spike with a certain probability
            if rng.random() < remove_spike_prob:
                noisy_train[i] = 0

    # after inserting or removing spikes, we can now jiggle the spikes
    for i in range(N):
        # If there is a spike at index i, we can jiggle it
        # Jiggle the spike left or right with a certain probability
        if noisy_train[i] == 1 and rng.random() < shift_spike_prob:
            shift = rng.integers(-shift_strength, shift_strength + 1)
            new_idx = i + shift
            if 0 <= new_idx < N:
                noisy_train[i] = 0
                noisy_train[new_idx] = 1

    return noisy_train


def shift_spike_train(spike_train, shift_amount):
    """
    Shifts the entire spike train by a given amount.
    If shift_amount > 0, shifts right (later in time).
    If shift_amount < 0, shifts left (earlier in time).
    Expands the time domain as needed.
    
    Parameters:
        spike_train (np.ndarray): 1D binary array.
        shift_amount (int): Number of bins to shift (+ right, - left).
        
    Returns:
        shifted_train (np.ndarray): New spike train with expanded time domain.
    """
    spike_times = np.where(spike_train == 1)[0]
    spike_times_new = spike_times + shift_amount

    # Only keep spikes that are within valid indices
    valid_spike_times = spike_times_new[spike_times_new >= 0]
    if len(valid_spike_times) == 0:
        # No spikes remain after shift
        return np.zeros(1, dtype=int)

    new_length = max(spike_train.shape[0], valid_spike_times.max() + 1)
    shifted_train = np.zeros(new_length, dtype=int)
    shifted_train[valid_spike_times] = 1

    return shifted_train


# Victor Purpura Distance
## trivial implementation of Victor Purpura distance 
## this corresponds to any q value with q > 2.0
def VP_trivial(raster): 
    """
    Input: raster is a collection of spike trains represented as a NxM numpy array where N = number of neurons and M = total time.
    Output: Distance matrix.
    """
    distance_matrix = np.zeros((raster.shape[0],raster.shape[0])) # initialize a dm
    
    # For each pair of spike trains calculate the distance
    for i in range(raster.shape[0]): 
        for j in range(i,raster.shape[0]):
            # spike trains are originally encoded as 0s and 1s. We turn them into lists of spike times below.
            s1 = spike_times(raster[i]) 
            s2 = spike_times(raster[j])
            # We assume WLOG that the first spike train contains fewer or equal number of spikes. Rather than python lists, we will represent them as python sets.
            if len(s1) <= len(s2):
                set1 = set(s1) 
                set2 = set(s2)
            # If this is not the case, make it so.
            else:
                set1 = set(s2)
                set2 = set(s1)
            
            difference = set1.difference(set2) # to account for the not-aligned spikes in s1 
            current_distance = 2 * len(difference) + len(set2) - len(set1) # distance between the current pair of spike trains. 
            # note that the above formula is the same as the symmetric difference: |S\S'| + |S'\S|
            distance_matrix[i][j] = current_distance # update the distance matrix

    np.fill_diagonal(distance_matrix,0)
    distance_matrix = distance_matrix + distance_matrix.T 
                   
    return distance_matrix


def VP(raster,q = 1.0,t_stop = 2000.0): 
    """
    Input: 
    -raster Numpy array. This is a collection of spike trains represented as a NxM numpy array where N = number of neurons and M = total time.
    - q = 1.0: Set to 1.0 by default. Temporal-sensitivity control parameter
    - t_stop: Float. Last time to consider in spike train
    Output: Victor purpura distance matrix
    """
    # We will base this on elephant's implementation unless q>2, in which case, we use a custom implementation.
    if q>2:
        victor_purpura_dm = VP_trivial(raster)
    else:
        q = q/(1.0 * pq.ms)
        spike_train_list = [SpikeTrain(spike_times(raster[i]), units = 'ms',t_stop = t_stop) for i in range(raster.shape[0])]
        victor_purpura_dm = victor_purpura_distance(spike_train_list,q)
    return victor_purpura_dm


# computing barcode from a raster
def compute_barcode_from_raster(raster,dim = 0,q = 3.0):
    """
    Computes the barcode from a raster.
    
    Parameters:
    - raster: 2D numpy array representing the raster.
    - dim = 0: Homology dimension. Set to 0 by default
    - q = 3.0: Victor-Purpura hyper parameter. Set to 3 by default, which is the trivial version.
    
    Returns:
    - barcode:  persistence barcode of raster in specified dimension
    """
    # Compute VP distance matrix
    vp_dm = VP(raster, q = q )
    
    # Compute barcode in given dimension
    barcode = ripser(vp_dm, distance_matrix=True)['dgms'][dim]
    
    return barcode

def bottleneck_zero(Y1, Y2):
    """
    Input:
      Y1, Y2 : Persistence Diagrams for H₀
               Each is an iterable of (birth, death) pairs (e.g. list of tuples or an (m×2) array).
    Returns:
      The H₀ bottleneck distance (a single float).
    """
    # 1) Extract only the death‐times, sort in descending order, drop infinities
    #    (Y1_raw might be an array of shape (m,2) or a list of (b,d) tuples).
    Y1_deaths = np.array([d for (_, d) in Y1])   # shape = (m,)
    Y2_deaths = np.array([d for (_, d) in Y2])   # shape = (k,)

    # Sort descending, then remove infinite bars
    Y1_sorted = np.sort(Y1_deaths)[::-1]
    Y1_sorted = Y1_sorted[np.isfinite(Y1_sorted)]

    Y2_sorted = np.sort(Y2_deaths)[::-1]
    Y2_sorted = Y2_sorted[np.isfinite(Y2_sorted)]

    # 2) Let m = length of the longer list; n = length of the shorter list
    m = max(len(Y1_sorted), len(Y2_sorted))
    n = min(len(Y1_sorted), len(Y2_sorted))

    distance_list = []

    # 3) Compare the top‐n bars one‐by‐one
    for i in range(n):
        diff = abs(Y1_sorted[i] - Y2_sorted[i])

        # If the difference already exceeds half of either death‐time, 
        # that half‐death penalty is the bottleneck for this pair.
        half_penalty = max(Y1_sorted[i]/2, Y2_sorted[i]/2)
        if diff > half_penalty:
            distance_list.append(half_penalty)
        else:
            distance_list.append(diff)

    # 4) If the two diagrams had unequal numbers of bars,
    #    there is one “extra” bar in the longer list at index = n.
    #    Its penalty is “(death of that bar)/2”. If lengths are equal, penalty = 0.
    if len(Y1_sorted) > len(Y2_sorted):
        extra_penalty = Y1_sorted[n] / 2
    elif len(Y2_sorted) > len(Y1_sorted):
        extra_penalty = Y2_sorted[n] / 2
    else:
        extra_penalty = 0.0

    distance_list.append(extra_penalty)

    # 5) The bottleneck distance is the maximum penalty across all paired bars
    return max(distance_list)




# --------------- TDA Pipeline Functions -----------------
## for validation Leave-One-Out
def LeaveOneOut(DistanceMatrix, y, trial_meta=None, return_predictions=False): 
    """
    Input:
    --------
    DistanceMatrix: NxN numpy array
        Pairwise distance matrix between all N samples. The diagonal entries are 0 (self-distances).
    
    y: 1D array-like of shape (N,)
        Class labels corresponding to each sample (row/column) in DistanceMatrix.
    
    trial_meta: list of tuples or None, optional
        List of (stimulus_ID, trial_ID) for each sample. The i-th element corresponds to
        the i-th row/column of DistanceMatrix. If None, correct_trials will be indices.

    return_predictions: bool, default=False
        If True, also return a list of predicted class labels for each sample.

    Returns:
    --------
    results: dict with keys
        - "correct_trials": list of (stimulus_ID, trial_ID) or indices for correctly classified samples
        - "accuracy_score": float, correct_count / N
        - "predicted_labels" (only if return_predictions=True): list of predicted labels in the same order as y

    Description:
    -----------
    This function performs leave-one-out (LOO) classification.
    For each sample i:
      1. Sort row i of DistanceMatrix.
      2. Pick the second-smallest entry (nearest neighbor, since the smallest is self-distance 0).
      3. Use that neighbor’s label as the prediction for sample i.
      4. If prediction == true label y[i], count it correct and record trial_meta[i] or i.

    Tie Situations:
    ----------------
    In case of ties in the distance values, `np.argsort` breaks them
    by index order—so the smallest index among equals is chosen, ensuring
    deterministic behavior under floating-point imprecision.
    * sorts first by value, then by index (i.e., row position)
    Example:
    DistanceMatrix[i] = [0.0, 1.5, 1.5, 1.5, 2.0]
    np.argsort(DistanceMatrix[i]) = [0, 1, 2, 3, 4]

    """
    correct_trials = []
    predicted_labels = []
    N = DistanceMatrix.shape[0]
    count = 0

    for i in range(N):
        # argsort gives indices of sorted distances; [0] is self, so [1] is nearest neighbor
        order = np.argsort(DistanceMatrix[i])
        nearest_index = order[1]
        predicted_class = y[nearest_index]
        predicted_labels.append(predicted_class)

        # record correctness
        if predicted_class == y[i]:
            count += 1
            if trial_meta is not None:
                correct_trials.append(trial_meta[i])
            else:
                correct_trials.append(i)

    accuracy_score = count / N

    # Build the results dictionary
    results = {
        "correct_trials": correct_trials,
        "accuracy_score": accuracy_score
    }
    if return_predictions:
        results["predicted_labels"] = predicted_labels

    return results


# TDA pipeline for custom data.
# this one uses the bottleneck_zero and VP_trivial
def TDA_pipeline(
    rasters,
    labels,
    time_interval_post=[2000, 4000],
    return_bdm=False,
    return_barcodes=False,
    return_DMs=False
):
    """
    rasters : list of np.ndarray, each shape (n_trains_i, T)
    labels  : array-like of ints, length == len(rasters)
    time_interval_post: list [start, end] Time window for the raster.
    q       : float, Victor–Purpura cost per ms
    t_stop  : float, endpoint for VP distance
    return_* flags : which intermediate results to include
    
    Returns
    -------
    results : dict with keys
      'network_score' : float
      'BDM'           : (n_rasters x n_rasters) array, if return_bdm
      'barcodes'      : list of persistence diagrams, if return_barcodes
      'DMs'           : list of VP‐distance matrices, if return_DMs
    """
    assert len(rasters) == len(labels), "must have same #rasters and #labels"
    n = len(rasters)

    # adjust time-window
    begin_time = time_interval_post[0]
    end_time   = time_interval_post[1]
    rasters = [raster[:,begin_time:end_time] for raster in rasters]

 
    # Create trial meta
    trial_counters = {label: 0 for label in np.unique(labels)}
    trial_meta = []
    
    for stim in labels:
        trial_meta.append((stim, trial_counters[stim]))
        trial_counters[stim] += 1
    
    trial_meta = np.array(trial_meta)

    # 1) Compute VP distance & barcodes for each raster
    DMs = []
    barcodes = []
    for R in rasters:
        # R is shape (n_trains, T)
        dm = VP_trivial(R)          # → shape (n_trains, n_trains)
        DMs.append(dm)
        dgm0 = ripser(dm, distance_matrix=True)['dgms'][0]  # 0-d barcode
        barcodes.append(dgm0)
    
    # 2) Build full bottleneck‐distance matrix between barcodes
    BDM = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            d = bottleneck_zero(barcodes[i], barcodes[j])
            BDM[i, j] = BDM[j, i] = d
    
    # 3) Run Leave‐One‐Out classification on the BDM
    loo_res = LeaveOneOut(BDM, y=np.array(labels), trial_meta=trial_meta)
    net_score = loo_res['accuracy_score']
    
    # 4) Package results
    results = {'network_score': net_score}
    if return_bdm:
        results['BDM'] = BDM
    if return_barcodes:
        results['barcodes'] = barcodes
    if return_DMs:
        results['DMs'] = DMs
    
    return results



## Below added April 21 2026

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage import gaussian_filter
from ripser import ripser
from tqdm import tqdm

def rasters_to_barcode(list_of_rasters, dim = 0, q = 3.0):
    list_of_barcodes = []
    for raster in tqdm(list_of_rasters, desc="Generating barcodes"):
        if q > 2.0:
            vp_dm = VP_trivial(raster)
        else:
            vp_dm = VP(raster, q = q)
        dgm = ripser(vp_dm,distance_matrix= True)['dgms'][dim]
        list_of_barcodes.append(dgm)

    return list_of_barcodes

def tda_rhv(rasters,labels,dim = 0,n_repeats = 20, test_size = 0.3, q =3.0, return_bdm = False, random_state = 42): # TDA pipeline with repeated holdout validation (rhv)
    y = np.array(labels)
    X_barcodes = rasters_to_barcode(rasters,dim = dim, q = q)
    n = len(X_barcodes)
    BDM = np.zeros((n, n), dtype=float)
    for i in tqdm(range(n), desc="Building BDM"):
        for j in range(i + 1, n):
            if dim == 0:
                d = bottleneck_zero(X_barcodes[i], X_barcodes[j]) 
            else:
                d = persim.bottleneck(X_barcodes[i],X_barcodes[j])
            
            BDM[i, j] = d
            BDM[j, i] = d

    splitter = StratifiedShuffleSplit(
        n_splits=n_repeats, test_size=test_size, random_state=random_state
    )

    scores = []
    for train_idx, test_idx in splitter.split(np.zeros_like(y), y):
        D_train = BDM[np.ix_(train_idx, train_idx)]
        D_test = BDM[np.ix_(test_idx, train_idx)]

        knn = KNeighborsClassifier(n_neighbors=1, metric="precomputed")
        knn.fit(D_train, y[train_idx])
        preds = knn.predict(D_test)

        scores.append(accuracy_score(y[test_idx], preds))

    mean_score = np.mean(scores) 
    if return_bdm:
        return mean_score, BDM
    else:
        return mean_score


    


def raster_to_svm_features(raster, sigma=30):
    """
    Computes smoothed firing rates for the SVM baseline.
    Flattens the array to create a 1D feature vector for each trial.
    """
    smoothed = gaussian_filter(raster.astype(float), sigma=(0, sigma))
    return smoothed.flatten()

def svm_smoothed_rasters(rasters, labels, sigma=30, n_repeats=20, test_size=0.3, random_state=42):
    """
    Evaluates SVM accuracy using smoothed firing rate feature vectors.
    """
    # Convert all rasters to smoothed 1D feature vectors
    X = np.array([raster_to_svm_features(r, sigma) for r in rasters])
    y = np.array(labels)
    
    unique_labels = np.unique(y)
    splitter = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=random_state)
    
    # Standard linear SVM pipeline
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=1.0)
    )

    scores = []
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in tqdm(splitter.split(X, y), total=n_repeats, desc="SVM (rhv)"):
        svm.fit(X[train_idx], y[train_idx])
        preds = svm.predict(X[test_idx])

        scores.append(accuracy_score(y[test_idx], preds))
        y_true_all.append(y[test_idx])
        y_pred_all.append(preds)

    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores, ddof=1))

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=unique_labels)

    return mean_acc, std_acc, cm