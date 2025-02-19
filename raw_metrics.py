import os
from pathlib import Path
import glob

import pandas as pd
import numpy as np
from scipy.stats import variation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
from tqdm import tqdm

def generate_metrics(session_folder, output_folder, hearing_status='NH'):
    """
    Generate summary metrics for a session.
    Input:  session_folder (str) - the folder containing the session data
            output_folder (str) - the folder containing all the animal IDs, output data, etc.
    Output Folder structure:
            animal_ID > ontask/passive > NH/NIHL > session_X > raw_metrics > unit_x
    """

    # Extract metadata from the session folder
    if 'HL' in session_folder or 'hl' in session_folder:
        hearing_status = 'NIHL'
    session_info = session_folder.split('/')[-1]
    session_info = session_info.split('-')
    animal_ID = session_info[-3] # e.g. J_F57_Back
    animal_ID = animal_ID.split('_')[1] # e.g. F57
    session_type = session_info[-4] # e.g. ontask
    date = session_info[-2] # e.g. 240825
    
    print(f'Processing session {animal_ID} {date}')

    # Find necessary folders and files
    session_folder = Path(session_folder)
    kilosort_folder = session_folder / 'kilosort4' # kilosort folder, contains spike info
    behavior_file = glob.glob(str(session_folder) + '\\*.csv')[0] # behavior data
    times_file = str(kilosort_folder) + '/spike_times.npy' # spike times
    cluster_file = str(kilosort_folder) + '/spike_clusters.npy' # cluster per spike
    label_file = str(kilosort_folder) + '/cluster_group.tsv' # cluster group (e.g. good, noise, mua)

    # Load data
    behavior_data = pd.read_csv(behavior_file, encoding='UTF-8') # Behavior data as pandas dataframe
    spike_times = np.load(times_file) / 24.414 # Spike times in seconds
    cluster_assignments = np.load(cluster_file) # Cluster IDs for each spike
    cluster_labels = pd.read_csv(label_file, sep='\t')['group'].to_numpy() # Cluster labels (e.g. good, noise, mua)
    raw_cluster_ids = pd.read_csv(label_file, sep='\t')['cluster_id'].to_numpy() # Cluster ID per label

    # Create output folder structure if it doesn't exist
    output_folder = Path(output_folder)
    animal_folder = output_folder / animal_ID
    animal_folder.mkdir(parents=True, exist_ok=True)
    session_type_folder = animal_folder / session_type
    session_type_folder.mkdir(parents=True, exist_ok=True)
    hearing_status_folder = session_type_folder / hearing_status
    hearing_status_folder.mkdir(parents=True, exist_ok=True)
    session_folder = hearing_status_folder / f"session_{date}"
    session_folder.mkdir(parents=True, exist_ok=True)
    raw_metrics_folder = session_folder / 'raw_metrics'
    raw_metrics_folder.mkdir(parents=True, exist_ok=True)

    # Append spike times and cluster assignments to behavior data
    trial_spike_times = []
    trial_cluster_assignments = []
    for i, trial in behavior_data.iterrows():
        TrialStart = trial['TrialStart'] * 1000
        TrialEnd = trial['TrialEnd'] * 1000
        trial_spikes = spike_times[(spike_times >= TrialStart - 200) & (spike_times <= TrialEnd)]
        trial_spike_times.append(trial_spikes - TrialStart)
        trial_clusters = cluster_assignments[(spike_times >= TrialStart - 200) & (spike_times <= TrialEnd)]
        trial_cluster_assignments.append(trial_clusters)
    behavior_data['spike_times'] = trial_spike_times
    behavior_data['cluster_assignments'] = trial_cluster_assignments
    behavior_data.to_csv(session_folder / 'behavior_data.csv', index=False)

    # Generate csv of number of each label
    label_counts = np.unique(cluster_labels, return_counts=True)
    label_counts = pd.DataFrame({'label': label_counts[0], 'count': label_counts[1]})
    label_counts.to_csv(session_folder/'cluster_label_counts.csv')

    # Identify and process clusters that meet criteria
    for i, cluster in tqdm(enumerate(raw_cluster_ids), total=len(raw_cluster_ids), desc="Processing clusters"):
        label = cluster_labels[i]

        # Skip if noise
        if label not in ['good', 'mua']:
            continue

        # Skip if not present in >= 80% of trials
        trial_presence = [len(trial) for trial in trial_cluster_assignments]
        if sum([cluster in trial for trial in trial_cluster_assignments]) / len(trial_presence) < 0.8:
            continue

        # Process cluster
        process_cluster(cluster, behavior_data, raw_metrics_folder)

    return session_folder, animal_ID, date


def process_cluster(cluster, behavior_data, raw_metrics_folder):
    """
    For a given cluster, generate a csv with spike times per trial.
    Additionally, calculate the following metrics (graph + csv):
        1. Firing rate (Hz) x AM rate
        2. Spontaneous firing rate (Hz) x AM rate
        3. Standard deviation x AM rate
        4. PSTH for each AM rate
        5. Standard deviation of firing rate x AM rate
        6. Coefficient of variation (CV) x AM rate
        7. Vector strength x AM rate
    """

    # Create folder for cluster
    cluster_folder = raw_metrics_folder / f"unit_{cluster}"
    if not os.path.exists(cluster_folder):
        os.makedirs(cluster_folder)
    
    # Generate csv with:
    # - trial index
    # - spike times
    # - AM rate
    # - score
    # - choice

    # Filter behavior data for trials with the cluster; save to csv
    cluster_df = pd.DataFrame(columns=['trial', 'spike_times', 'AM_rate', 'score', 'choice'])
    #trials_with_cluster = behavior_data[behavior_data['cluster_assignments'].apply(lambda x: cluster in x)]
    for i, trial in behavior_data.iterrows():
        trial_index = i
        spike_times = trial['spike_times'][trial['cluster_assignments'] == cluster]
        if len(spike_times) == 0: # If no spikes, set to NaN
            spike_times = np.array([np.nan])
        AM_rate = trial['AMRate']
        score = trial['Score']
        choice = trial['Direction']
        curr_trial = pd.DataFrame({'trial': [trial_index], 'spike_times': [spike_times], 'AM_rate': [AM_rate], 
                                             'score': [score], 'choice': [choice]})
        cluster_df = pd.concat([cluster_df, curr_trial], ignore_index=True)
    cluster_df.to_csv(cluster_folder / f'cluster_{cluster}_data.csv', index=False)

    # Generate metrics
    metrics_df = pd.DataFrame(columns=['am_rate', 'mfr', 'sfr', 'std', 'std_firing_rate', 'cv', 'vs'])
    psths_by_rate = {}
    for AM_rate in np.sort(cluster_df['AM_rate'].unique()):
        AM_rate_df = cluster_df[cluster_df['AM_rate'] == AM_rate]
        mfr = calc_mfr(AM_rate_df)
        sfr = calc_sfr(AM_rate_df)
        std = calc_std(AM_rate_df)

        psth = calc_psth(AM_rate_df)
        psths_by_rate[AM_rate] = psth
        plt.title(f'PSTH for Cluster {cluster} at AM Rate {AM_rate}')
        plt.savefig(cluster_folder / f'cluster_{cluster}_psth_{AM_rate}.pdf', format='pdf')
        plt.close()

        std_firing_rate = calc_std_firing_rate(AM_rate_df)
        cv = calc_cv(AM_rate_df)
        vs = calc_vs(AM_rate_df)
        curr_metrics = pd.DataFrame({'am_rate': [AM_rate], 'mfr': [mfr], 'sfr': [sfr], 'std': [std], 'std_firing_rate': [std_firing_rate], 'cv': [cv], 'vs': [vs]})
        metrics_df = pd.concat([metrics_df, curr_metrics], ignore_index=True)

        # Plot raster
        graph_raster(AM_rate_df, cluster, AM_rate)
        plt.savefig(cluster_folder / f'cluster_{cluster}_raster_{AM_rate}.pdf', format='pdf')
        plt.close()

        # Plot PSTH
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(psth)), psth, width=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (Hz)')
        plt.title(f'PSTH for AM Rate {AM_rate}')
        plt.savefig(cluster_folder / f'cluster_{cluster}_psth_{AM_rate}.pdf', format='pdf')
        plt.close()

    # Plot all PSTHs in a stack
    # Plot all PSTHs in a stack
    num_rates = len(psths_by_rate)
    fig, axes = plt.subplots(num_rates, 1, figsize=(10, 6 * num_rates), sharex=True)
    for i, (AM_rate, psth) in enumerate(psths_by_rate.items()):
        axes[i].bar(np.arange(len(psth)), psth, width=1)
        axes[i].set_title(f'PSTH for AM Rate {AM_rate}')
        axes[i].set_ylabel('Spike Count')
    axes[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.savefig(cluster_folder / f'cluster_{cluster}_psth_stack.pdf', format='pdf')
    plt.close()
    
    # Plot each metric vs. AM rate
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['am_rate'], metrics_df['mfr'], label='Mean Firing Rate')
    plt.plot(metrics_df['am_rate'], metrics_df['sfr'], label='Spontaneous Firing Rate')
    plt.xscale('log')
    plt.xlabel('AM Rate')
    plt.ylabel('Firing Rate (Hz)')
    plt.title(f'Cluster {cluster} MFR and SFR vs. AM Rate')
    plt.legend()
    plt.savefig(cluster_folder / f'cluster_{cluster}_mfrsfr.pdf', format='pdf')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['am_rate'], metrics_df['std'], label='Standard Deviation')
    plt.xscale('log')
    plt.xlabel('AM Rate')
    plt.ylabel('Standard Deviation')
    plt.title(f'Cluster {cluster} Standard Deviation vs. AM Rate')
    plt.legend()
    plt.savefig(cluster_folder / f'cluster_{cluster}_std.pdf', format='pdf')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['am_rate'], metrics_df['std_firing_rate'], label='Standard Deviation of Firing Rate')
    plt.xscale('log')
    plt.xlabel('AM Rate')
    plt.ylabel('Standard Deviation of Firing Rate')
    plt.title(f'Cluster {cluster} Standard Deviation of Firing Rate vs. AM Rate')
    plt.legend()
    plt.savefig(cluster_folder / f'cluster_{cluster}_std_firing_rate.pdf', format='pdf')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['am_rate'], metrics_df['cv'], label='Coefficient of Variation')
    plt.xscale('log')
    plt.xlabel('AM Rate')
    plt.ylabel('Coefficient of Variation')
    plt.title(f'Cluster {cluster} Coefficient of Variation vs. AM Rate')
    plt.legend()
    plt.savefig(cluster_folder / f'cluster_{cluster}_cv.pdf', format='pdf')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['am_rate'], metrics_df['vs'], label='Vector Strength')
    plt.xscale('log')
    plt.xlabel('AM Rate')
    plt.ylabel('Vector Strength')
    plt.title(f'Cluster {cluster} Vector Strength vs. AM Rate')
    plt.legend()
    plt.savefig(cluster_folder / f'cluster_{cluster}_vs.pdf', format='pdf')
    plt.close()

    # Save metrics to csv
    metrics_df.to_csv(cluster_folder / f'cluster_{cluster}_metrics.csv', index=False)

    return 

##### METRIC GENERATIONS #####

def graph_raster(cluster_df, cluster, rate):
    """
    Generate raster plot for a cluster.
    Plot spike times vs. trial index.
    """
    plt.figure(figsize=(10, 6))
    i = 1
    for _, trial in cluster_df.iterrows():
        spike_times = trial['spike_times']
        if not np.isnan(spike_times).all():
            plt.scatter(spike_times, [i] * len(spike_times), s=10, c='black')
        i += 1
    plt.xlabel('Time (ms)')
    plt.ylabel('Trial')
    plt.title('Raster Plot for Cluster ' + str(cluster) + ' at ' + str(rate) + ' Hz')
    return

def calc_mfr(cluster_df):
    """
    Calculate mean firing rate (Hz) for a cluster.
    """
    # Positive spikes only
    cluster_df_copy = cluster_df.copy()
    cluster_df_copy['spike_times'] = cluster_df_copy['spike_times'].apply(lambda x: x[x > 0] if len(x[x > 0]) > 0 else np.array([np.nan]))

    total_spikes = sum([len(trial['spike_times']) for _, trial in cluster_df_copy.iterrows() if not np.isnan(trial['spike_times']).all()])
    total_time = sum([trial['spike_times'].max() for _, trial in cluster_df_copy.iterrows() if not np.isnan(trial['spike_times']).all()])
    return total_spikes / total_time * 1000 if total_time > 0 else 0

def calc_sfr(cluster_df):
    """
    Calculate spontaneous firing rate (Hz) for a cluster.
    """
    # Pre-trial spikes only
    cluster_df_copy = cluster_df.copy()
    cluster_df_copy['spike_times'] = cluster_df_copy['spike_times'].apply(lambda x: x[x < 0] if len(x[x < 0]) > 0 else np.array([np.nan]))

    total_spikes = sum([len(trial['spike_times']) for _, trial in cluster_df_copy.iterrows() if not np.isnan(trial['spike_times']).all()])
    return total_spikes / 200

def calc_std(cluster_df):
    """
    Calculate standard deviation for a cluster.
    """
    # Positive spikes only
    cluster_df_copy = cluster_df.copy()
    cluster_df_copy['spike_times'] = cluster_df['spike_times'].apply(lambda x: x[x > 0] if len(x[x > 0]) > 0 else np.array([np.nan]))

    spike_counts = [len(trial['spike_times']) for _, trial in cluster_df_copy.iterrows() if not np.isnan(trial['spike_times']).all()]
    return np.std(spike_counts)

def calc_psth(cluster_df, bin_size=25):
    """
    Calculate poststimulus time histogram for a cluster.
    """
    # Positive spikes only
    cluster_df_copy = cluster_df.copy()
    cluster_df_copy['spike_times'] = cluster_df_copy['spike_times'].apply(lambda x: x[x > 0] if len(x[x > 0]) > 0 else np.array([np.nan]))

    max_time = max([trial['spike_times'].max() for _, trial in cluster_df_copy.iterrows() if not np.isnan(trial['spike_times']).all()])
    bins = np.arange(0, max_time + bin_size, bin_size)
    psth, _ = np.histogram(np.concatenate([trial['spike_times'] for _, trial in cluster_df_copy.iterrows() if not np.isnan(trial['spike_times']).all()]), bins=bins)

    # Graph PSTH
    plt.figure(figsize=(10, 6))
    plt.bar(bins[:-1] * bin_size, psth, width=bin_size)
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing Rate (Hz)')
    return psth

def calc_std_firing_rate(cluster_df):
    """
    Calculate standard deviation of firing rate for a cluster.
    """
    # Positive spikes only
    cluster_df_copy = cluster_df.copy()
    cluster_df_copy['spike_times'] = cluster_df_copy['spike_times'].apply(lambda x: x[x > 0] if len(x[x > 0]) > 0 else np.array([np.nan]))

    firing_rates = [len(trial['spike_times']) / trial['spike_times'].max() for _, trial in cluster_df_copy.iterrows() if not np.isnan(trial['spike_times']).all()]
    return np.std(firing_rates) * 1000

def calc_cv(cluster_df):
    """
    Calculate coefficient of variation for a cluster.
    """
    # Positive spikes only
    cluster_df_copy = cluster_df.copy()
    cluster_df_copy['spike_times'] = cluster_df_copy['spike_times'].apply(lambda x: x[x > 0] if len(x[x > 0]) > 0 else np.array([np.nan]))

    firing_rates = [len(trial['spike_times']) / trial['spike_times'].max() for _, trial in cluster_df_copy.iterrows() if not np.isnan(trial['spike_times']).all()]
    return variation(firing_rates)

def calc_vs(cluster_df):
    """
    Calculate vector strength for a cluster.
    """
    # Positive spikes only
    cluster_df_copy = cluster_df.copy()
    cluster_df_copy['spike_times'] = cluster_df_copy['spike_times'].apply(lambda x: x[x > 0] if len(x[x > 0]) > 0 else np.array([np.nan]))

    vector_strengths = []
    for _, trial in cluster_df_copy.iterrows():
        spike_times = trial['spike_times']
        if not np.isnan(spike_times).all():
            period = 1 / trial['AM_rate']
            phases = (spike_times % period) / period * 2 * np.pi
            vector_strengths.append(np.abs(np.mean(np.exp(1j * phases))))
    return np.mean(vector_strengths) if vector_strengths else 0