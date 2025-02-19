import spikeinterface.full as si
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import probeinterface as pi
from probeinterface.plotting import plot_probe
import pathlib

import warnings
warnings.simplefilter("ignore")

matplotlib.rcParams['pdf.fonttype'] = 42

# Running program
def run_spike_sorting(tdt_folder, output_folder, name_scheme, sel_probe, snip=False, overwrite=False):
    # print(si.get_sorter_params_description('kilosort4'))
    if sel_probe == 'neuronexus':
        probe = load_neuronexus_probe(output_folder)
    else:
        probe = load_probe(output_folder)
    full_raw_rec = load_rec(tdt_folder)
    #probe = load_probe(output_folder)
    if snip:
        print("Snipping data...")
        fs = full_raw_rec.get_sampling_frequency()
        recording_sub = full_raw_rec.frame_slice(start_frame=0*fs, end_frame=100*fs)
        recording_sub
    else:
        recording_sub=full_raw_rec
    p_full_raw_rec = recording_sub.set_probe(probe, group_mode="by_shank")

    # Create folder to save files
    results_folder = create_output_folder(output_folder, name_scheme)

    # Export to binary
    print("Exporting to binary...")
    si.write_binary_recording(p_full_raw_rec, str(results_folder) +  '\\' + name_scheme + '-binary.bin', dtype='float32')

    # Plot traces
    plt.ion()
    si.plot_traces(p_full_raw_rec, backend="matplotlib", mode='line', channel_ids=p_full_raw_rec.channel_ids[::])
    plt.savefig(str(results_folder) + '/' + name_scheme + '-traces.pdf', format='pdf')
    print("Traces plotted and saved.")
    plt.close()
    """plt.show()
    plt.pause(0.001)"""
    
    # Preprocess
    """recording_processed, recording_f = preprocess_rec(p_full_raw_rec, name_scheme, results_folder)
    w = si.plot_traces({"raw": p_full_raw_rec, "filtered": recording_f, "common": recording_processed}, mode='map',
                   time_range=[10, 10.1], backend="ipywidgets")"""

    # Spike sorting
    #sorted_data = spike_sort(p_full_raw_rec, results_folder, name_scheme)

    # Export to Phy
    #export_to_phy(p_full_raw_rec, sorted_data, results_folder)

    return
    
# Crrate results folder
def create_output_folder(output_folder, name_scheme):
    name_res = "results-" + name_scheme
    output_folder = Path(output_folder) / name_res
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

# Loading in file
def load_rec(tdt_folder):
    print("Finding file...")
    for file_name in os.listdir(tdt_folder):
        if file_name.endswith('.Tbk'):
            print("File found: ", file_name)
            file_path = os.path.join(tdt_folder, file_name)
            break
    print("Loading file...")
    try:
        full_raw_rec = si.read_tdt(folder_path=file_path, stream_name = "b'RSn1'")
    except:
        full_raw_rec = si.read_tdt(folder_path=file_path, stream_name = "RSn1")
    fs = full_raw_rec.get_sampling_frequency()
    full_trace = full_raw_rec.get_traces()
    print('Traces shape:', full_trace.shape)
    return full_raw_rec

# Load probe
def load_probe(output_folder):
    output_folder = pathlib.Path(output_folder)
    manufacturer = 'cambridgeneurotech'
    probe_name = 'ASSY-276-P-1'

    probe = pi.get_probe(manufacturer, probe_name)

    # Set wiring:
    device_channel_indices = [
        8,4,12,5,22,6,21,17,13,14,28,1,30,25,29,20,52,53,54,61,49,41,38,36,62,57,60,44,37,46,45,33,
        2,3,9,7,11,10,16,15,19,18,24,23,27,26,32,31,35,34,40,39,43,42,48,47,51,50,56,55,59,58,64,63]

    new_device_channel_indices = [] # zero-indexed
    for id in device_channel_indices:
        new_device_channel_indices.append(id-1)

    probe.set_device_channel_indices(new_device_channel_indices)
    pi.write_probeinterface(output_folder/'cambridge_probe.json', probe)
    pi.write_probeinterface(output_folder/'cambridge_probe.prb', probe)

    return probe

def load_neuronexus_probe(output_folder):

    output_folder = pathlib.Path(output_folder)

    #probe = pi.generate_multi_shank(num_shank=5)
    probe = pi.Probe()
    x_coors = [0,17.32,0,17.32,0,17.32,0,17.32,0,17.32,0,17.32,200,217.32,200,217.32,200,217.32,
               200,217.32,200,217.32,200,217.32,400,417.32,400,417.32,400,417.32,400,417.32,400,
               417.32,400,417.32,408.66,408.66,408.66,408.66,600,617.32,600,617.32,600,617.32,600,
               617.32,600,617.32,600,617.32,800,817.32,800,817.32,800,817.32,800,817.32,800,817.32,800,817.32,]
    y_coors = [110,100,90,80,70,60,50,40,30,20,10,0,110,100,90,80,70,60,50,40,30,20,10,0,110,100,90,80,
               70,60,50,40,30,20,10,0,1110,860,610,360,110,100,90,80,70,60,50,40,30,20,10,0,110,100,90,
               80,70,60,50,40,30,20,10,0]
    shank_ids = [1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5]
    
    device_channel_indices = [
        2,9,11,16,19,24,27,32,3,7,10,15,18,23,26,31,20,25,1,17,6,5,29,30,28,21,22,12,8,13,4,14,
        36,33,38,45,37,60,62,49,54,52,46,44,57,41,61,53,34,39,42,47,50,55,58,63,35,40,43,48,51,56,59,64]
    probe.set_contacts(positions=np.array([x_coors, y_coors]).T, shank_ids=shank_ids)

    new_device_channel_indices = [] # zero-indexed
    for id in device_channel_indices:
        new_device_channel_indices.append(id-1)

    probe.set_device_channel_indices(new_device_channel_indices)

    pi.write_probeinterface(output_folder/'neuronexus_probe.json', probe)
    pi.write_probeinterface(output_folder/'neuronexus_probe.prb', probe)
    
    return probe

def spike_sort(preprocessed_data, results_folder, name_scheme):
    # convert to binary


    print("Spike sorting...")
    # print(si.get_sorter_params_description('spykingcircus2'))
    res_name = name_scheme + '-spike-sorted'
    # job_kwargs = dict(n_jobs=20, chunk_duration="1s", progress_bar=True)
    dtype = np.float32
    job_kwargs = dict(n_jobs=20, chunk_duration="1s", progress_bar=True)
    sorting_ks4 = si.run_sorter(sorter_name='kilosort4', recording=preprocessed_data, remove_existing_folder=True,
                                output_folder=results_folder / name_scheme,
                                verbose=True, scale=1000, nblocks=0)
    # sorting_sk2 = si.run_sorter(sorter_name='spykingcircus2', recording=preprocessed_data, output_folder=results_folder/res_name, apply_preprocessing=True, remove_existing_folder=True)

    plt.ion()
    si.plot_rasters(sorting_ks4, backend="matplotlib")
    plt.title("KS4 Rasters")
    plt.savefig(str(results_folder) + '/' + name_scheme + '-ks4-rasters.pdf', format='pdf')
    plt.show()
    plt.pause(0.001)
    return sorting_ks4

def preprocess_rec(recording, name_scheme, results_folder):
    print("Preprocessing...")

    # Filtering
    recording_filt = si.bandpass_filter(recording, 300, 5000) # applies bandpass filter
    recording_cmr = si.common_reference(recording_filt, reference='global', operator='median') # apply common median reference; noise reduction, re-references all channels against global median
    plt.ion()
    si.plot_traces(recording_cmr, mode='line', backend="matplotlib", channel_ids=recording_cmr.channel_ids[::])
    plt.title("Filtered and Common Median Reference")
    plt.savefig(str(results_folder) + '/' + name_scheme + '-filtered-cmr.pdf', format='pdf')
    plt.show()
    plt.pause(0.001)

    bad_channel_ids, channel_labels = si.detect_bad_channels(recording_filt, method='coherence+psd')
    print('bad_channel_ids', bad_channel_ids)
    print('channel_labels', channel_labels)

    return recording_cmr, recording_filt

def export_to_phy(recording_saved, sorting, results_folder):
    print("Exporting to Phy...")
    # recording_saved.annotate(is_filtered=True)

    # ok now I'll export:
    cleaned = si.remove_excess_spikes(sorting, recording_saved)
    we = si.extract_waveforms(recording_saved, cleaned, folder=results_folder / "waveforms_sparse", overwrite=True)
    si.export_to_phy(we, output_folder=results_folder / 'phy_ks4', 
                 compute_amplitudes=False, compute_pc_features=False, copy_binary=False, remove_if_exists=True)
    return