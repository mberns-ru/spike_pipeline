from kilosort import run_kilosort
from pathlib import Path
from numpy import array
from numpy import inf

# Loop through all data folders and run Kilosort
def sort_spikes(base_dir):

    # Specify the directory containing the .bin files
    # e.g. E:/NH/On-Task/
    base_dir = Path(base_dir)

    # Create an array of Path objects for all .bin files in the directory and its subdirectories
    bin_files = list(base_dir.rglob("*\\*\\*\\*.bin"))
    print(bin_files)

    # Print the array of Path objects
    for data_file in bin_files:
        if 'F57' or 'F73' in str(data_file):
            # PC; use neuronexus probe
            print(f"Processing {data_file} as PC")
            try:
                run_pc(data_file)
            except:
                print(f"Error processing {data_file}")
        else:
            # AC; use cambridge probe
            print(f"Processing {data_file} as AC")
            try:
                run_ac(data_file)
            except:
                print(f"Error processing {data_file}")
    return

def run_pc(data_file):
    settings = { 'probe_name': 'pc.prb',
    'n_chan_bin': 64,
    'data_dtype': 'float32',
    'fs': 24414.1,
    'batch_size': 60000,
    'nblocks': 0,
    'Th_universal': 9.0,
    'Th_learned': 8.0,
    'tmin': 0.0,
    'tmax': inf,
    'nt': 61,
    'shift': None,
    'scale': 1000.0,
    'artifact_threshold': inf,
    'nskip': 25,
    'whitening_range': 32,
    'highpass_cutoff': 300.0,
    'binning_depth': 5.0,
    'sig_interp': 20.0,
    'drift_smoothing': [0.5, 0.5, 0.5],
    'nt0min': 20,
    'dmin': None,
    'dminx': 32.0,
    'min_template_size': 10.0,
    'template_sizes': 5,
    'nearest_chans': 10,
    'nearest_templates': 100,
    'max_channel_distance': None,
    'templates_from_data': True,
    'n_templates': 6,
    'n_pcs': 6,
    'Th_single_ch': 6.0,
    'acg_threshold': 0.2,
    'ccg_threshold': 0.25,
    'cluster_downsampling': 20,
    'x_centers': None,
    'duplicate_spike_ms': 0.25,
    'save_preprocessed_copy': False,
    'clear_cache': False,
    'do_CAR': True,
    'invert_sign': False,
    'data_dir': data_file.parent,
    'filename': data_file,
    'NTbuff': 60122,
    'Nchan': 64,
    'duplicate_spike_bins': 6,
    'chanMap': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]),
    'xc': array([200.  ,   0.  ,   0.  , 400.  , 217.32, 200.  ,  17.32, 400.  ,
        17.32,   0.  ,   0.  , 417.32, 417.32, 417.32,  17.32,  17.32,
       217.32, 200.  ,   0.  , 200.  , 417.32, 400.  , 217.32,  17.32,
       217.32, 200.  ,   0.  , 400.  , 200.  , 217.32, 217.32,  17.32,
       417.32, 600.  , 800.  , 400.  , 408.66, 400.  , 617.32, 817.32,
       617.32, 600.  , 800.  , 617.32, 417.32, 600.  , 617.32, 817.32,
       408.66, 800.  , 800.  , 617.32, 617.32, 600.  , 817.32, 817.32,
       600.  , 800.  , 800.  , 408.66, 600.  , 408.66, 817.32, 817.32],
      dtype='float32'),
    'yc': array([  50.,  110.,   30.,   50.,   20.,   30.,   20.,   70.,  100.,
         10.,   90.,   80.,   60.,   40.,    0.,   80.,   40.,  110.,
         70.,   70.,  100.,   90.,  100.,   60.,   60.,   90.,   50.,
        110.,   10.,    0.,   80.,   40.,   20.,   30.,   70.,   30.,
       1110.,   10.,   20.,   60.,   60.,   10.,   50.,   80.,    0.,
         90.,    0.,   40.,  360.,  110.,   30.,  100.,   40.,  110.,
        100.,   20.,   70.,   90.,   10.,  860.,   50.,  610.,   80.,
          0.], dtype='float32'),
    'kcoords': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'),
    'n_chan': 64}

    results = run_kilosort(settings, data_dtype='float32')
    return

def run_ac(data_file):
    settings = {'probe_name': 'ac.prb',
    'n_chan_bin': 64,
    'data_dtype': 'float32',
    'fs': 24414.0,
    'batch_size': 60000,
    'nblocks': 0,
    'Th_universal': 9.0,
    'Th_learned': 8.0,
    'tmin': 0.0,
    'tmax': inf,
    'nt': 61,
    'shift': None,
    'scale': 1000.0,
    'artifact_threshold': inf,
    'nskip': 25,
    'whitening_range': 32,
    'binning_depth': 5.0,
    'sig_interp': 20.0,
    'drift_smoothing': [0.5, 0.5, 0.5],
    'nt0min': 20,
    'dmin': None,
    'dminx': 32.0,
    'min_template_size': 10.0,
    'template_sizes': 5,
    'nearest_chans': 10,
    'nearest_templates': 100,
    'max_channel_distance': None,
    'templates_from_data': True,
    'n_templates': 6,
    'n_pcs': 6,
    'Th_single_ch': 6.0,
    'acg_threshold': 0.2,
    'ccg_threshold': 0.25,
    'cluster_downsampling': 20,
    'x_centers': None,
    'duplicate_spike_bins': 7,
    'save_preprocessed_copy': False,
    'data_dir': data_file.parent,
    'filename': data_file,
    'do_CAR': True,
    'invert_sign': False,
    'NTbuff': 60122,
    'Nchan': 64,
    'chanMap': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]),
    'xc': array([750. , 750. , 500. , 772.5, 750. , 750. , 500. , 522.5, 750. ,
       522.5, 772.5, 500. , 772.5, 500. , 500. , 772.5, 500. , 522.5,
       772.5, 750. , 500. , 522.5, 522.5, 772.5, 772.5, 522.5, 522.5,
       772.5, 750. , 750. , 272.5, 500. ,  22.5, 272.5, 250. , 250. ,
       250. ,  22.5, 272.5, 272.5,   0. , 272.5,  22.5,   0. , 522.5,
         0. , 272.5,  22.5,  22.5, 250. ,  22.5,   0. ,   0. ,   0. ,
       272.5,  22.5, 250. , 250. ,   0. , 272.5,  22.5, 250. , 250. ,
         0. ], dtype='float32'),
    'yc': array([125. ,  50. ,   0. ,  62.5, 100. ,   0. ,  75. , 137.5,  75. ,
        37.5,  37.5,  50. , 112.5,  25. , 100. , 137.5, 175. ,  87.5,
        87.5, 150. , 125. , 187.5, 162.5, 162.5,  12.5,  62.5,  12.5,
       187.5, 175. ,  25. , 137.5, 150. ,  62.5, 112.5, 150. ,  25. ,
        50. , 112.5, 162.5,  12.5, 125. ,  87.5, 162.5,   0. , 112.5,
       100. ,  62.5,  87.5, 187.5, 100. , 137.5, 175. , 150. ,  25. ,
        37.5,  37.5, 175. ,  75. ,  75. , 187.5,  12.5, 125. ,   0. ,
        50. ], dtype='float32'),
    'kcoords': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'),
    'n_chan': 64}

    results = run_kilosort(settings, data_dtype='float32')
    return