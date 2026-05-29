from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/ivl5/hiptrack/HIPTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/ivl5/hiptrack/HIPTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/ivl5/hiptrack/HIPTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/home/ivl5/hiptrack/HIPTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/ivl5/hiptrack/HIPTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/ivl5/hiptrack/HIPTrack/data/lasot'
    settings.network_path = '/home/ivl5/hiptrack/HIPTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/ivl5/hiptrack/HIPTrack/data/nfs'
    settings.otb_path = '/home/ivl5/hiptrack/HIPTrack/data/otb'
    settings.prj_dir = '/home/ivl5/hiptrack/HIPTrack'
    settings.result_plot_path = '/home/ivl5/hiptrack/HIPTrack/output/test/result_plots'
    settings.results_path = '/home/ivl5/hiptrack/HIPTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/ivl5/hiptrack/HIPTrack/output'
    settings.segmentation_path = '/home/ivl5/hiptrack/HIPTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/ivl5/hiptrack/HIPTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/ivl5/hiptrack/HIPTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/ivl5/hiptrack/HIPTrack/data/trackingnet'
    settings.uav_path = '/home/ivl5/hiptrack/HIPTrack/data/uav'
    settings.vot18_path = '/home/ivl5/hiptrack/HIPTrack/data/vot2018'
    settings.vot22_path = '/home/ivl5/hiptrack/HIPTrack/data/vot2022'
    settings.vot_path = '/home/ivl5/hiptrack/HIPTrack/data/VOT2019'
    settings.youtubevos_dir = ''
    settings.mvtd_path = '/home/ivl5/hiptrack/HIPTrack/data/mvtd/train'
    return settings

