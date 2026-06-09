import _init_paths
import matplotlib.pyplot as plt

from lib.test.evaluation.data import SequenceList
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'mvtd'
target_sequences = ["2-ship", "7-Boat", "10-USV"]  # Replace with actual target sequence names
"""hiptrack"""
trackers.extend(trackerlist(name='hiptrack', parameter_name='hiptrack', dataset_name=dataset_name,
                            run_ids=None, display_name='HIPTrack'))
dataset = get_dataset('mvtd')
dataset = SequenceList([seq for seq in dataset
                        if seq.name in target_sequences])
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
#plot_results(trackers, dataset, 'otb', merge_results=True, plot_types=('success', 'prec','norm_prec'),
#             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'),force_evaluation=True)
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
