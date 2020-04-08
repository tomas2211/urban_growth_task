import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import data_utils, dataset

'''
Script to create data insights visualizations.
Before execution, set the appropriate input and output paths below.
'''

data_folder = 'data'  # Folder containing provided dataset
vis_folder = 'visualizations'  # Output folder
sites = [1, 12]  # Available sites


def count_classes(labels_array):
    # Count classes distribution in an array of given labels
    n_veg = 0
    n_urb = 0
    n_cld = 0

    for lab in labels_array:
        n_veg += np.sum(lab == 0)  # non-urb
        n_urb += np.sum(lab == 1)  # urban
        n_cld += np.sum(lab == 2)  # clouds
    return n_veg, n_urb, n_cld


def plot_classes_distribution(labels_list, ax, title):
    n_veg, n_urb, n_cld = count_classes(labels_list)

    ax.bar([0, 1, 2], [n_veg, n_urb, n_cld])
    ax.set_title(title)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Non-urb', 'Urban', 'Cloud'])
    mkfunc = lambda x, pos: '%1.1fM' % (x * 1e-6) if x >= 1e6 else \
        '%1.0fK' % (x * 1e-3) if x >= 1e3 else '%1.0f' % x
    mkformatter = matplotlib.ticker.FuncFormatter(mkfunc)
    ax.yaxis.set_major_formatter(mkformatter)
    ax.set_ylabel('# of pixels')

if __name__ == '__main__':
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)

    # Load the data
    data_dict = data_utils.load_data(data_folder, sites)

    # Create plots showing the time distribution and label availability
    for site in sites:
        site_data = data_dict[site]
        num_imgs = len(site_data)
        num_labels = len(
            [lab for _, lab, _ in site_data if lab is not None]
        )  # Count labeled images
        timestamps = [timest for _, _, timest in site_data]

        # Assign colors to each sample - red/green based on label availability
        colors = ['#ff0000' if lab is None else '#00ff00'
                  for _, lab, _ in site_data]

        fig, ax = plt.subplots(figsize=(9, 1), constrained_layout=True)
        ax.set(title="Site {} - available data (labeled {} of {})".format(
            site, num_labels, num_imgs))
        ax.scatter(timestamps, [site] * len(timestamps), s=10, c=colors)

        # Set x ticks - month+year
        ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=2))
        ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        ax.get_yaxis().set_visible(False)  # Hide Y axis
        fig_fn = os.path.join(vis_folder,
                              'time_dist_s{:02d}.png'.format(site))
        plt.savefig(fig_fn)
        plt.close()

    # Distribution of labeled classes per site
    fig, axs = plt.subplots(1, 2, figsize=(5, 3))

    for site, ax in zip(sites, axs):
        site_labels = [lab for _, lab, _ in data_dict[site] if lab is not None]
        plot_classes_distribution(site_labels, ax, 'Site {}'.format(site))

    plt.tight_layout()
    plt.savefig(os.path.join(vis_folder, 'label_dist_sites.png'))
    plt.close()

    # Distribution of classes in the training/testing data
    fig, axs = plt.subplots(1, 3, figsize=(7.5, 3))

    def collect_labels(data_indexes_dict):
        labels_list = []
        for site in data_indexes_dict:
            for idx in data_indexes_dict[site]:
                _, lab, _ = data_dict[site][idx]
                labels_list.append(lab)
        return labels_list

    plot_classes_distribution(
        collect_labels(dataset.train_data), axs[0], 'Training (site 1)')
    plot_classes_distribution(
        collect_labels(
            dataset.test_data['test_site1']), axs[1], 'Test - site 1')
    plot_classes_distribution(
        collect_labels(
            dataset.test_data['test_site12']), axs[2], 'Test - site 12')

    plt.tight_layout()
    plt.savefig(os.path.join(vis_folder, 'label_dist_split.png'))
    plt.close()

