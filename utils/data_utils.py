import glob
import os
from datetime import datetime

import imageio
import numpy as np


def load_data(path, sites=None):
    '''
    Load the given task data.
    :param path: Path pointing to a folder that contains 'imgs/' and 'labels/'
     folders.
    :param sites: List of site numbers (default 1 and 12).
    :return: Dictionary containing a list of tuples (image, label, timestamp)
     for each site.
    '''
    if sites is None:
        sites = [1, 12]
    timestamp_format = "%Y-%m-%d"  # Format of timestamps in filenames

    # Scan the folders and load data
    data_dict = {}
    for site in sites:
        data_dict[site] = []
        found_filenames = sorted(glob.glob(
            os.path.join(path, 'imgs/set-%05d/*.npz' % site)))

        for image_fn in found_filenames:
            with np.load(image_fn) as data:  # Load the image
                image = data['arr_0']

            timstmp = datetime.strptime(os.path.basename(image_fn)[3:13],
                                        timestamp_format)  # Parse timestamp

            # Change to image_fn to label filename
            label_fn = \
                image_fn.replace('imgs', 'labels')[:-4] + '_labels.png'
            label = None
            if os.path.exists(label_fn):
                label = imageio.imread(label_fn)
                label = convert_label(label)

            data_dict[site].append((image, label, timstmp))

    return data_dict


def convert_label(label):
    '''
    Convert label from a colored image to an array with indexes.
    Assigned indexes:
         - 0 = non-urban (green in image)
         - 1 = urban (red in image)
         - 2 = cloud (blue in image)
    :param label: numpy array with input colored image (h x w x c)
    :return: numpy array with indexes (h x w)
    '''
    ret_label = np.zeros(label.shape[:2], dtype=np.int32)
    ret_label[label[:, :, 0] > 127] = 1  # red - urban
    ret_label[label[:, :, 1] > 127] = 0  # green - non-urban
    ret_label[label[:, :, 2] > 127] = 2  # cloud
    return ret_label

