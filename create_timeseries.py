import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import os
from utils.data_utils import load_data
from utils.model import SKModel

'''
Load the segmentation model, evaluate it on the data and aggregate 
the urbanization index into timeseries.
'''


def uindex_from_probabilities(probs):
    h, w, _ = probs.shape
    prob_veg = np.sum(probs[:, :, 0])
    prob_urb = np.sum(probs[:, :, 1])

    uindex = prob_urb / (prob_veg + prob_urb)

    # Cloud cover - count the labeled pixels
    cloud_cover = np.sum(np.argmax(probs, axis=2) == 2) / (h*w)

    return uindex, cloud_cover


def uindex_from_labels(labels):
    h, w = labels.shape
    n_veg = np.sum(labels == 0)  # non-urb
    n_urb = np.sum(labels == 1)  # urban
    n_cld = np.sum(labels == 2)  # clouds

    if n_urb + n_veg > 0:
        uindex = n_urb / (n_urb + n_veg)
    else:
        uindex = 0

    cloud_cover = n_cld / (h * w)

    return uindex, cloud_cover


def datetime_to_days(series_time):
    series_time = series_time - series_time[0]
    ret = np.zeros_like(series_time, dtype=np.int)
    for i in range(len(series_time)):
        ret[i] = series_time[i].days

    return ret


def filter_outliers(x, y, meanwindow=90, thresh=0.05):
    # filter outliers
    fil_x = np.array([])
    fil_y = np.array([])
    fil_mask = np.zeros_like(x, dtype=np.bool)

    for i in range(len(x)):
        # calculate local mean
        mask = (x > (x[i] - meanwindow)) & (x < (x[i] + meanwindow))
        accept = False

        if np.any(mask):
            avg = np.median(y[mask])
            if (avg + thresh) > y[i] > (avg - thresh):  # in the range
                accept = True
        else:
            accept = True

        fil_mask[i] = accept
        if accept:
            fil_x = np.hstack((fil_x, x[i]))
            fil_y = np.hstack((fil_y, y[i]))

    return fil_x, fil_y, fil_mask


def gaus_filter_timeseries(x, y, sigma=60.0):
    y_fil = np.zeros_like(y)
    for i in range(len(x)):
        dists = x[i] - x  # distances to current sample
        weights = np.exp(-0.5 * dists ** 2 / sigma ** 2)
        weights = weights / np.sum(weights)
        y_fil[i] = np.sum(y * weights)

    return y_fil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Path to checkpoint.')
    parser.add_argument('--data_folder', default='data', help='Folder with the dataset (with /imgs and /labels folders).')
    parser.add_argument('--out_folder', required=True, help="Folder to save images.")
    parser.add_argument('--device', type=str, default='cpu', help='cpu|cuda')
    parser.add_argument('--cloud_cover_threshold', type=float, default=0.1, help='Cloud cover threshold.')
    parser.add_argument('--filter_outlier_avgwin', type=int, default=180, help='Window over which to compute the average for outlier detection [days].')
    parser.add_argument('--filter_outlier_thresh', type=int, default=0.1, help='Threshold for outliers (max deviation from mean).')
    parser.add_argument('--filter_sigma', type=float, default=60.0, help='Sigma for gaussian filter [days].')
    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Load the checkpoint
    model = SKModel(0).to(args.device)
    model.load_state_dict(
        torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    )
    model.eval()

    images = load_data(args.data_folder)  # Load data

    for site in images.keys():
        images_site = images[site]

        series_time = []
        series_uindex_prob = []

        gt_series_time = []  # Ground-truth index and timestamps
        gt_series_uindex = []

        # Evaluate the index for each observation
        for im, lab_gt, tim in tqdm.tqdm(images_site):
            im_t = torch.from_numpy(
                im.astype(np.float32) / 255.0).permute(2, 0, 1)
            _, h, w = im_t.shape

            with torch.no_grad():
                output = model(im_t[None, :])

            labels = torch.argmax(output[0], dim=0).cpu().numpy()
            probs = output[0].detach().cpu().numpy().transpose(1, 2, 0)

            uindex_prob, cloud_cover = uindex_from_probabilities(probs)

            if cloud_cover < args.cloud_cover_threshold:
                series_time.append(tim)
                series_uindex_prob.append(uindex_prob)

            if lab_gt is not None:
                gt_uindex, gt_cloud_cover = uindex_from_labels(lab_gt)
                if gt_cloud_cover < args.cloud_cover_threshold:
                    gt_series_time.append(tim)
                    gt_series_uindex.append(gt_uindex)

        # Filtering - outlier detection and gaussian
        series_t = datetime_to_days(np.array(series_time))
        series_p = np.array(series_uindex_prob)
        series_t, series_p, filter_mask =\
            filter_outliers(
                series_t, series_p, meanwindow=args.filter_outlier_avgwin,
                thresh=args.filter_outlier_thresh)

        series_fil_p = gaus_filter_timeseries(
            series_t, series_p, sigma=args.filter_sigma)
        series_fil_t = np.array(series_time)[filter_mask]

        # Create plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set(title="Site {} - urban index".format(site))

        ax.plot(gt_series_time, gt_series_uindex, c='r', linewidth=1,
                linestyle='--', marker='.', markersize=5)
        ax.scatter(series_time, series_uindex_prob, marker='x', c='g')
        ax.plot(series_fil_t, series_fil_p, c='b', linewidth=1,
                linestyle='--', marker='.', markersize=5)
        # ax.scatter(gt_series_time, gt_series_uindex, s=10, c='r')

        # set x ticks - month+year
        ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=2))
        ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        plt.xlabel('Time')
        plt.ylabel('Urban index')
        plt.legend(['Ground Truth', 'Filtered estimates', 'Raw estimates'])
        plt.ylim([0, 1])
        plt.tight_layout()

        plt.savefig(
            os.path.join(args.out_folder, 'timeseries_site%02d.png' % site))
        plt.close()
