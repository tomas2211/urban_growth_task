import colorcet
import numpy as np
import torch
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader

from utils import dataset as dataset
from utils.visualizations import plot_confusion_matrix, lab2color


def fetch_train_dataset(data_path, batch_size, args):
    # Prepare training dataset
    af_aug = iaa.Affine(  # Define augmentations
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        rotate=(-180, 180), shear=(-10, 10), mode='reflect'
    )
    af_aug._mode_segmentation_maps = 'reflect'
    cr_h, cr_w = args.crop_size

    aug_list = [
        iaa.HorizontalFlip(0.5),
        iaa.VerticalFlip(0.5),
        af_aug,
        iaa.CropToFixedSize(cr_h, cr_w, position='uniform')
    ]

    if args.color_augs:  # Optionally add color augmentations
        aug_list += [
            iaa.AdditiveGaussianNoise(scale=5, per_channel=True),
            iaa.Add(value=(-15, 15), per_channel=True),
            iaa.Multiply(mul=(0.8, 1.2), per_channel=True),
            iaa.SaltAndPepper(p=0.0001)
        ]

    train_aug = iaa.Sequential(aug_list)
    train_dataset = dataset.SKDataset(
        data_path, dataset.train_data, train_aug, invalidate_boudnaries=True,
        multiplier=args.dat_muliplier
    )
    print('Train dataset - %d samples' % len(train_dataset))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    return train_loader


def fetch_test_datasets(data_path):
    # Testing datasets
    test_loaders = {}
    for test in dataset.test_data:
        test_dataset = dataset.SKDataset(
            data_path, dataset.test_data[test], None,
            invalidate_boudnaries=False
        )
        print('Test [%s] dataset - %d samples' % (test, len(test_dataset)))
        test_loaders[test] = DataLoader(
            test_dataset, batch_size=1, pin_memory=True, shuffle=False)

    # Dataset with images for visualization
    vis_dataset = dataset.SKDataset(
        data_path, dataset.vis_examples, None, invalidate_boudnaries=False)
    vis_loader = DataLoader(
        vis_dataset, batch_size=1, pin_memory=True, shuffle=False)

    return test_loaders, vis_loader


def get_next_sample(train_loader, train_iter):
    # Helper for PyTorch DataLoader class - renew iterator when needed.
    if train_iter is None:
        train_iter = train_loader.__iter__()

    try:
        smp = train_iter.__next__()
    except StopIteration:
        train_iter = train_loader.__iter__()
        smp = train_iter.__next__()

    return train_iter, smp


def evaluate(model, test_loaders, device):
    # Calculate precision, recall and F1 for each test set and generate
    # a confusion matrix.
    model.eval()
    target_names = ['Non-urban', 'Urban', 'Cloud']

    ret_prec_rec_fscores = {}
    ret_figures = {}

    for test in test_loaders:
        pred_all = []
        labels_all = []

        # Evaluate model
        for i, input_dict in enumerate(test_loaders[test]):
            with torch.no_grad():
                out = model(input_dict['im'].to(device))
                pred = torch.argmax(out, dim=1).reshape(-1).cpu().numpy()

            pred_all.append(pred)
            label = input_dict['lab'].reshape(-1).numpy()
            labels_all.append(label)

        pred_all = np.concatenate(pred_all)  # Concat predictions to one array
        labels_all = np.concatenate(labels_all)

        # Precision, recall and F1 for each class
        prec_rec_fscore = metrics.precision_recall_fscore_support(
            labels_all, pred_all, labels=[0, 1, 2])

        ret_prec_rec_fscores[test] = prec_rec_fscore

        # Confusion matrix
        fig = plt.figure(figsize=(4, 4))
        confmat = metrics.confusion_matrix(
            labels_all, pred_all, labels=[0, 1, 2])
        plot_confusion_matrix(
            confmat, target_names, fig.gca(), normalize=False,
            cmap=colorcet.m_bmy, title=test)
        fig.tight_layout()
        ret_figures[test] = fig

    return ret_prec_rec_fscores, ret_figures


def generate_visualizations(model, vis_loader, device):
    # Create example visualization images
    model.eval()
    ret_figures = []
    for i, input_dict in enumerate(vis_loader):
        im = input_dict['im']
        with torch.no_grad():
            out = model(im.to(device))
            label_probability, label_prediction = torch.max(out[0], dim=0)

        # label_probability = label_probability.cpu().numpy()
        label_prediction = label_prediction.detach().cpu().numpy()
        label_prediction = lab2color(label_prediction)

        rgb_image = im[0].permute(1, 2, 0).cpu().numpy()[:, :, 3:0:-1]
        for c in [0,1,2]:
            rgb_image[:,:,c] -= np.min(rgb_image[:,:,c])
            rgb_image[:,:,c] /= np.max(rgb_image[:,:,c])

        # Create figure with all visualizations
        fig, axs = plt.subplots(1, 3, figsize=(6, 2))
        axs[0].imshow(rgb_image)
        axs[0].set_title('RGB')
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        axs[1].imshow(label_prediction)
        axs[1].set_title('Prediction')
        axs[1].set_xticks([])
        axs[1].set_yticks([])

        if 'lab' in input_dict:
            label_gt = input_dict['lab'][0].numpy()
            label_gt = lab2color(label_gt)
            axs[2].imshow(label_gt)
            axs[2].set_title('GT')
            axs[2].set_xticks([])
            axs[2].set_yticks([])
        else:
            axs[2].set_visible(False)  # Remove axes when no label

        # im_prob = axs[3].imshow(label_probability, cmap=colorcet.m_bmy)
        # axs[3].set_title('Probability')

        # divider = make_axes_locatable(axs[3])  # Colorbar for probability
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(im_prob, cax=cax, orientation='vertical')
        fig.tight_layout()
        ret_figures.append(fig)

    return ret_figures