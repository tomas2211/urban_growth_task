import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.utils.data as data
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from skimage.segmentation import find_boundaries

from utils import data_utils

'''
PyTorch dataset class for the SpaceKnow dataset and definition
 of training/testing data split.

When executed, provides examples of data augmetation.
'''

train_data = {
    1: [
        9, 11, 13,  # whole image occluded by cloud
        1, 5, 25,  # large cloud coverage
        15, 17,  # small clouds
        2, 3, 4, 6, 7, 16, 18, 19, 21, 23, 24, 28
    ]
}

test_data = {
    # 'test_all': {
    #     1: [
    #         27, 33,  # whole-image clouds
    #         20, 30,  # large clouds
    #         22, 24, 29,  # small clouds
    #         0, 34  # no occlusion
    #     ],
    #     12: [0, 1, 12, 20, 22, 25, 26, 32, 55, 59, 62, 64, 80, 88, 90]
    #     # all labeled from site 12
    # },
    # 'test_med_clouds': {1: [22, 24, 29, 30, 20]},
    # 'test_whole_clouds': {1: [27, 33]},
    'training_set': {
        1: [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 16, 17, 18, 19, 21, 23, 24, 25,
            28]
    },
    'test_site1': {
        1: [0, 20, 22, 24, 27, 29, 30, 33, 34]
    },
    'test_site12': {
        # all labeled from site 12
        12: [0, 1, 12, 20, 22, 25, 26, 32, 55, 59, 62, 64, 80, 88, 90]
    }
}

vis_examples = {
    1: [0, 20],
    12: [0, 83]
}


def do_invalidate_boundaries(lab, inval=-1):
    bnds = find_boundaries(lab, connectivity=2, mode='thick')
    lab[bnds] = inval
    return lab


class SKDataset(data.Dataset):
    def __init__(self, path, selection_dict, aug_seq: iaa.Sequential,
                 invalidate_boudnaries=False, multiplier=1):
        '''
        Create pytorch dataset object for the SpaceKnow dataset.
        :param path: path containing /imgs and /labels folders
        :param selection_dict: dictionary {site: [index1,...],...}
            specifying the data selection
        :param aug_seq: imaug augmentations sequence
        :param invalidate_boudnaries: invalidate boundaries between classes
            in labels
        :param multiplier: repeat the gathered images list
            (to allow for a large batch size)
        '''
        self.aug_seq = aug_seq
        self.invalidate_boudnaries = invalidate_boudnaries
        images = data_utils.load_data(path)

        self.images = []

        for site in selection_dict:
            for idx in selection_dict[site]:
                self.images.append(images[site][idx])

        self.images = self.images * multiplier

    def __getitem__(self, index):
        im, lab, tim = self.images[index]

        if lab is not None:
            lab = lab.copy()
            if self.invalidate_boudnaries:
                lab = do_invalidate_boundaries(lab, inval=5)

            if self.aug_seq is not None:
                segmap = SegmentationMapsOnImage(lab, shape=im.shape)
                im, lab = self.aug_seq(image=im, segmentation_maps=segmap)
                lab = lab.get_arr()

            lab[lab == 5] = -1

            im_t = torch.from_numpy(
                im.astype(np.float32) / 255.0).permute(2, 0, 1)
            lab_t = torch.from_numpy(lab[:, :]).type(torch.long)
            return {'im': im_t, 'lab': lab_t}

        im_t = torch.from_numpy(im.astype(np.float32) / 255.0).permute(2, 0, 1)
        return {'im': im_t}

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    # Showcase augmentations
    import matplotlib.pyplot as plt
    import colorcet


    def visualize(image, mask, original_image=None, original_mask=None):
        fontsize = 18

        if original_image is None and original_mask is None:
            f, ax = plt.subplots(1, 2, figsize=(8, 8))

            ax[0].imshow(image)
            ax[1].imshow(mask, cmap=colorcet.m_glasbey_category10)
        else:
            f, ax = plt.subplots(2, 2, figsize=(8, 8))

            ax[0, 0].imshow(original_image)
            ax[0, 0].set_title('Original image', fontsize=fontsize)

            ax[1, 0].imshow(original_mask, cmap=colorcet.m_bmy)
            ax[1, 0].set_title('Original mask', fontsize=fontsize)

            ax[0, 1].imshow(image)
            ax[0, 1].set_title('Transformed image', fontsize=fontsize)

            ax[1, 1].imshow(mask, cmap=colorcet.m_bmy)
            ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
        plt.show()


    images = data_utils.load_data('../data')
    im, lab, _ = images[1][5]

    # lab = data_utils.convert_label(lab)
    lab = do_invalidate_boundaries(lab, inval=5)

    af_aug = iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        rotate=(-180, 180),
        shear=(-10, 10),
        mode='reflect'
    )
    af_aug._mode_segmentation_maps = 'reflect'

    seq = iaa.Sequential([
        iaa.HorizontalFlip(0.5),
        iaa.VerticalFlip(0.5),
        af_aug,
        iaa.CropToFixedSize(30, 30, position='uniform'),
        iaa.AdditiveGaussianNoise(scale=5, per_channel=True),
        iaa.Add(value=(-15, 15), per_channel=True),
        iaa.Multiply(mul=(0.8, 1.2), per_channel=True),
        iaa.SaltAndPepper(p=0.0001)
    ])

    segmap = SegmentationMapsOnImage(lab, shape=im.shape)
    aug_im, aug_lab = seq(image=im, segmentation_maps=segmap)
    aug_lab = aug_lab.get_arr()

    visualize(aug_im[:, :, 5:2:-1], aug_lab, im[:, :, 5:2:-1], lab)
