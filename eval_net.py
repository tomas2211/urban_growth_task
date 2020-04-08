import argparse
import os

import numpy as np
import torch

from utils.train_utils import fetch_test_datasets, evaluate, \
    generate_visualizations
from utils.model import SKModel

'''
Evaluate a trained network on test data (specified in dataset.py).
Prints out the segmentation scores, creates figures with confusion matricies
and segmentation examples.
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint.')
    parser.add_argument('--data_folder', default='data', help='Folder with the dataset.')
    parser.add_argument('--out_folder', type=str, required=True, help="Folder to save images.")
    parser.add_argument('--device', type=str, default='cpu', help='cpu|cuda')
    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    test_loaders, vis_loader = \
        fetch_test_datasets(args.data_folder)

    model = SKModel(0).to(args.device)
    model.load_state_dict(
        torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    )

    # Evaluate the model
    prec_rec_fscores, confm_figures = \
        evaluate(model, test_loaders, args.device)

    target_names = ['Non-urban', 'Urban', 'Cloud']

    # Print the statistics
    print('--- F1 scores ---')
    for test in sorted(prec_rec_fscores.keys()):
        print('Set - %s' % test)
        fscores = np.array(prec_rec_fscores[test])[2, :]
        print(('%.2f\t' * 3) % tuple(fscores))

    print()
    print('--- Precision, recall, F1 for each class ---')
    for test in sorted(prec_rec_fscores.keys()):
        print('Set - %s' % test)
        prf = np.array(prec_rec_fscores[test])[:3].T.reshape(-1)
        print(('%.2f\t' * 9) % tuple(prf))

    # Save the confusion matricies
    for test in confm_figures:
        fig_fn = os.path.join(args.out_folder, '%s_confmat.png' % test)
        confm_figures[test].savefig(fig_fn)

    # Generate visualizations
    vis_figs = generate_visualizations(model, vis_loader, args.device)
    for i, fig in enumerate(vis_figs):
        fig_fn = os.path.join(args.out_folder, 'vis_%02d.png' % i)
        fig.savefig(fig_fn)
