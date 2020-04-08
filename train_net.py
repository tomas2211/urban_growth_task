import argparse
import os

import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import tqdm

from utils.train_utils import fetch_train_dataset, fetch_test_datasets, \
    get_next_sample, evaluate, generate_visualizations
from utils.model import SKModel


def fetch_optimizer(model, lr, wd, lr_milestones, lr_gamma):
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, eps=1e-8
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=lr_gamma
    )
    return optimizer, scheduler


def train(args):
    summary_writer = tensorboardX.SummaryWriter(args.out_folder)
    class_weights = torch.tensor(args.class_weight).to(args.device)

    train_loader = fetch_train_dataset(args.data_folder, args.batch_size, args)
    test_loaders, vis_loader = fetch_test_datasets(args.data_folder)

    target_names = ['Non-urban', 'Urban', 'Cloud']

    model = SKModel(args.dropout).to(args.device)

    optimizer, scheduler = fetch_optimizer(
        model, args.lr, args.wd, args.lr_milestones, args.lr_gamma)
    loss_layer = torch.nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=-1, reduction='mean')

    train_iter = None
    iters = 0

    for epoch in tqdm.tqdm(range(args.epochs), desc='Epoch'):
        loss_sum = 0
        trange = tqdm.trange(
            args.iter_per_epoch, leave=True, desc='Training iteration')

        for iter in trange:
            model.train()
            train_iter, input_dict = get_next_sample(train_loader, train_iter)
            out = model(input_dict['im'].to(args.device))
            lab = input_dict['lab'].to(args.device)

            optimizer.zero_grad()

            loss = loss_layer(out, lab)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip)  # Gradient clipping

            optimizer.step()
            scheduler.step()

            loss_sum += loss.cpu().item()

            act_lr = scheduler.get_lr()[0]  # Record the LR
            summary_writer.add_scalar('lr', act_lr, iters)

            trange.set_postfix({'loss': loss_sum / (iter + 1), 'lr': act_lr})
            iters += 1

        if epoch % args.save_freq == 0:  # Save checkpoint
            save_path = os.path.join(
                args.out_folder, 'checkpoint_%06d.pth' % epoch)
            torch.save(model.state_dict(), save_path)

        summary_writer.add_scalar(
            'loss_avg', loss_sum / args.iter_per_epoch, epoch)

        # Evaluate the model
        prec_rec_fscores, confm_figures = \
            evaluate(model, test_loaders, args.device)

        for test in prec_rec_fscores:  # Add  Precision/Recall/F1 to TB
            # Reshape so that we iterate over classes
            #  (for each class [prec, rec, f1]).
            prf_perline = np.array(prec_rec_fscores[test])[:3].T

            for cl_name, cl_prf in zip(target_names, prf_perline):
                summary_writer.add_scalar(
                    '%s/%s_%s' % (test, cl_name, 'prec'), cl_prf[0], epoch)
                summary_writer.add_scalar(
                    '%s/%s_%s' % (test, cl_name, 'rec'), cl_prf[1], epoch)
                summary_writer.add_scalar(
                    '%s/%s_%s' % (test, cl_name, 'f1'), cl_prf[2], epoch)

            summary_writer.add_figure(
                '%s/confmat' % test, confm_figures[test], epoch)

        # Generate visualizations
        vis_figs = generate_visualizations(
            model, vis_loader, args.device)

        for i, fig in enumerate(vis_figs):
            summary_writer.add_figure('vis/%02d' % i, fig, epoch)

        summary_writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='data', help='Folder with the training dataset.')
    parser.add_argument('--dat_muliplier', type=int, default=1, help='Training sample list multiplier (to allow batch size > training dataset).')
    parser.add_argument('--out_folder', help="Folder to save checkpoints and TB")
    parser.add_argument('--save_freq', type=int, default=1, help='Checkpoint save frequency')
    parser.add_argument('--device', type=str, default='cpu', help='cpu|cuda')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[1000, 2000], help='Learning rate scheduler milestones.')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Learning rate scheduler gamma.')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[50, 50], help='Random crop size during training')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping threshold (norm).')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')
    parser.add_argument('--class_weight', type=float, nargs='+', default=[6.15, 1.59, 4.78], help='Weights of classes in the loss function in order: non-urban, urban, cloud')
    parser.add_argument('--color_augs', action='store_true', help='Use also color augmentations')

    parser.add_argument('--iter_per_epoch', type=int, default=1000, help='Number of iterations per epoch')
    parser.add_argument('--epochs', type=int, default=12, help='Number of epochs')

    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    torch.manual_seed(1)
    np.random.seed(2)

    train(args)
