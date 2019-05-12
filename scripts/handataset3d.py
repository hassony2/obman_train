import argparse
import matplotlib.pyplot as plt
import os
import random
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn.parallel
import torch.optim

from mano_train.datautils import ConcatDataloader
from mano_train.networks.handnet import HandNet
from mano_train.networks import netutils
from mano_train.netscripts.get_datasets import get_dataset
from mano_train.exputils.monitoring import Monitor
from mano_train.exputils import argutils
from mano_train.modelutils import modelio
from mano_train.netscripts import epochpass3d, simulate
from mano_train.options import datasetopts, nets3dopts, expopts

from handobjectdatasets.queries import BaseQueries, TransQueries
plt.switch_backend('agg')

def main(args):
    best_score = None

    # Initialize randoms seeds
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # create checkpoint dir
    os.makedirs(args.exp_id, exist_ok=True)

    # Initialize model
    model = HandNet(
        resnet_version=args.resnet_version,
        absolute_lambda=args.absolute_lambda,
        atlas_residual=args.atlas_residual,
        atlas_separate_encoder=args.atlas_separate_encoder,
        atlas_loss=args.atlas_loss,
        atlas_emd_regul=args.atlas_emd_regul,
        atlas_lambda=args.atlas_lambda,
        atlas_final_lambda=args.atlas_final_lambda,
        atlas_mesh=args.atlas_mesh,
        atlas_mode=args.atlas_mode,
        atlas_lambda_regul_edges=args.atlas_lambda_regul_edges,
        atlas_lambda_laplacian=args.atlas_lambda_laplacian,
        atlas_points_nb=args.atlas_points_nb,
        atlas_predict_trans=args.atlas_predict_trans,
        atlas_predict_scale=args.atlas_predict_scale,
        atlas_trans_weight=args.atlas_trans_weight,
        atlas_scale_weight=args.atlas_scale_weight,
        atlas_use_tanh=False,
        atlas_out_factor=200,
        contact_target=args.contact_target,
        contact_zones=args.contact_zones,
        contact_lambda=args.contact_lambda,
        contact_thresh=args.contact_thresh,
        contact_mode=args.contact_mode,
        collision_lambda=args.collision_lambda,
        collision_thresh=args.collision_thresh,
        collision_mode=args.collision_mode,
        fc_dropout=args.fc_dropout,
        inject_hands=args.inject_hands,
        mano_neurons=args.hidden_neurons,
        mano_center_idx=args.center_idx,
        mano_root='misc/mano',
        mano_comps=args.mano_comps,
        mano_use_pca=args.mano_use_pca,
        mano_use_pose_prior=args.mano_use_pose_prior,
        mano_use_shape=args.use_shape,
        mano_lambda_joints3d=args.mano_lambda_joints3d,
        mano_lambda_pose_reg=args.mano_lambda_pose_reg,
        mano_lambda_joints2d=args.mano_lambda_joints2d,
        mano_lambda_shape=args.mano_lambda_shape,
        mano_lambda_pca=args.mano_lambda_pca,
        mano_lambda_verts=args.mano_lambda_verts)
    max_queries = [
        TransQueries.affinetrans,
        TransQueries.images,
        TransQueries.verts3d,
        TransQueries.center3d,
        TransQueries.joints3d,
        TransQueries.objpoints3d,
        TransQueries.camintrs,
        BaseQueries.sides
    ]
    if args.mano_lambda_joints2d:
        max_queries.append(TransQueries.joints2d)

    # Optionally freeze parts of the network
    if args.freeze_batchnorm:
        netutils.freeze_batchnorm_stats(model)
    if args.freeze_encoder:
        netutils.rec_freeze(model.base_net)
        print('Froze encoder')
    if args.atlas_separate_encoder and args.atlas_freeze_encoder:
        netutils.rec_freeze(model.atlas_base_net)
        print('Froze atlas encoder')
    if args.atlas_freeze_decoder and hasattr(model, 'atlas_branch'):
        netutils.rec_freeze(model.atlas_branch.decoder)
        print('Froze atlas decoder')

    # Optimize unfrozen parts of the network
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    model_param_names = [
        name for name, val in model.named_parameters() if val.requires_grad
    ]
    if args.debug:
        print('=== Optimized params  === ')
        print(model_param_names)

    # Initialize optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rms':
        optimizer = torch.optim.RMSprop(
            model_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

    # Optionally resume from a checkpoint
    model = torch.nn.DataParallel(model)
    print('Using {} GPUs !'.format(torch.cuda.device_count()))
    if args.atlas_resume and args.resume:
        raise NotImplementedError('resume and atlas_resume incompatible for now')
    if args.atlas_resume:
        # Load atlas encoder and decoder to atlas-specific encoder-decoder branch
        start_epoch, _ = modelio.load_checkpoint(
            model, resume_path=args.atlas_resume, strict=False, load_atlas=True)
        print('Loaded ATLAS checkpoint from epoch {}, starting from there'.format(
            start_epoch))
        if args.evaluate:
            args.epochs = start_epoch + 1
    if args.resume is not None:
        # Load full model weights
        if len(args.resume) == 1:
            start_epoch, _ = modelio.load_checkpoint(
                model, resume_path=args.resume[0], optimizer=optimizer, strict=False)
            print('Loaded checkpoint from epoch {}, starting from there'.format(
                start_epoch))
        else:
            if not args.evaluate:
                raise ValueError('Multiple checkpoint resume only works in evaluate mode')
            start_epoch, _ = modelio.load_checkpoints(model, args.resume, strict=False)
        if args.evaluate:
            args.epochs = start_epoch + 1
    model.cuda()
    # Override loaded learning rate
    for group in optimizer.param_groups:
        group['lr'] = args.lr
        group['initial_lr'] = args.lr

    if args.lr_decay_gamma:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args.lr_decay_step, gamma=args.lr_decay_gamma)

    if args.debug:
        num_params = sum(p.numel() for p in model.parameters()) / 1000000.0
        print('Total params: {} Million'.format(num_params))

    if not args.evaluate:
        # Initialize train datasets
        train_loaders = []

        if args.controlled_exp:
            # Use subset of datasets so that final dataset size is constant
            limit_size = int(args.controlled_size / len(args.train_datasets))
        else:
            limit_size = None

        # Initialize train datasets
        for train_split, dat_name in zip(args.train_splits,
                                         args.train_datasets):
            train_dat = get_dataset(
                dat_name,
                meta={
                    'mode': args.mode,
                    'override_scale': args.override_scale,
                    'fhbhands_split_type': args.fhbhands_split_type,
                    'fhbhands_split_choice': args.fhbhands_split_choice,
                    'fhbhands_topology': args.fhbhands_topology,
                },
                split=train_split,
                sides=args.sides,
                train_it=True,
                max_queries=max_queries,
                mini_factor=args.mini_factor,
                point_nb=args.atlas_points_nb,
                center_idx=args.center_idx,
                limit_size=limit_size)
            print('Final dataset size: {}'.format(len(train_dat)))

            # Initialize train dataloader
            train_loader = torch.utils.data.DataLoader(
                train_dat,
                batch_size=args.train_batch,
                shuffle=True,
                num_workers=int(args.workers / len(args.train_splits)),
                pin_memory=True,
                drop_last=True)
            train_loaders.append(train_loader)
        train_loader = ConcatDataloader(train_loaders)

    # Initialize validation datasets
    val_loaders = []

    # Add black padding if trained only on ganerated
    for val_split, dat_name in zip(args.val_splits, args.val_datasets):
        val_dat = get_dataset(
            dat_name,
            max_queries=max_queries,
            meta={
                'mode': args.mode,
                'fhbhands_split_type': args.fhbhands_split_type,
                'fhbhands_split_choice': args.fhbhands_split_choice,
                'fhbhands_topology': args.fhbhands_topology,
                'override_scale': args.override_scale,
            },
            sides=args.sides,
            split=val_split,
            train_it=False,
            mini_factor=args.mini_factor,
            point_nb=args.atlas_points_nb,
            center_idx=args.center_idx)

        # Initialize val dataloader
        if args.evaluate:
            drop_last = True
        else:
            drop_last = True  # Keeps batch_size constant
        val_loader = torch.utils.data.DataLoader(
            val_dat,
            batch_size=args.test_batch,
            shuffle=False,
            num_workers=int(args.workers / len(args.val_datasets)),
            pin_memory=True,
            drop_last=drop_last)
        val_loaders.append(val_loader)
    val_loader = ConcatDataloader(val_loaders)

    # Get evaluation indexes
    val_idxs = None
    train_idxs = None

    hosting_folder = os.path.join(
        args.host_folder,
        args.exp_id)
    monitor = Monitor(args.exp_id, hosting_folder=hosting_folder)
    fig = plt.figure(figsize=(12, 12))

    for epoch in range(start_epoch, args.epochs):
        display = epoch % args.epoch_display_freq == 0
        # train for one epoch if not evaluating
        if not args.evaluate:
            print('Using lr {}'.format(optimizer.param_groups[0]['lr']))
            train_avg_meters, train_pck_infos = epochpass3d.epoch_pass(
                loader=train_loader,
                model=model,
                optimizer=optimizer,
                freeze_batchnorm=args.freeze_batchnorm,
                epoch=epoch,
                debug=args.debug,
                display_freq=args.train_display_freq,
                display=display,
                save_path=args.exp_id,
                idxs=train_idxs,
                train=True,
                fig=fig)

            # Save custom logs
            train_dict = {
                meter_name: meter.avg
                for meter_name, meter in
                train_avg_meters.average_meters.items()
            }
            if train_pck_infos:
                train_pck_dict = {
                    'auc': train_pck_infos['auc'],
                    'epe_mean': train_pck_infos['epe_mean'],
                    'epe_median': train_pck_infos['epe_median'],
                }
            else:
                train_pck_dict = {}
            train_full_dict = {**train_dict, **train_pck_dict}
            monitor.log_train(epoch + 1, train_full_dict)

        # Evaluate on validation set
        with torch.no_grad():
            val_avg_meters, val_pck_infos = epochpass3d.epoch_pass(
                loader=val_loader,
                model=model,
                epoch=epoch,
                optimizer=None,
                debug=args.debug,
                display_freq=args.test_display_freq,
                display=display,
                save_path=args.exp_id,
                idxs=val_idxs,
                train=False,
                fig=fig,
                save_results=args.save_results)
            val_dict = {
                meter_name: meter.avg
                for meter_name, meter in val_avg_meters.average_meters.items()
            }
            if val_pck_infos:
                val_pck_dict = {
                    'auc': val_pck_infos['auc'],
                    'epe_mean': val_pck_infos['epe_mean'],
                    'epe_median': val_pck_infos['epe_median'],
                }
            else:
                val_pck_dict = {}
            val_full_dict = {**val_dict, **val_pck_dict}
            monitor.log_val(epoch + 1, val_full_dict)

        # Interupt if evaluating
        if args.evaluate:
            if not args.no_simulate:
                # Get simulation results
                simulate.full_simul(exp_id=os.path.join(args.exp_id, 'save_results/val/epoch_{}'.format(epoch)), workers=args.workers, cluster=True, use_gui=False)
            return

        save_dict = {}
        for key in train_full_dict:
            save_dict[key] = {}
            if key in val_full_dict:
                save_dict[key]['val'] = val_full_dict[key]
            save_dict[key]['train'] = train_full_dict[key]

        monitor.metrics.save_metrics(epoch + 1, save_dict)
        monitor.metrics.plot_metrics()

        # remember best acc and save checkpoint
        if 'auc' in val_pck_infos:
            best_metric = 'auc'
            if best_score is None:
                best_score = val_full_dict[best_metric]
            is_best = val_full_dict[best_metric] > best_score
            best_score = max(val_full_dict[best_metric], best_score)
        else:
            best_metric = 'total_loss'
            if best_score is None:
                best_score = val_full_dict[best_metric]
            is_best = val_full_dict[best_metric] < best_score
            best_score = min(val_full_dict[best_metric], best_score)
        modelio.save_checkpoint(
            {
                'epoch': epoch + 1,
                'network': args.network,
                'state_dict': model.state_dict(),
                'best_score': best_score,
                'optimizer': optimizer.state_dict(),
            },
            is_best=is_best,
            checkpoint=args.exp_id,
            snapshot=args.snapshot)
        if args.lr_decay_gamma:
            scheduler.step()
        if epoch % args.regul_decay_step == 0:
            model.module.decay_regul(args.regul_decay_gamma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mano training')
    datasetopts.add_dataset_opts(parser)
    datasetopts.add_dataset3d_opts(parser)
    nets3dopts.add_nets3d_opts(parser)
    nets3dopts.add_train3d_opts(parser)
    expopts.add_exp_opts(parser)

    args = parser.parse_args()
    argutils.print_args(args)
    argutils.save_args(args, args.exp_id, 'opt')
    main(args)
    print('All done !')
