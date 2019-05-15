#!/usr/bin/env python
import argparse
from collections import OrderedDict
import os
import pickle
import sys

import dominate
from dominate.tags import *

from mano_train.exputils import argutils
from mano_train import logutils as manologutils


def analyze_experiment(checkpoint_path,
                       metrics={'epe_mean': False,
                                'auc': True},
                       epoch=None,
                       host_folder=None,
                       no_train=False,
                       show_best_idx=True,
                       show_compare_epoch=True):

    pickle_path = os.path.join(checkpoint_path, 'opt.pkl')
    with open(pickle_path, 'rb') as pickle_f:
        opt = pickle.load(pickle_f)
    exp = OrderedDict()
    if checkpoint_path.endswith('/'):
        if opt['evaluate']:
            checkpoint_path = os.path.dirname(os.path.dirname(checkpoint_path))
        experiment_name = os.path.basename(os.path.dirname(checkpoint_path))
    else:
        experiment_name = os.path.basename(checkpoint_path)
    if host_folder is not None:
        if opt['evaluate']:
            # If evaluation results, still link training curves
            plotly_path = os.path.join(host_folder,
                                       os.path.dirname(opt['resume'][0]),
                                       'plotly.html')
        else:
            plotly_path = os.path.join(host_folder, opt['exp_id'],
                                       'plotly.html')
            exp['url'] = plotly_path
    exp['exp_id'] = experiment_name
    exp['train_dataset'] = '_'.join(sorted(opt['train_datasets']))
    exp['train_splits'] = ' '.join(opt['train_splits'])
    if 'val_dataset' in opt:
        exp['val_dataset'] = opt['val_dataset']
        exp['val_split'] = opt['val_split']
    elif 'val_datasets' in opt:
        exp['val_dataset'] = '_'.join(opt['val_datasets'])
        exp['val_split'] = '_'.join(opt['val_splits'])

    exp['network'] = opt['network']
    if not no_train:
        train_infos = manologutils.get_split_info(
            checkpoint,
            split='train',
            metrics=metrics,
            epoch=epoch,
            show_best_idx=show_best_idx,
            show_compare_epoch=show_compare_epoch)
        train_exp = manologutils.append_info(exp, train_infos)
    else:
        train_exp = exp

    val_infos = manologutils.get_split_info(
        checkpoint,
        split='val',
        metrics=metrics,
        epoch=epoch,
        show_best_idx=show_best_idx,
        show_compare_epoch=show_compare_epoch)
    val_exp = manologutils.append_info(exp, val_infos)

    return train_exp, val_exp


def make_table(exp_list):
    table = []
    headers = list(exp_list[0].keys())
    table.append(headers)
    for exp_info in exp_list:
        exp_vals = [exp_info[header] for header in headers]
        table.append(exp_vals)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoints',
        nargs='+',
        type=str,
        help='path to checkpoints folders')
    parser.add_argument(
        '--vis', action='store_true', help='Whether to plot the log curves')
    parser.add_argument(
        '--no_train', action='store_true', help='Do not show train metrics')
    parser.add_argument(
        '--save_folder',
        default=
        '/meleze/data0/public_html/yhasson/experiments/mano_train/tables')
    parser.add_argument(
        '--host_path',
        default=
        'https://www.rocq.inria.fr/cluster-willow/yhasson/experiments/mano_train'
    )
    parser.add_argument(
        '--compare_metric',
        default='epe_mean',
        help='Metric on which'
        'to choose best epoch')
    parser.add_argument(
        '--epoch', type=int, help='Epoch at which to show results')
    parser.add_argument(
        '--metric_higher_better',
        action='store_true',
        help='Activate if higher compare_metric is better')
    parser.add_argument(
        '--vis_metrics', nargs='+', help='Additional metrics to show')
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--print_doc', action='store_true')
    parser.add_argument(
        '--show_best_idx',
        action='store_true',
        help='Show idx of best iteration')
    args = parser.parse_args()
    argutils.print_args(args)

    all_train_infos = []
    all_val_infos = []
    if args.epoch is None:
        metrics = OrderedDict({args.compare_metric: args.metric_higher_better})
    else:
        metrics = OrderedDict({})
    if args.vis_metrics is not None:
        for metric in args.vis_metrics:
            metrics[metric] = False
    for checkpoint in args.checkpoints:
        train_info, val_info = analyze_experiment(
            checkpoint,
            metrics=metrics,
            epoch=args.epoch,
            host_folder=args.host_path,
            no_train=args.no_train,
            show_best_idx=args.show_best_idx)
        all_train_infos.append(train_info)
        all_val_infos.append(val_info)
    if not args.no_train:
        train_table = manologutils.make_table(all_train_infos)
    # for val_idx, val_info in enumerate(all_val_infos):
    #     epe_mean = val_info['epe_mean best_val']
    #     plt.scatter(0, epe_mean, label='{}'.format(val_idx))
    # plt.legend()
    # plt.show()
    val_table = manologutils.make_table(all_val_infos)

    # Create html
    doc = dominate.document('my doc')
    with doc.head:
        link(
            rel='stylesheet',
            href=
            'https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/2.10.0/github-markdown.css'
        )
        link(
            rel='stylesheet',
            href=
            'https://www.rocq.inria.fr/cluster-willow/yhasson/markdown-reports/css/perso-git-markdown.css'
        )

        with doc:
            with article(cls='markdown-body'):
                comment(sys.argv[0] + ' ' + ' '.join(sys.argv[1:]))
            if not args.no_train:
                h2().add('Training table')
                manologutils.add_table(train_table)
            h2().add('Validation table')
            manologutils.add_table(val_table)
    if args.print_doc:
        print(doc)
    if args.save_name is not None:
        if '/' in args.save_name:
            os.makedirs(
                os.path.join(args.save_folder,
                             os.path.dirname(args.save_name)),
                exist_ok=True)
            save_path = os.path.join(args.save_folder, args.save_name)
        with open(save_path, 'w') as f:
            f.write(doc.render())
        print('Write html to {}'.format(save_path))
