def add_nets3d_opts(parser):
    # Model structure
    parser.add_argument(
        '--network',
        default='manonet',
        choices=['manonet'],
        help='Network architecture')
    parser.add_argument('--right_only', action='store_true')
    parser.add_argument(
        '--hidden_neurons',
        nargs='+',
        default=[256],
        type=int,
        help='number of neurons in hidden layer')
    # AE options
    parser.add_argument(
        '--ae_bottleneck', default=512, type=int, help='use shape')

    parser.add_argument('--use_shape', action='store_true', help='Predict MANO shape parameters')
    parser.add_argument('--resnet_version', default=18, type=int, choices=[18, 50])
    parser.add_argument('--absolute_lambda', default=0, type=float)
    parser.add_argument('--atlas_residual', action='store_true')
    parser.add_argument('--atlas_resume', help='Path to atlas checkpoint')
    parser.add_argument('--atlas_separate_encoder', action='store_true')
    parser.add_argument('--atlas_lambda', default=0, type=float)
    parser.add_argument('--atlas_loss', default='chamfer', choices=['chamfer'])
    parser.add_argument('--atlas_final_lambda', default=0.167, type=float)
    parser.add_argument(
        '--atlas_mesh', action='store_true', help='Deform spheres')
    parser.add_argument(
        '--atlas_mode',
        default='sphere',
        choices=['sphere', 'disk'],
        help='Sphere in disk mesh')
    parser.add_argument(
        '--atlas_lambda_regul_edges',
        type=float,
        help='Penalize edges having very different sizes')
    parser.add_argument(
        '--atlas_lambda_laplacian',
        type=float,
        default=0,
        help='Penalize bendings')
    parser.add_argument('--atlas_predict_trans', action='store_true')
    parser.add_argument('--atlas_trans_weight', default=0.167, type=float)
    parser.add_argument('--atlas_predict_scale', action='store_true')
    parser.add_argument('--atlas_scale_weight', default=0.167, type=float)
    parser.add_argument('--atlas_points_nb', default=600, type=int)
    parser.add_argument('--regul_decay_gamma', type=float, default=1)
    parser.add_argument('--regul_decay_step', type=int, default=300)
    parser.add_argument('--mano_use_pca', action='store_true')
    parser.add_argument('--mano_use_pose_prior', action='store_true')
    parser.add_argument('--mano_lambda_pose_reg', default=0.167, type=float)
    parser.add_argument('--fc_dropout', default=0, type=float)
    parser.add_argument('--mano_lambda_prior', action='store_true')
    parser.add_argument('--mano_lambda_shape', default=0.167, type=float, help='Weight to regularize hand shapes')
    parser.add_argument('--mano_lambda_joints3d', default=0.167, type=float)
    parser.add_argument('--mano_lambda_joints2d', default=0, type=float)
    parser.add_argument('--mano_lambda_verts', default=0.167, type=float)
    parser.add_argument('--mano_lambda_pca', default=0, type=float)
    parser.add_argument(
        '--contact_target',
        default='all',
        choices=['all', 'hand', 'obj'],
        help='Loss only acts on hand/obj vertices, or on both')
    parser.add_argument(
        '--contact_zones',
        default='zones',
        choices=['all', 'tips', 'zones'],
        help='Contact loss applied on all, tips or connected contact zones')
    parser.add_argument('--contact_lambda', default=0, type=float)
    parser.add_argument('--contact_thresh', default=10, type=float)
    parser.add_argument(
        '--contact_mode',
        default='dist_tanh',
        choices=['dist_sq', 'dist', 'dist_tanh'])
    parser.add_argument('--collision_lambda', default=0, type=float)
    parser.add_argument('--collision_thresh', default=20, type=float)
    parser.add_argument(
        '--collision_mode',
        default='dist_tanh',
        choices=['dist_sq', 'dist', 'dist_tanh'])
    parser.add_argument('--render_lambda', default=0, type=float)
    parser.add_argument(
        '--mano_comps',
        choices=list(range(5, 46)),
        default=30,
        type=int,
        help='Number of PCA components')


def add_train3d_opts(parser):
    # Training strategy
    parser.add_argument(
        '-j',
        '--workers',
        default=8,
        type=int,
        help='number of data loading workers (default: 1)')
    parser.add_argument(
        '--epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--no_pretrain', action='store_true')
    parser.add_argument(
        '--train_batch', default=6, type=int, help='train batchsize')
    parser.add_argument(
        '--test_batch',
        default=6,
        type=int,
        metavar='N',
        help='test batchsize')
    parser.add_argument(
        '--optimizer', default='adam', choices=['rms', 'adam', 'sgd'])
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--freeze_batchnorm', action='store_true')
    parser.add_argument('--atlas_decoder', help='Path to atlas decoder to load')
    parser.add_argument('--atlas_freeze_decoder', action='store_true')
    parser.add_argument('--atlas_freeze_encoder', action='store_true')
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=0.0001,
        type=float,
        metavar='LR',
        help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=300, type=int)
    parser.add_argument('--lr_decay_gamma', default=0.5, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
