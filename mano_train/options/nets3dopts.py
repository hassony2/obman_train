def add_nets3d_opts(parser):
    # Model structure
    parser.add_argument(
        "--network",
        default="manonet",
        choices=["manonet"],
        help="Network architecture",
    )
    parser.add_argument(
        "--right_only",
        action="store_true",
        help="Flip all left hands to work with right hands obly",
    )

    parser.add_argument(
        "--absolute_lambda",
        default=0,
        type=float,
        help="Root for absolute position of root joint",
    )

    # AtlasNet options
    parser.add_argument(
        "--atlas_separate_encoder",
        action="store_true",
        help="Use two encoders, one for the hand branch"
        "one for the object branch",
    )
    parser.add_argument(
        "--atlas_lambda",
        default=0,
        type=float,
        help="Weight of supervision on normalized vertices",
    )
    parser.add_argument("--atlas_loss", default="chamfer", choices=["chamfer"])
    parser.add_argument(
        "--atlas_final_lambda",
        default=0.167,
        type=float,
        help="Weight of point supervision for final vertices"
        "(after translation and scaling is applied)",
    )
    parser.add_argument(
        "--atlas_mesh",
        action="store_true",
        help="Whether to get points on the mesh instead or randomling "
        "generating a point cloud. This allows to use regularizations "
        "that rely on an underlying triangulation. If true, points are "
        "sampled on an icosphere of granularity 3.",
    )
    parser.add_argument(
        "--atlas_mode",
        default="sphere",
        choices=["sphere"],
        help="Whether to sample points on a sphere or on a disk mesh",
    )
    parser.add_argument(
        "--atlas_points_nb",
        default=600,
        type=int,
        help="Number of points to sample for atlas, ignored if atlas_mesh is "
        "False",
    )

    # Options for mesh regularization
    parser.add_argument(
        "--atlas_lambda_regul_edges",
        type=float,
        help="Penalize edges having very different sizes",
    )
    parser.add_argument(
        "--atlas_lambda_laplacian",
        type=float,
        default=0,
        help="Penalize bendings",
    )

    # Options to predict and supervise scale and translation in separate
    # branches
    parser.add_argument(
        "--atlas_predict_trans",
        action="store_true",
        help="Predict translation in separate branch",
    )
    parser.add_argument(
        "--atlas_trans_weight",
        default=0.167,
        type=float,
        help="Weighting parameter for translation prediction loss",
    )
    parser.add_argument(
        "--atlas_predict_scale",
        action="store_true",
        help="Predict scale in separate branch",
    )
    parser.add_argument(
        "--atlas_scale_weight",
        default=0.167,
        type=float,
        help="Weighting parameter for scale prediction loss",
    )

    # Optiosn to decay the importance of the mesh regularization terms
    parser.add_argument("--regul_decay_gamma", type=float, default=1)
    parser.add_argument("--regul_decay_step", type=int, default=300)

    # Options for mano prevision and supervision
    parser.add_argument(
        "--hidden_neurons",
        nargs="+",
        default=[1024, 256],
        type=int,
        help="Number of neurons in hidden layer for mano decoder",
    )
    parser.add_argument(
        "--mano_use_shape",
        action="store_true",
        help="Predict MANO shape parameters",
    )
    parser.add_argument(
        "--mano_lambda_shape",
        default=0.167,
        type=float,
        help="Weight to regularize hand shapes",
    )
    parser.add_argument(
        "--mano_lambda_pose_reg",
        default=0.167,
        type=float,
        help="Weight to supervise hand pose in axis-angle space",
    )
    parser.add_argument(
        "--mano_lambda_joints3d",
        default=0.167,
        type=float,
        help="Weight to supervise joint distances in 3d",
    )
    parser.add_argument("--mano_lambda_joints2d", default=0, type=float)
    parser.add_argument("--mano_lambda_verts", default=0.167, type=float)
    parser.add_argument(
        "--mano_use_pca",
        action="store_true",
        help="Predict pca parameters directly instead of rotation angles",
    )
    parser.add_argument(
        "--mano_lambda_pca",
        default=0,
        type=float,
        help="Weight to supervise hand pose in PCA space",
    )
    parser.add_argument(
        "--mano_comps",
        choices=list(range(5, 46)),
        default=30,
        type=int,
        help="Number of PCA components",
    )

    # Image encoder options
    parser.add_argument(
        "--resnet_version", default=18, type=int, choices=[18, 50]
    )
    parser.add_argument(
        "--no_pretrain",
        action="store_true",
        help="Disable ImageNet pretraining for ResNet encoder",
    )

    # Freezing and initalization options
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze weights of encoder",
    )
    parser.add_argument(
        "--freeze_batchnorm",
        action="store_true",
        help="Freeze batchnorm layers",
    )
    parser.add_argument("--atlas_resume", help="Path to atlas checkpoint")
    parser.add_argument(
        "--atlas_decoder", help="Path to atlas decoder to load"
    )
    parser.add_argument(
        "--atlas_freeze_decoder",
        action="store_true",
        help="Freeze atlas decoder",
    )
    parser.add_argument(
        "--atlas_freeze_encoder",
        action="store_true",
        help="Freeze atlas encoder",
    )

    # Options for contact loss
    parser.add_argument(
        "--contact_target",
        default="all",
        choices=["all", "hand", "obj"],
        help="Loss only acts on hand/obj vertices, or on both",
    )
    parser.add_argument(
        "--contact_zones",
        default="zones",
        choices=["all", "tips", "zones"],
        help="Contact loss applied on all, tips or connected contact zones",
    )
    parser.add_argument("--contact_lambda", default=0, type=float)
    parser.add_argument("--contact_thresh", default=10, type=float)
    parser.add_argument(
        "--contact_mode",
        default="dist_tanh",
        choices=["dist_sq", "dist", "dist_tanh"],
        help="Function for supervision on colliding and attracted vertices",
    )
    parser.add_argument("--collision_lambda", default=0, type=float)
    parser.add_argument("--collision_thresh", default=20, type=float)
    parser.add_argument(
        "--collision_mode",
        default="dist_tanh",
        choices=["dist_sq", "dist", "dist_tanh"],
    )


def add_train3d_opts(parser):
    # Training strategy
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        help="number of data loading workers (default: 1)",
    )
    parser.add_argument(
        "--epochs", default=30, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "--train_batch", default=32, type=int, help="Train batch size"
    )
    parser.add_argument(
        "--test_batch",
        default=32,
        type=int,
        metavar="N",
        help="Test batch size",
    )

    # Optimization options
    parser.add_argument(
        "--optimizer", default="adam", choices=["rms", "adam", "sgd"]
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.0001,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument(
        "--lr_decay_step",
        default=300,
        type=int,
        help="Epochs after which to decay learning rate",
    )
    parser.add_argument(
        "--lr_decay_gamma",
        default=0.5,
        type=float,
        help="Factor by which to decay the learning rate",
    )
    parser.add_argument("--weight_decay", default=0, type=float)
