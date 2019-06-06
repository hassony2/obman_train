def add_exp_opts(parser):
    parser.add_argument(
        "--exp_id",
        default="checkpoints/debug",
        type=str,
        help="Path of current experience (default:debug)",
    )
    parser.add_argument(
        "--host_folder",
        default="/meleze/data0/public_html/yhasson/experiments/mano_train",
        type=str,
        help="Path to folder where to save plotly train/validation curves",
    )
    parser.add_argument(
        "--resume",
        type=str,
        nargs="+",
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--train_display_freq", type=int, default=500, help="show intermediate results"
    )
    parser.add_argument("--test_display_freq", type=int, default=100)
    parser.add_argument("--epoch_display_freq", type=int, default=2)
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--pyapt_id")
    parser.add_argument("--no_simulate", action="store_true")
    parser.add_argument(
        "--snapshot",
        default=5,
        type=int,
        metavar="N",
        help="How often to take a snapshot of the model (0 = never)",
    )
    parser.add_argument("--manual_seed", default=0, type=int)
