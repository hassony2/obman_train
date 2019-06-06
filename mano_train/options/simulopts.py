def add_simul_opts(parser):
    parser.add_argument(
        "--wait_time", default=0, type=float, help="Wait time for simulation"
    )
    parser.add_argument("--use_gui", action="store_true")
    parser.add_argument(
        "--batch_step", default=1, type=int, help="Step between batches"
    )
    parser.add_argument(
        "--sample_step", default=1, type=int, help="Step between samples in batch"
    )
    parser.add_argument(
        "--workers", default=8, type=int, help="Step between samples in batch"
    )
    parser.add_argument(
        "--sample_vis_freq", default=100, type=int, help="Step between samples in batch"
    )
    parser.add_argument("--cluster", action="store_true")
