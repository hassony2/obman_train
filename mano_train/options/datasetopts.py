def add_dataset_opts(parser):
    # Selected datasets
    parser.add_argument(
        "--train_datasets",
        choices=[
            "fhbhands",
            "fhbhands_obj",
            "fhbhands_hand",
            "dexterobj",
            "dimsynthands",
            "panoptic",
            "ganhands",
            "tomasreal",
            "tomasynth",
            "zimsynth",
            "yanasynth",
            "stereohands",
            "lsm3d",
            "zimsynth",
            "tzionas",
            "synthcube",
        ].extend(["synthands_{}".format(i) for i in range(100)]),
        nargs="+",
        default=["obman"],
        help="training dataset to load [tomasreal|tomasynth|yanasynth|lsm3d]",
    )
    parser.add_argument("--mini_factor", type=float, help="work on subset of datasets")
    parser.add_argument(
        "--sides",
        default="left",
        choices=["left", "right", "both"],
        help="Hand sides to train/test on",
    )
    # Synthgrasp_options
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "hand", "obj"],
        help="Mode for synthgrasp dataset",
    )
    parser.add_argument(
        "--fhbhands_split_type",
        default="subjects",
        choices=["objects", "subjects", "actions"],
        help="Mode for synthgrasp dataset",
    )
    parser.add_argument(
        "--fhbhands_split_choice",
        choices=["juice_bottle", "salt", "milk", "liquid_soap", None],
        help="Mode for synthgrasp dataset",
    )
    parser.add_argument(
        "--fhbhands_filter_object",
        choices=["juice_bottle", "salt", "milk", "liquid_soap", None],
        help="Only keep given object for synthgrasp dataset",
    )
    parser.add_argument(
        "--fhbhands_topology",
        choices=[None, "0", "1"],
        help="Filter fhb objects on topology",
    )
    parser.add_argument(
        "--synthgrasp_class_ids",
        nargs="+",
        help="Only use subset of classes, cellphone:02992529, bottle:02876657",
    )
    parser.add_argument(
        "--override_scale", action="store_true", help="Override object scale"
    )

    parser.add_argument(
        "--train_splits",
        type=str,
        nargs="+",
        default=["train"],
        help="One split per dataset",
    )
    parser.add_argument(
        "--val_datasets",
        choices=[
            "dexterobj",
            "tomasreal",
            "tomasynth",
            "yanasynth",
            "stereohands",
            "synthands",
            "lsm3d",
            "zimsynth",
        ].extend(["synthands_{}".format(i) for i in range(100)]),
        nargs="+",
        default=["obman"],
        help="validation dataset to load [tomasreal|tomasynth]",
    )
    parser.add_argument(
        "--val_splits",
        default=["train"],
        type=str,
        nargs="+",
        help="split to load [train|test]",
    )
    parser.add_argument(
        "--controlled_exp",
        action="store_true",
        help="Fixed size of dataset hard-coded to controlled_size",
    )
    parser.add_argument("--controlled_size", type=int, default=60000)
    parser.add_argument("--synthgrasps_segment", action="store_true")


def add_dataset3d_opts(parser):
    # Data preprocessing
    parser.add_argument("--center_idx", type=int, default=0)
