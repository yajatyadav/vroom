import os

from anycam.datasets.common import get_flow_selector, get_index_selector, get_sequence_sampler


def make_datasets(cfg, **kwargs):
    dataset_type = cfg["type"]

    data_path_training = cfg["data_path_training"]
    data_path_testing = cfg["data_path_testing"]

    data_path_training = os.path.realpath(data_path_training)
    data_path_testing = os.path.realpath(data_path_testing)

    split = cfg["split"]

    if split is not None:
        training_split = os.path.join(split, "train_files.txt")
        testing_split = os.path.join(split, "test_files.txt")
    else:
        training_split = None
        testing_split = None

    preprocessed_path_training = cfg.get("preprocessed_path_training", None)
    preprocessed_path_testing = cfg.get("preprocessed_path_testing", None)

    if preprocessed_path_training is not None:
        preprocessed_path_training = os.path.realpath(preprocessed_path_training)
    if preprocessed_path_testing is not None:
        preprocessed_path_testing = os.path.realpath(preprocessed_path_testing)

    dilation = cfg.get("dilation", 1)


    image_size = cfg.get("image_size", None)
    frame_count = cfg.get("frame_count", 2)

    return_depth = cfg.get("return_depth", False)
    return_flow = cfg.get("return_flow", False)

    full_size_depth = cfg.get("full_size_depth", False)

    sequential = False

    # Use kwargs to overwrite values for global dataset settings

    if 'image_size' in kwargs:
        image_size = kwargs['image_size']
    if 'frame_count' in kwargs:
        frame_count = kwargs['frame_count']
    if 'sequential' in kwargs:
        sequential = kwargs['sequential']
    if 'return_depth' in kwargs:
        return_depth = kwargs['return_depth']
    if 'return_flow' in kwargs:
        return_flow = kwargs['return_flow']

    flow_selector = get_flow_selector(frame_count, is_sequential=sequential)
    index_selector = get_index_selector(is_sequential=sequential)
    sequence_sampler = get_sequence_sampler(crop=True)

    if dataset_type == 'sintel':
        from anycam.datasets.sintel.sintel_dataset import SintelDataset

        train_dataset = SintelDataset(
            data_path_training,
            training_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=return_depth,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_training,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
        test_dataset = SintelDataset(
            data_path_testing,
            testing_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=True,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_testing,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )

    elif dataset_type == 'sintel-gt':
        from anycam.datasets.sintel.sintel_dataset import SintelGTDataset

        train_dataset = SintelGTDataset(
            data_path_training,
            training_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=True,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_training,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
        test_dataset = SintelGTDataset(
            data_path_testing,
            testing_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=True,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_testing,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )

    elif dataset_type == 'waymo':
        from anycam.datasets.waymo.waymo_dataset import WaymoDataset

        train_dataset = WaymoDataset(
            data_path_training,
            training_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=return_depth,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_training,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
        test_dataset = WaymoDataset(
            data_path_testing,
            testing_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=True,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_testing,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )

    elif dataset_type == 're10k':
        from anycam.datasets.realestate10k.re10k_dataset import RealEstate10kDataset

        train_dataset = RealEstate10kDataset(
            data_path_training,
            training_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=return_depth,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_training,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
        test_dataset = RealEstate10kDataset(
            data_path_testing,
            testing_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=True,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_testing,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )    
    elif dataset_type == 'youtubevos':
        from anycam.datasets.youtubevos.youtubevos_dataset import YouTubeVOSDataset

        train_dataset = YouTubeVOSDataset(
            data_path_training,
            training_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=return_depth,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_training,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
        test_dataset = YouTubeVOSDataset(
            data_path_testing,
            testing_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=True,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_testing,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
    elif dataset_type == 'davis':
        from anycam.datasets.davis.davis_dataset import DavisDataset

        train_dataset = DavisDataset(
            data_path_training,
            training_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=return_depth,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_training,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
        test_dataset = DavisDataset(
            data_path_testing,
            testing_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=True,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_testing,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
    elif dataset_type == 'tumrgbd':
        from anycam.datasets.tum_rgbd.tumrgbd_dataset import TUMRGBDDataset

        train_dataset = TUMRGBDDataset(
            data_path_training,
            training_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=return_depth,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_training,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
        test_dataset = TUMRGBDDataset(
            data_path_testing,
            testing_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=return_depth,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_testing,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
    elif dataset_type == 'aria-everyday-activities':
        from anycam.datasets.aria_everyday_activities.aea_dataset import AriaEADataset

        train_dataset = AriaEADataset(
            data_path=data_path_training,
            split_path=training_split,
            image_size=image_size,
            frame_count=frame_count,
            return_depth=return_depth,
            return_flow=return_flow,
            dilation=dilation,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
        test_dataset = AriaEADataset(
            data_path=data_path_testing,
            split_path=testing_split,
            image_size=image_size,
            frame_count=frame_count,
            return_depth=False,
            return_flow=return_flow,
            dilation=dilation,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
    elif dataset_type == 'aria-everyday-activities-extracted':
        from anycam.datasets.aria_everyday_activities.aea_dataset import ExtractedAEADataset

        train_dataset = ExtractedAEADataset(
            data_path=data_path_training,
            split_path=training_split,
            image_size=image_size,
            frame_count=frame_count,
            return_depth=return_depth,
            return_flow=return_flow,
            dilation=dilation,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
        test_dataset = ExtractedAEADataset(
            data_path=data_path_testing,
            split_path=testing_split,
            image_size=image_size,
            frame_count=frame_count,
            return_depth=False,
            return_flow=return_flow,
            dilation=dilation,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
    elif dataset_type == 'opendv':
        from anycam.datasets.opendv.opendv_dataset import OpenDVDataset

        train_dataset = OpenDVDataset(
            data_path_training,
            training_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=return_depth,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_training,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
        test_dataset = OpenDVDataset(
            data_path_testing,
            testing_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=True,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_testing,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
    elif dataset_type == 'walkingtours':
        from anycam.datasets.walkingtours.walkingtours_dataset import WalkingToursDataset

        train_dataset = WalkingToursDataset(
            data_path_training,
            training_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=return_depth,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_training,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
        test_dataset = WalkingToursDataset(
            data_path_testing,
            testing_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=True,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_testing,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
    elif dataset_type == 'epickitchens':
        from anycam.datasets.epickitchens.epickitchens_dataset import EpicKitchensDataset

        train_dataset = EpicKitchensDataset(
            data_path_training,
            training_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=return_depth,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_training,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
        test_dataset = EpicKitchensDataset(
            data_path_testing,
            testing_split,
            image_size,
            frame_count,
            dilation=dilation,
            return_depth=True,
            return_flow=return_flow,
            full_size_depth=full_size_depth,
            preprocessed_path=preprocessed_path_testing,
            flow_selector=flow_selector,
            index_selector=index_selector,
            sequence_sampler=sequence_sampler,
        )
    else:
        raise ValueError(f'Unknown dataset: {cfg.dataset.name}')
    
    return train_dataset, test_dataset