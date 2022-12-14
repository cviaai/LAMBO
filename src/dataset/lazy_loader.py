from typing import Optional, Type, Callable, Dict

import albumentations
import torch
from torch import nn, Tensor
from torch.utils import data
from torch.utils.data import DataLoader, Subset

# from dataset.brats import BraTS3D, Dataset3DTo2D, FilterDataset, BraTS2D, MasksDataset
from dataset.cardio_dataset import ImageMeasureDataset, ImageDataset, CelebaWithLandmarks
from dataset.cardio_keypts import CardioDataset, LandmarksDataset, LandmarksDatasetAugment
from dataset.d300w import ThreeHundredW
from dataset.MAFL import MAFLDataset
from albumentations.pytorch.transforms import ToTensorV2 as AlbToTensor, ToTensorV2

from dataset.hum36 import SimpleHuman36mDataset
from parameters.path import Paths


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch



class AbstractLoader:
    pass


# class BraTSLoader(AbstractLoader):
#
#     batch_size = 8
#     test_batch_size = 8
#     image_size = 256
#
#     transforms = albumentations.Compose([
#         albumentations.Resize(image_size, image_size),
#         albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#         ToTensorV2()
#     ])
#
#     def __init__(self):
#
#         path = "/raid/data/BraTS_dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
#         self.dataset = FilterDataset(
#             BraTS2D(path, BraTSLoader.transforms),
#             lambda data: data[1].sum() > 100, load=True
#         )
#
#         N = self.dataset.__len__()
#
#         self.dataset_train = Subset(self.dataset, range(int(N * 0.8)))
#
#         self.loader_train = data.DataLoader(
#             self.dataset_train,
#             batch_size=BraTSLoader.batch_size,
#             sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
#             drop_last=True,
#             num_workers=10
#         )
#
#         self.loader_train_inf = sample_data(self.loader_train)
#
#         self.dataset_test = Subset(self.dataset, range(int(N * 0.8), N))
#
#         self.test_loader = data.DataLoader(
#             self.dataset_test,
#             batch_size=BraTSLoader.test_batch_size,
#             drop_last=False,
#             num_workers=10
#         )
#
#         N = 3529
#         self.masks_bc = MasksDataset(path=f"{Paths.default.data()}/brats_{N}", transform=ToTensorV2())
#
#         self.loader_masks_bc = data.DataLoader(
#             self.masks_bc,
#             batch_size=BraTSLoader.batch_size,
#             sampler=data_sampler(self.masks_bc, shuffle=True, distributed=False),
#             drop_last=True,
#             num_workers=10
#         )
#
#         self.loader_masks_bc_inf = sample_data(self.loader_masks_bc)
#
#         print("BraTS initialize")
#         print(f"train size: {len(self.dataset_train)}, test size: {len(self.dataset_test)}, bc size: {len(self.masks_bc)}")
#


class W300DatasetLoader:

    batch_size = 8
    test_batch_size = 16

    def __init__(self):
        self.dataset_train = ThreeHundredW(f"{Paths.default.data()}/300w", train=True, imwidth=500, crop=15)

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=W300DatasetLoader.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        self.test_dataset = ThreeHundredW(f"{Paths.default.data()}/300w", train=False, imwidth=500, crop=15)

        self.test_loader = data.DataLoader(
            self.test_dataset,
            batch_size=W300DatasetLoader.test_batch_size,
            drop_last=False,
            num_workers=20
        )

        print("300 W initialize")
        print(f"train size: {len(self.dataset_train)}, test size: {len(self.test_dataset)}")

        self.test_loader_inf = sample_data(self.test_loader)


class CelebaWithKeyPoints:

    image_size = 256
    batch_size = 8

    @staticmethod
    def transform():
        return albumentations.Compose([
            albumentations.HorizontalFlip(),
            albumentations.Resize(CelebaWithKeyPoints.image_size, CelebaWithKeyPoints.image_size),
            # albumentations.ElasticTransform(p=0.5, alpha=100, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.5, rotate_limit=10),
            albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            AlbToTensor()
        ])

    def __init__(self):

        print("init calaba with masks")

        dataset = ImageMeasureDataset(
            f"{Paths.default.data()}/celeba",
            f"{Paths.default.data()}/celeba_masks",
            img_transform=CelebaWithKeyPoints.transform()
        )

        print("dataset size: ", len(dataset))

        self.loader = data.DataLoader(
            dataset,
            batch_size=CelebaWithKeyPoints.batch_size,
            sampler=data_sampler(dataset, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        print("batch size: ", CelebaWithKeyPoints.batch_size)

        self.loader = sample_data(self.loader)


class Celeba:

    image_size = 256
    batch_size = 8

    transform = albumentations.Compose([
            albumentations.HorizontalFlip(),
            albumentations.Resize(CelebaWithKeyPoints.image_size, CelebaWithKeyPoints.image_size),
            # albumentations.ElasticTransform(p=0.5, alpha=50, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.5, rotate_limit=10, scale_limit=(-0.1, 0.3)),
            albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            AlbToTensor()
    ])

    def __init__(self):
        print("init calaba")

        dataset = ImageDataset(
            f"{Paths.default.data()}/celeba",
            img_transform=Celeba.transform
        )

        print("dataset size: ", len(dataset))

        self.loader = data.DataLoader(
            dataset,
            batch_size=Celeba.batch_size,
            sampler=data_sampler(dataset, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=Celeba.batch_size
        )

        print("batch size: ", Celeba.batch_size)

        self.loader = sample_data(self.loader)


class Cardio:

    image_size = 256
    batch_size = 4
    test_batch_size = 4

    transforms = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

    def __init__(self):
        path = "/raid/data/ibespalov/CHAZOV_dataset/folds4chamb.csv"
        self.dataset_train = CardioDataset(path, train=True, transform=Cardio.transforms)

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=Cardio.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        self.test_dataset = CardioDataset(path, train=False, transform=Cardio.transforms)

        print("train:", self.dataset_train.__len__())
        print("test:", self.test_dataset.__len__())

        self.test_loader = data.DataLoader(
            self.test_dataset,
            batch_size=Cardio.test_batch_size,
            drop_last=False,
            num_workers=20
        )


class CardioLandmarks:

    batch_size = 8

    transforms = albumentations.Compose([
        ToTensorV2()
    ])

    def __init__(self):
        path = "/raid/data/cardio"
        self.dataset_train = LandmarksDataset(path, transform=CardioLandmarks.transforms)

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=CardioLandmarks.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        print("CardioLandmarks")
        print("train:", self.dataset_train.__len__())


class W300Landmarks:

    batch_size = 4

    transforms = albumentations.Compose([
        ToTensorV2()
    ])

    def __init__(self, sub_path: str):
        path = f"{Paths.default.data()}/{sub_path}"
        self.dataset_train = LandmarksDataset(path, transform=W300Landmarks.transforms)

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=W300Landmarks.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        print("W300Landmarks")
        print("train:", self.dataset_train.__len__())


class HumanLandmarks:

    batch_size = 8

    transforms = albumentations.Compose([
        ToTensorV2()
    ])

    def __init__(self, sub_path: str):
        path = f"{Paths.default.data()}/{sub_path}"
        self.dataset_train = LandmarksDataset(path, transform=HumanLandmarks.transforms)

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=HumanLandmarks.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        print("HumanLandmarks")
        print("train:", self.dataset_train.__len__())


class W300LandmarksAugment:

    batch_size = 4

    transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0, rotate_limit=15, p=0.3),
        ToTensorV2()
    ])

    def __init__(self, sub_path: str):
        path = f"{Paths.default.data()}/{sub_path}"
        self.dataset_train = LandmarksDatasetAugment(path, transform=W300LandmarksAugment.transforms)

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=W300LandmarksAugment.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        print("W300LandmarksAugment")
        print("train:", self.dataset_train.__len__())


class MAFL:

    batch_size = 8
    test_batch_size = 32

    def __init__(self):
        dataset_train = MAFLDataset(f"{Paths.default.data()}", split="train", target_type="landmarks")

        self.loader_train = data.DataLoader(
            dataset_train,
            batch_size=MAFL.batch_size,
            sampler=data_sampler(dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        self.test_dataset = MAFLDataset(f"{Paths.default.data()}", split="test", target_type="landmarks")

        self.test_loader = data.DataLoader(
            self.test_dataset,
            batch_size=MAFL.test_batch_size,
            drop_last=False,
            num_workers=20
        )

        print("MAFL initialize")
        print(f"train size: {len(dataset_train)}, test size: {len(self.test_dataset)}")

        self.test_loader_inf = sample_data(self.test_loader)


class HumanLoader:

    batch_size = 8
    test_batch_size = 16

    def __init__(self, use_mask=False):

        self.dataset_train = SimpleHuman36mDataset()
        self.dataset_train.initialize(f"{Paths.default.data()}/human_images", use_mask=use_mask)

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=HumanLoader.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        self.test_dataset = SimpleHuman36mDataset()
        self.test_dataset.initialize(f"{Paths.default.data()}/human_images", subset="test", use_mask=use_mask)

        print("train:", self.dataset_train.__len__())
        print("test:", self.test_dataset.__len__())

        self.test_loader = data.DataLoader(
            self.test_dataset,
            batch_size= HumanLoader.test_batch_size,
            drop_last=False,
            num_workers=20
        )


class LazyLoader:

    saved = {}

    w300_save: Optional[W300DatasetLoader] = None
    celeba_kp_save: Optional[CelebaWithKeyPoints] = None
    celeba_save: Optional[Celeba] = None
    cardio_save: Optional[Cardio] = None
    cardio_lm_save: Optional[CardioLandmarks] = None
    w300_lm_save: Optional[W300Landmarks] = None
    w300_lm_augment_save: Optional[W300LandmarksAugment] = None
    celebaWithLandmarks: Optional[CelebaWithLandmarks] = None
    mafl_save: Optional[MAFL] = None
    human_save: Optional[HumanLoader] = None
    human_lm_save: Optional[HumanLandmarks] = None

    @staticmethod
    def register_loader(cls: Type[AbstractLoader]):
        LazyLoader.saved[cls.__name__] = None

    @staticmethod
    def w300() -> W300DatasetLoader:
        if not LazyLoader.w300_save:
            LazyLoader.w300_save = W300DatasetLoader()
        return LazyLoader.w300_save

    @staticmethod
    def mafl() -> MAFL:
        if not LazyLoader.mafl_save:
            LazyLoader.mafl_save = MAFL()
        return LazyLoader.mafl_save

    @staticmethod
    def celeba_with_kps():
        if not LazyLoader.celeba_kp_save:
            LazyLoader.celeba_kp_save = CelebaWithKeyPoints()
        return LazyLoader.celeba_kp_save

    @staticmethod
    def celeba():
        if not LazyLoader.celeba_save:
            LazyLoader.celeba_save = Celeba()
        return LazyLoader.celeba_save

    @staticmethod
    def cardio():
        if not LazyLoader.cardio_save:
            LazyLoader.cardio_save = Cardio()
        return LazyLoader.cardio_save

    @staticmethod
    def cardio_landmarks():
        if not LazyLoader.cardio_lm_save:
            LazyLoader.cardio_lm_save = CardioLandmarks()
        return LazyLoader.cardio_lm_save

    @staticmethod
    def w300_landmarks(path: str):
        if not LazyLoader.w300_lm_save:
            LazyLoader.w300_lm_save = W300Landmarks(path)
        return LazyLoader.w300_lm_save

    @staticmethod
    def human_landmarks(path: str):
        if not LazyLoader.human_lm_save:
            LazyLoader.human_lm_save = HumanLandmarks(path)
        return LazyLoader.human_lm_save

    @staticmethod
    def w300augment_landmarks(path: str):
        if not LazyLoader.w300_lm_augment_save:
            LazyLoader.w300_lm_augment_save = W300LandmarksAugment(path)
        return LazyLoader.w300_lm_augment_save

    @staticmethod
    def celeba_test(batch_size=1):
        if not LazyLoader.celebaWithLandmarks:
            LazyLoader.celebaWithLandmarks = sample_data(DataLoader(
                CelebaWithLandmarks(),
                batch_size=batch_size,
                drop_last=False))
        return LazyLoader.celebaWithLandmarks


    @staticmethod
    def human36(use_mask=False):
        if not LazyLoader.human_save:
            LazyLoader.human_save = HumanLoader(use_mask=use_mask)
        return LazyLoader.human_save
