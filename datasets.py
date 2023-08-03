import csv
import hashlib
import numpy as np
import os
import PIL.Image
import torch
import torchvision.datasets as tvds
import wget
from collections import namedtuple
from functools import partial
from torchvision import transforms


ROOT = os.path.expanduser("~/datasets")
CSV = namedtuple("CSV", ["header", "index", "data"])
DATASET_DICT = dict()


def register_dataset(dataset):
    try:
        name = dataset.name
    except AttributeError:
        name = dataset.__name__
    DATASET_DICT[name] = dataset
    return dataset


def crop_celeba(img):
    # the cropping parameters match the ones used in DDIM
    return transforms.functional.crop(img, top=57, left=25, height=128, width=128)  # noqa


def clip(x, clip_range, boundary="trunc"):
    if clip_range is None:
        return x
    else:
        if boundary == "trunc":
            return np.clip(x, *clip_range)
        elif boundary == "reflect":
            return 2 * np.clip(x, *clip_range) - x
        else:
            raise NotImplementedError(boundary)


def rand_like(x):
    if isinstance(x, np.ndarray):
        return 2 * np.random.rand(*x.shape).astype(x.dtype) - 1
    else:
        return 2 * torch.rand_like(x) - 1


def randn_like(x):
    if isinstance(x, np.ndarray):
        return np.random.randn(*x.shape).astype(x.dtype)
    else:
        return torch.randn_like(x)


def get_smoothing_transform(type, bandwidth, **clip_kwargs):
    def transform(x):
        return clip(x + bandwidth * {
            "uniform": rand_like,
            "gaussian": randn_like
        }[type](x), **clip_kwargs)

    return transform


bilinear = transforms.InterpolationMode.BILINEAR


def to_float32(x):
    return x.to(torch.float32)


def to_numpy(x):
    if x.ndim == 2:
        x = x[:, :, None]
    return np.array(x).transpose((2, 0, 1))


def c3hw(x):
    if x.shape[0] == 1:
        x = np.repeat(x, 3, axis=0)
    return x


def to_numpy3(x):
    return c3hw(to_numpy(x))


def transform_patch(transform, out_type="numpy", smooth_dict=None):
    if out_type == "0-1":
        transform.append(transforms.ToTensor())
    elif out_type == "norm":
        transform.extend([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
    elif out_type == "raw":
        transform.extend([transforms.PILToTensor(), to_float32])
    elif out_type == "numpy":
        transform.append(to_numpy)
    elif out_type == "numpy3":
        transform.append(to_numpy3)
    elif out_type == "smooth_0-1":
        assert smooth_dict is not None, "smooth_dict must be provided!"
        transform.extend([
            transforms.PILToTensor(), to_float32,
            get_smoothing_transform(**smooth_dict),
            partial(torch.div, other=255.)
        ])
    else:
        raise NotImplementedError(out_type)


class ImageFolder(tvds.ImageFolder):
    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            return [None, ], {None: 0}
        else:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            return classes, class_to_idx

    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        directory = os.path.expanduser(directory)

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.endswith(tuple(extensions))  # type: ignore[arg-type]

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class) if target_class else directory
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances


class NPZLoader:
    def __init__(
            self,
            npz_file,
            data_key="arr_0",
            label_key="arr_1",
            transform=None,
            **ignore_kwargs
    ):
        npz_data = np.load(npz_file)
        self.name = os.path.basename(npz_file)[:-4]
        self.data = npz_data[data_key]
        self.labels = npz_data.get(label_key, None)
        self.transform = transform

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = 0
        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def __len__(self):
        return len(self.data)


@register_dataset
class MNIST(tvds.MNIST):
    name = "mnist"
    shape = (1, 32, 32)

    def __init__(self, root=ROOT, train=True, out_type="numpy", smooth_dict=None, **kwargs):
        transform = [transforms.Resize((32, 32), interpolation=bilinear)]
        transform_patch(transform, out_type=out_type, smooth_dict=smooth_dict)
        transform = transforms.Compose(transform)
        super().__init__(root=root, train=train, transform=transform)

        self.out_type = out_type
        self.size = len(self.data)


@register_dataset
class CIFAR10(tvds.CIFAR10):
    name = "cifar10"
    shape = (3, 32, 32)

    def __init__(self, root=ROOT, train=True, hflip=False, out_type="numpy", smooth_dict=None, **kwargs):
        transform = []
        if hflip:
            transform.append(transforms.RandomHorizontalFlip())
        transform_patch(transform, out_type=out_type, smooth_dict=smooth_dict)
        transform = transforms.Compose(transform)
        super().__init__(root=root, train=train, transform=transform)

        self.out_type = out_type
        self.size = len(self.data)


@register_dataset
class CelebA(tvds.VisionDataset):
    """
    Large-scale CelebFaces Attributes (CelebA) Dataset [1]
    source: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    [^1]: Liu, Ziwei, et al. ‘Deep Learning Face Attributes in the Wild’.
     Proceedings of International Conference on Computer Vision (ICCV), 2015.
    """
    name = "celeba"
    base_folder = "celeba"
    shape = (3, 64, 64)

    def __init__(
            self,
            root=ROOT,
            split="all",
            download=False,
            hflip=False,
            out_type="numpy",
            smooth_dict=None,
            **kwargs
    ):
        transform = [crop_celeba, transforms.Resize((64, 64))]
        if hflip:
            transform.append(transforms.RandomHorizontalFlip())
        transform_patch(transform, out_type=out_type, smooth_dict=smooth_dict)
        transform = transforms.Compose(transform)
        super().__init__(root, transform=transform)
        self.split = split
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[split.lower()]
        splits = self._load_csv("list_eval_partition.txt")
        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()
        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.download = download

        self.out_type = out_type

    def _load_csv(
            self,
            filename,
            header=None,
    ):
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.as_tensor(data_int))

    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(
            self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        if self.transform is not None:
            X = self.transform(X)

        return X, 0

    def __len__(self):
        return len(self.filename)

    def extra_repr(self):
        lines = ["Split: {split}", ]
        return "\n".join(lines).format(**self.__dict__)


@register_dataset
class DSprites(tvds.VisionDataset):
    """
    dSprites - Disentanglement testing Sprites dataset [2]
    source: https://github.com/deepmind/dsprites-dataset
    [^2]: Matthey, Loic, et al. DSprites: Disentanglement Testing Sprites Dataset. 2017,
     https://github.com/deepmind/dsprites-dataset/.
    """
    name = "dsprites"
    shape = (1, 64, 64)
    binary = True
    url = "https://github.com/deepmind/dsprites-dataset/raw/fa310c66517cfc1939d77fe17c725154efc97127/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    npz_md5 = "7da33b31b13a06f4b04a70402ce90c2e"

    def __init__(self, root=ROOT, download=False, **kwargs):
        self.data_dir = os.path.join(root, "dsprites-dataset")
        npz_file = os.path.basename(self.url)
        self.data_dir = os.path.join(root, self.name)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.fpath = os.path.join(self.data_dir, npz_file)
        assert os.path.exists(self.fpath) or download, "Dataset NPZ file does not exists! Please set download=True!"
        if download:
            self.download()
        super().__init__(root=root, transform=None)
        self.data = self.load_data()

    def download(self):
        if not os.path.exists(self.fpath):
            wget.download(self.url, self.data_dir)
        with open(self.fpath, "rb") as f:
            assert hashlib.md5(f.read()).hexdigest() == self.npz_md5,\
                "MD5 Validation failed! Data file might be corrupted!"

    def load_data(self):
        return np.load(self.fpath)["imgs"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx, np.newaxis].astype("float32")), 0

