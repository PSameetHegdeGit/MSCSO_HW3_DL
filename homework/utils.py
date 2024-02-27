import torch
import torchvision.datasets
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import random

from . import dense_transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
DENSE_LABEL_NAMES = ['background', 'kart', 'track', 'bomb/projectile', 'pickup/nitro']
# Distribution of classes on dense training set (background and track dominate (96%)
DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]



class SuperTuxDataset(Dataset):


    def __init__(self, dataset_path, probability_threshold=1, resize=None, random_crop=None, random_horizontal_flip=False, random_color_jitter=False, normalize=False, is_resnet=False):
        import csv
        from os import path

        transform_image = self.get_transform(resize, random_crop, random_horizontal_flip, random_color_jitter, normalize, is_resnet)
        default_transform = transforms.ToTensor()
        self.data = []
        print(f"probability threshold used: {probability_threshold}")

        with open(path.join(dataset_path, 'labels.csv'), newline='') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                fname, label, _ = row
                if label in LABEL_NAMES:
                    image = Image.open(path.join(dataset_path, fname))
                    label_id = LABEL_NAMES.index(label)

                    # as probability_threshold approaches 1, percentage of images transformed also approaches 100%, and vice versa relationship is also true
                    if random.random() < probability_threshold:
                        image = transform_image(image)
                    else:
                        image = default_transform(image)

                    self.data.append((image, label_id))

    def get_transform(self, resize=None, random_crop=None, random_horizontal_flip=False, color_jitter = False, normalize=False, is_resnet=False):
        """
        applies transform on image set

        """

        if is_resnet:
            return transforms.Compose([
               # torchvision.transforms.Scale(256), Scale fxn is not recognized
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])

        transform = []
        if resize is not None:
            transform.append(transforms.Resize(resize))
        if random_crop is not None:
            transform.append(transforms.RandomResizedCrop(random_crop))
        if random_horizontal_flip:
            transform.append(transforms.RandomHorizontalFlip())
        if color_jitter:
            transform.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        transform.append(transforms.ToTensor())
        if normalize:
            transform.append(
                transforms.Normalize(mean=[0.4701, 0.4308, 0.3839], std=[0.2595, 0.2522, 0.2541]))
        #always add to tensor transform
        return transforms.Compose(transform)


    def __len__(self):
        """
        Your code here
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        """
        return self.data[idx]

class DenseSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*_im.jpg')):
            self.files.append(im_f.replace('_im.jpg', ''))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        im = Image.open(b + '_im.jpg')
        lbl = Image.open(b + '_seg.png')
        if self.transform is not None:
            im, lbl = self.transform(im, lbl)
        return im, lbl


def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = SuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_dense_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


if __name__ == '__main__':

    # Can use the below for visualizing the transforms

    dense_transforms_compose = dense_transforms.Compose(
        [
            dense_transforms.RandomHorizontalFlip(),
            dense_transforms.ColorJitter(brightness=0.5, contrast=1, saturation=1, hue=0.5),
            dense_transforms.ToTensor()
        ]
    )

    dataset = DenseSuperTuxDataset('dense_data/train', transform=dense_transforms_compose)
    from pylab import show, imshow, subplot, axis, figure

    figure(figsize=(30,20))
    for i in range(3):
        im, lbl = dataset[i]
        if i == 1:
            print(lbl)
        subplot(5, 6, 2 * i + 1)
        imshow(F.to_pil_image(im))
        axis('off')
        subplot(5, 6, 2 * i + 2)
        imshow(dense_transforms.label_to_pil_image(lbl))
        axis('off')
    show()
    import numpy as np

    c = np.zeros(5)
    for im, lbl in dataset:
        c += np.bincount(lbl.view(-1), minlength=len(DENSE_LABEL_NAMES))
    print(100 * c / np.sum(c))
