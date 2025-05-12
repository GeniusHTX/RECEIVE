import csv
import numpy as np
from util import _size
import os
import torch.utils.data as data
from PIL import Image
from torchvision import datasets
from torchvision.transforms import Resize


class CelebA_attr(data.Dataset):
    def __init__(self, data_root, train, transform):
        self.split = 'train' if train else 'test'
        self.dataset = datasets.CelebA(root=data_root, split=self.split,
                                       target_type='attr', download=False)
        self.list_attributes = [18, 31, 21]
        self.transforms = transform

    @staticmethod
    def _convert_attributes(bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) \
               + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        inputs, target = self.dataset[index]
        inputs = self.transforms(inputs)
        target = self._convert_attributes(target[self.list_attributes])
        return inputs, target


class GTSRB_backup(data.Dataset):
    def __init__(self, data_root, train, transform):
        super(GTSRB, self).__init__()
        classes = ['Speed limit 20km/h',
                   'Speed limit 30km/h',
                   'Speed limit 50km/h',
                   'Speed limit 60km/h',
                   'Speed limit 70km/h',
                   'Speed limit 80km/h',  # 5
                   'End of speed limit 80km/h',
                   'Speed limit 100km/h',
                   'Speed limit 120km/h',
                   'No passing sign',
                   'No passing for vehicles over 3.5 metric tons',  # 10
                   'Right-of-way at the next intersection',
                   'Priority road sign',
                   'Yield sign',
                   'Stop sign',  # 14
                   'No vehicles sign',  # 15
                   'Vehicles over 3.5 metric tons prohibited',
                   'No entry',
                   'General caution',
                   'Dangerous curve to the left',
                   'Dangerous curve to the right',  # 20
                   'Double curve',
                   'Bumpy road',
                   'Slippery road',
                   'Road narrows on the right',
                   'Road work',  # 25
                   'Traffic signals',
                   'Pedestrians crossing',
                   'Children crossing',
                   'Bicycles crossing',
                   'Beware of ice or snow',  # 30
                   'Wild animals crossing',
                   'End of all speed and passing limits',
                   'Turn right ahead',
                   'Turn left ahead',
                   'Ahead only',  # 35
                   'Go straight or right',
                   'Go straight or left',
                   'Keep right',
                   'Keep left',
                   'Roundabout mandatory',  # 40
                   'End of no passing',
                   'End of no passing by vehicles over 3.5 metric tons']

        if train:
            self.data_folder = os.path.join(data_root, 'GTSRB/Train')
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(data_root, 'GTSRB/Test')
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transform

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + '/' + format(c, '05d') + '/'
            gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')
            gtReader = csv.reader(gtFile, delimiter=';')
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images = np.array(images)[indices]
        # labels = np.array(labels)[indices]
        # return images, labels
        return images

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, 'GT-final_test.csv')
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=';')
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + '/' + row[0])
            labels.append(int(row[7]))
        # return images, labels
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        # label = self.labels[index]
        return image


class GTSRB(data.Dataset):
    def __init__(self, data_root, train, transform):
        super(GTSRB, self).__init__()
        self.classes = ['Speed limit 20km/h',
                        'Speed limit 30km/h',
                        'Speed limit 50km/h',
                        'Speed limit 60km/h',
                        'Speed limit 70km/h',
                        'Speed limit 80km/h',  # 5
                        'End of speed limit 80km/h',
                        'Speed limit 100km/h',
                        'Speed limit 120km/h',
                        'No passing sign',
                        'No passing for vehicles over 3.5 metric tons',  # 10
                        'Right-of-way at the next intersection',
                        'Priority road sign',
                        'Yield sign',
                        'Stop sign',  # 14
                        'No vehicles sign',  # 15
                        'Vehicles over 3.5 metric tons prohibited',
                        'No entry',
                        'General caution',
                        'Dangerous curve to the left',
                        'Dangerous curve to the right',  # 20
                        'Double curve',
                        'Bumpy road',
                        'Slippery road',
                        'Road narrows on the right',
                        'Road work',  # 25
                        'Traffic signals',
                        'Pedestrians crossing',
                        'Children crossing',
                        'Bicycles crossing',
                        'Beware of ice or snow',  # 30
                        'Wild animals crossing',
                        'End of all speed and passing limits',
                        'Turn right ahead',
                        'Turn left ahead',
                        'Ahead only',  # 35
                        'Go straight or right',
                        'Go straight or left',
                        'Keep right',
                        'Keep left',
                        'Roundabout mandatory',  # 40
                        'End of no passing',
                        'End of no passing by vehicles over 3.5 metric tons']

        if train:
            self.input_array = np.load(f'{data_root}/gtsrb/train.npz')
        else:
            self.input_array = np.load(f'{data_root}/gtsrb/test.npz')
        self.data = self.input_array['x']
        self.targets = self.input_array['y'][:, 0].tolist()
        self.transforms = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target


class BackdoorDataset(data.Dataset):
    def __init__(self, dataset, reference_label, trigger_file, transform, source_label=-1):
        super(BackdoorDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.reference_label = reference_label
        self.trigger_input_array = np.load(trigger_file)
        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        img = np.array(img)
        img[:] = img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        img_backdoor = self.transform(Image.fromarray(img))
        return img_backdoor, self.reference_label

    def __len__(self):
        return len(self.dataset)


class BadCIFAR(BackdoorDataset):
    def __init__(self, data_root, train, transform, trigger_file, reference_label):
        super(BadCIFAR, self).__init__(
            dataset=datasets.CIFAR10(data_root, train=train),
            reference_label=reference_label,
            trigger_file=trigger_file,
            transform=transform
        )


class BadSVHN(BackdoorDataset):
    def __init__(self, data_root, train, transform, trigger_file, reference_label):
        di = {True: 'train', False: 'test'}
        super(BadSVHN, self).__init__(
            dataset=datasets.SVHN(data_root, di[train], transform=Resize(_size['svhn'][:2])),
            reference_label=reference_label,
            trigger_file=trigger_file,
            transform=transform
        )


class BadSTL(BackdoorDataset):
    def __init__(self, data_root, train, transform, trigger_file, reference_label):
        di = {True: 'train', False: 'test'}
        super(BadSTL, self).__init__(
            dataset=datasets.STL10(data_root, di[train], transform=Resize(_size['stl10'][:2])),
            reference_label=reference_label,
            trigger_file=trigger_file,
            transform=transform
        )


class BadGTSRB(BackdoorDataset):
    def __init__(self, data_root, train, transform, trigger_file, reference_label):
        super(BadGTSRB, self).__init__(
            dataset=GTSRB(data_root=data_root, train=train, transform=Resize(_size['gtsrb'][:2])),
            reference_label=reference_label,
            trigger_file=trigger_file,
            transform=transform
        )


class BadSTL_targeted(BackdoorDataset):
    def __init__(self, data_root, train, transform, trigger_file, reference_label):
        di = {True: 'train', False: 'test'}
        super(BadSTL_targeted, self).__init__(
            dataset=datasets.STL10(data_root, di[train], transform=Resize(_size['stl10'][:2])),
            reference_label=reference_label,
            trigger_file=trigger_file,
            transform=transform
        )

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = np.array(img)
        img[:] = img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        img_backdoor = self.transform(Image.fromarray(img))
        return img_backdoor, label
