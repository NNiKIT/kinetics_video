# Modify from torchvision
import os
import torch
import torch.utils.data as data
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    
    return pil_loader(path)


class ImageData(data.Dataset):
    
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        import platform
        import sys
        the_os = platform.system()
        if the_os == 'Windows' and sys.version_info[0] == 3:
            with open(root, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            import codecs
            with codecs.open(root, 'r', 'utf-8') as f:
                lines = f.readlines()
        imgs = [(os.path.join(os.path.split(root)[0], x.strip().split(' ')[0]), int(x.strip().split(' ')[1]))
                for x in lines]

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)