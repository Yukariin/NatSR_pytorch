from os import listdir
from os.path import join
import sqlite3
import io

import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import RandomCrop, Resize
from torchvision.transforms.functional import to_tensor


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, patch_size=48, scale_factor=4, interpolation=Image.BICUBIC):
        super().__init__()

        self.samples = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.cropper = RandomCrop(size=patch_size*scale_factor)
        self.resizer = Resize(patch_size, interpolation)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        img = Image.open(sample_path).convert('RGB')
        target = self.cropper(img)
        input = target.copy()
        input = self.resizer(input)

        return to_tensor(input), to_tensor(target)


class SQLDataset(data.Dataset):
    def __init__(self, db_file, db_table='images', hr_col='hr_img', lr_col='lr_img'):
        self.db_file = db_file
        self.db_table = db_table
        self.lr_col = lr_col
        self.hr_col = hr_col
        self.total_images = self.get_num_rows()

    def get_num_rows(self):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT MAX(ROWID) FROM {self.db_table}')
            db_rows = cursor.fetchone()[0]

        return db_rows

    def __len__(self):
        return self.total_images

    def __getitem__(self, item):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT {self.lr_col}, {self.hr_col} FROM {self.db_table} WHERE ROWID={item+1}')
            lr, hr = cursor.fetchone()

        lr = Image.open(io.BytesIO(lr)).convert("RGB")
        hr = Image.open(io.BytesIO(hr)).convert("RGB")

        return to_tensor(lr), to_tensor(hr)


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0
