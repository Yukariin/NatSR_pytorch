from os import listdir
from os.path import join

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
