import os
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image, resize
# import albumentations as A
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, \
 RandomVerticalFlip, ToTensor, Normalize
import torch

class Augmentation:
    def __init__(self, data_type='train'):
        if data_type == 'train':
            self.transform = Compose([
                #RandomHorizontalFlip(p=0.5),
                #RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            # 검증/테스트 데이터셋에 대한 변환 (증강 없음)
            self.transform = Compose([
                ToTensor(),
                Normalize(mean=[0.5], std=[0.5])
            ])

    def __call__(self, image, mask):
        seed = torch.random.seed()
        torch.manual_seed(seed)
        image = self.transform(image)

        # 마스크는 정규화하지 않음
        if 'Normalize' in str(self.transform.transforms[-1]):
            mask_transform = Compose(self.transform.transforms[:-1])  # Normalize 제외
        else:
            mask_transform = Compose(self.transform.transforms)

        torch.manual_seed(seed)
        mask = mask_transform(mask)

        return image, mask
'''

class Augmentation:
    def __init__(self, data_type='train'):
        if data_type == 'train':
            self.transform = Compose([
                #RandomHorizontalFlip(p=0.5),
                #RandomVerticalFlip(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                ToTensor(),
                Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            # 검증/테스트 데이터셋에 대한 변환 (증강 없음)
            self.transform = Compose([
                ToTensor(),
                Normalize(mean=[0.5], std=[0.5])
            ])

    def __call__(self, image, mask):
        seed = torch.random.seed()
        torch.manual_seed(seed)
        image = self.transform(image=img)

        # 마스크는 정규화하지 않음
        if 'Normalize' in str(self.transform.transforms[-1]):
            mask_transform = Compose(self.transform.transforms[:-1])  # Normalize 제외
        else:
            mask_transform = Compose(self.transform.transforms)

        torch.manual_seed(seed)
        mask = mask_transform(mask=mask)

        return image, mask
'''

class CLC_ClinicDBDataset(Dataset):
    def __init__(self, root_dir, data_type='train', transform=None, args=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, data_type, 'image')
        self.gt_dir = os.path.join(root_dir, data_type, 'mask')
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        self.args = args
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        gt_path = os.path.join(self.gt_dir, self.image_files[idx])

        image = Image.open(image_path).convert('L')
        gt_image = Image.open(gt_path).convert('L')


        #image = to_pil_image(image).convert('L')
        #gt_image = to_pil_image(gt_image)

        
        image = image.resize((self.args.img_size_w, self.args.img_size_h), resample=Image.BICUBIC)
        
        gt_image = gt_image.resize((self.args.img_size_w, self.args.img_size_h), resample=Image.NEAREST)
        #print(image.shape, gt_image.shape)
        if self.transform:
            image, gt_image = self.transform(image, gt_image)
        else:
            image = to_tensor(image)
            gt_image = to_tensor(gt_image)

        return image, gt_image



class CLC_ClinicDBDataset_test(Dataset):
    def __init__(self, root_dir, data_type='test', transform=None, args=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, data_type, 'image')
        self.gt_dir = os.path.join(root_dir, data_type, 'mask')
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        self.args = args
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        gt_path = os.path.join(self.gt_dir, self.image_files[idx])

        image = Image.open(image_path).convert('L')
        gt_image = Image.open(gt_path).convert('L')


        #image = to_pil_image(image).convert('L')
        #gt_image = to_pil_image(gt_image)

        
        image = image.resize((224, 224), resample=Image.BICUBIC)
        
        gt_image = gt_image.resize((224, 224), resample=Image.NEAREST)
        #print(image.shape, gt_image.shape)
        if self.transform:
            image, gt_image = self.transform(image, gt_image)
        else:
            image = to_tensor(image)
            gt_image = to_tensor(gt_image)

        return image, gt_image