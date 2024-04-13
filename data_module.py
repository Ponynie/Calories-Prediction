import os
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split
from PIL import Image
from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
ImageFile.LOAD_TRUNCATED_IMAGES = True
class ImageDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image_path']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.data.iloc[idx]['label']
        return image, label

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=1, transform=None, val_split=0.2, test_split=0.2, random_state=42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state

    def setup(self, stage: Optional[str] = None):
        data = self.load_data()
        train_data, test_val_data = train_test_split(data, test_size=self.val_split + self.test_split, random_state=self.random_state, stratify=data['label'])
        val_data, test_data = train_test_split(test_val_data, test_size=self.test_split / (self.val_split + self.test_split), random_state=self.random_state, stratify=test_val_data['label'])

        if stage == 'fit' or stage is None:
            self.train_dataset = ImageDataset(train_data, transform=self.transform)
            self.val_dataset = ImageDataset(val_data, transform=self.transform)

        if stage == 'test' or stage is None:
            self.test_dataset = ImageDataset(test_data, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def load_data(self):
        image_paths = []
        labels = []
        class_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        class_names = sorted(class_dirs)

        for idx, class_name in enumerate(class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            for img_path in os.listdir(class_dir):
                if img_path == '.DS_Store':  # Skip .DS_Store files
                    continue
                image_path = os.path.join(class_dir, img_path)
                image_paths.append(image_path)
                labels.append(idx)

        data = pd.DataFrame({'image_path': image_paths, 'label': labels})
        return data
