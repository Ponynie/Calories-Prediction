from torchvision import transforms
import os

data_dir = 'data'

num_classes = len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

random_state = 42

num_workers = int(os.cpu_count() / 2)

val_split = 0.2

test_split = 0.1

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

