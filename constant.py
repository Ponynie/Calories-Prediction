from torchvision import transforms
import os

num_classes = 10

random_state = 42

num_workers = int(os.cpu_count() / 2)

val_split = 0.2

test_split = 0.1

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])