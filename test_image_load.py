import os
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import properties as pt

def test_load_data(data_dir: str):
        
    image_paths = []
    labels = []
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_names = sorted(class_dirs)

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_path in os.listdir(class_dir):
            if img_path == '.DS_Store':  # Skip .DS_Store files
                continue
            image_path = os.path.join(class_dir, img_path)
            image_paths.append(image_path)
            labels.append(idx)

    data = pd.DataFrame({'image_path': image_paths, 'label': labels})
    return data

def test_image_tranform(data: pd.DataFrame, transform=None):
    success_count = 0
    error_count = 0
    
    for idx in range(len(data)):
        image_path = data.iloc[idx]['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
            if transform:
                image = transform(image)
            success_count += 1
        except Exception as e:
            print(f"Error loading or transforming image {image_path}: {e}")
            error_count += 1

    if error_count == 0:
        print("Successfully loaded and transformed all images.")
    else:
        print(f"{success_count} images loaded and transformed successfully, {error_count} errors occurred.")

if __name__ == '__main__':
    data_dir = 'augmented'
    tranform = pt.transform
    test_image_tranform(test_load_data(data_dir), transform=tranform)