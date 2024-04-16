from torch import jit
from PIL import Image
import torch
from torchvision import transforms


model = jit.load('model.pt')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

food_list = ['frenchfries', 
             'gaithod', 
             'gaiyang', 
             'greencurry', 
             'hamburger', 
             'kaijjaw', 
             'kaomokgai', 
             'kapraomukrob', 
             'kapraomukrob_egg', 
             'kapraomusub', 
             'kapraomusub_egg', 
             'mamuang', 
             'padseaew', 
             'padthai', 
             'pizza', 
             'somtam', 
             'tomkha', 
             'tomyumkung']

image_path = 'analyze/source_image/2-Klongphai-Farm-session-220786-Edit_augmented_4.jpg' #ใส่ PATH ตรงนี้
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    logits = model(image_tensor)
pred = torch.argmax(logits, dim=1).item()

print(food_list[pred], pred)