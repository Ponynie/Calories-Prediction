# Standard library imports
import os
from argparse import ArgumentParser

# Third-party library imports
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from PIL import Image, ImageFile
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from captum.attr import NoiseTunnel

# Local imports
import properties as pt
from image_logger import ImagePredictionLogger
from data_module import ImageDataModule
from model import MobileNetV2Lightning

# Adjusting PIL behavior
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_model(hparams):
    
    data_dir = 'augmented'
    
    batch_size = int(hparams.batch_size)
    lr = float(hparams.lr)
    last_drop = float(hparams.last_drop)
    max_epochs = int(hparams.max_epochs)
    patience = int(hparams.patience)
    gamma = float(hparams.gamma)
    lr_decay = bool(int(hparams.lr_decay))

    data_module = ImageDataModule(data_dir=data_dir, 
                                 batch_size=batch_size, 
                                 num_workers=pt.num_workers, 
                                 transform=pt.transform, 
                                 val_split=pt.val_split, 
                                 test_split=pt.test_split, 
                                 random_state=pt.random_state)
    
    data_module.setup()
    val_samples = next(iter(data_module.val_dataloader()))

    print(f"Train on {data_dir} data with {data_module.num_classes} classes")
    print(f"Training model with the following hyperparameters:\n"
        f"   - Batch Size   : {batch_size}\n"
        f"   - Learning Rate: {lr}\n"
        f"   - Last Drop    : {last_drop}\n"
        f"   - Max Epochs   : {max_epochs}\n"
        f"   - Patience     : {patience}\n"
        f"   - LR Decay     : {lr_decay}\n"
        f"   - Gamma        : {gamma}\n")
    
    model = MobileNetV2Lightning(num_classes=data_module.num_classes, 
                                 lr=lr, 
                                 last_drop=last_drop, 
                                 lr_decay=lr_decay, 
                                 gamma=gamma)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    wandb_logger = WandbLogger(project='MLProject', job_type='train')
    image_logger = ImagePredictionLogger(val_samples=val_samples, num_samples=32, class_names=data_module.class_names)
    
    trainer = Trainer(devices='auto', 
                      accelerator='auto', 
                      max_epochs=max_epochs,
                      logger=wandb_logger, 
                      enable_checkpointing=True,
                      callbacks=[lr_monitor, early_stopping, image_logger])
    
    trainer.fit(model, datamodule=data_module)

def test_model():
    checkpoint_path = 'MLProject/jp5s24vj/checkpoints/epoch=29-step=3570.ckpt'
    model = MobileNetV2Lightning.load_from_checkpoint(checkpoint_path)

    data_module = ImageDataModule(data_dir='augmented', 
                                 batch_size=32, 
                                 num_workers=pt.num_workers, 
                                 transform=pt.transform, 
                                 val_split=pt.val_split, 
                                 test_split=pt.test_split, 
                                 random_state=pt.random_state)
    
    wandb_logger = WandbLogger(project='MLProject', job_type='test')
    
    trainer = Trainer(logger=wandb_logger, devices='auto', accelerator='auto')
    trainer.test(model, datamodule=data_module)

def analyze_results():
    checkpoint_path = 'MLProject/jp5s24vj/checkpoints/epoch=29-step=3570.ckpt'
    model = MobileNetV2Lightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    image = Image.open('analyze/2-Klongphai-Farm-session-220786-Edit_augmented_4.jpg').convert('RGB')
    image_tensor = pt.transform(image).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(image_tensor)
    pred = torch.argmax(logits, dim=1).item()

    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(image_tensor, target=pred, n_steps=200)
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)
    _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                 np.transpose(image_tensor.squeeze().cpu().detach().numpy(), (1,2,0)),
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='positive',
                                 outlier_perc=1)
    
    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions_ig_nt = noise_tunnel.attribute(image_tensor, nt_samples=10, nt_type='smoothgrad_sq', target=pred)
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          np.transpose(image_tensor.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          ["original_image", "heat_map"],
                                          ["all", "positive"],
                                          cmap=default_cmap,
                                          show_colorbar=True)
        
def predict_image(image_path):
    food_list = ['frenchfries', 'gaithod', 'gaiyang', 'greencurry', 'hamburger', 'kaijjaw', 'kaomokgai', 'kapraomukrob', 'kapraomukrob_egg', 'kapraomusub', 'kapraomusub_egg', 'mamuang', 'padseaew', 'padthai', 'pizza', 'somtam', 'tomkha', 'tomyumkung']
    checkpoint_path = 'MLProject/jp5s24vj/checkpoints/epoch=29-step=3570.ckpt'
    model = MobileNetV2Lightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = pt.transform(image).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(image_tensor)
    pred = torch.argmax(logits, dim=1).item()
    print(food_list[pred])

def export_model():
    # from torch.utils.mobile_optimizer import optimize_for_mobile
    
    checkpoint_path = 'MLProject/jp5s24vj/checkpoints/epoch=29-step=3570.ckpt'
    model = MobileNetV2Lightning.load_from_checkpoint(checkpoint_path)
    
    example_input = torch.rand(1, 3, 224, 224)
    traced_module = model.to_torchscript(method='trace', example_inputs=example_input)
    # optimized_model = optimize_for_mobile(traced_module)
    # optimized_model.save('model.pt')
    traced_module.save('model.pt')

def main(args):
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model()
    elif args.mode == 'analyze':
        analyze_results()
    elif args.mode == 'predict' and args.predict_path is not None:
        predict_image(args.predict_path)
    elif args.mode == 'export':
        export_model()
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--lr", default=0.005)
    parser.add_argument("--last_drop", default=0.2)
    parser.add_argument("--max_epochs", default=20)
    parser.add_argument("--patience", default=5)
    parser.add_argument("--lr_decay", default=1)
    parser.add_argument("--gamma", default=0.90)
    parser.add_argument("--mode", default='train')
    parser.add_argument("--predict_path", default=None)
    args = parser.parse_args()
    main(args)







