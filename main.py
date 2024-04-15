# Standard library imports
import os
from argparse import ArgumentParser

# Third-party library imports
from PIL import Image, ImageFile
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

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
    checkpoint_path = ''
    model = MobileNetV2Lightning.load_from_checkpoint(checkpoint_path)

    data_module = ImageDataModule(data_dir='', 
                                 batch_size=32, 
                                 num_workers=pt.num_workers, 
                                 transform=pt.transform, 
                                 val_split=pt.val_split, 
                                 test_split=pt.test_split, 
                                 random_state=pt.random_state)

    trainer = Trainer()
    trainer.test(model, datamodule=data_module)

# def predict():

#     model = MobileNetV2Lightning.load_from_checkpoint(PATH)
#     dataset = WikiText2()
#     test_dataloader = DataLoader(dataset)
#     trainer = L.Trainer()
#     pred = trainer.predict(model, dataloaders=test_dataloader)

#     return pred

def main(args):
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model()
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
    args = parser.parse_args()
    main(args)







