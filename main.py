from data_module import ImageDataModule
from model import MobileNetV2Lightning
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning import Trainer
import properties as pt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from argparse import ArgumentParser
import os

def train_model(hparams):
    
    if hparams.data == 'real':
        data_dir = 'augmented'
    else:
        data_dir = 'data'
    
    num_classes = len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    batch_size = int(hparams.batch_size)
    lr = float(hparams.lr)
    last_drop = float(hparams.last_drop)
    max_epochs = int(hparams.max_epochs)
    patience = int(hparams.patience)
    
    print(f"Training model with the following hyperparameters:\n"
        f"   - Batch Size   : {batch_size}\n"
        f"   - Learning Rate: {lr}\n"
        f"   - Last Drop    : {last_drop}\n"
        f"   - Max Epochs   : {max_epochs}\n"
        f"   - Patience     : {patience}")
    
    random_state = pt.random_state
    num_workers = pt.num_workers
    transform = pt.transform
    val_split = pt.val_split
    test_split = pt.test_split

    data_module = ImageDataModule(data_dir=data_dir, 
                                 batch_size=batch_size, 
                                 num_workers=num_workers, 
                                 transform=transform, 
                                 val_split=val_split, 
                                 test_split=test_split, 
                                 random_state=random_state)
    
    model = MobileNetV2Lightning(num_classes=num_classes, lr=lr, last_drop=last_drop)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = Trainer(devices='auto', 
                      accelerator='auto', 
                      max_epochs=max_epochs,
                      logger=True, 
                      enable_checkpointing=True,
                      callbacks=[lr_monitor, early_stopping])
    
    trainer.fit(model, datamodule=data_module)

def test_model():
    model = MobileNetV2Lightning.load_from_checkpoint(
        checkpoint_path="lightning_logs/version_1/checkpoints/epoch=13-step=28.ckpt",
        hparams_file="lightning_logs/version_1/hparams.yaml",
        map_location=None,
    )

    data_module = None # Load the data module here

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
    parser.add_argument("--data", default=None)
    parser.add_argument("--mode", default='train')
    args = parser.parse_args()
    main(args)







