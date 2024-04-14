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
    
    if hparams.mode == 'real':
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

    dataModule = ImageDataModule(data_dir=data_dir, 
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
    
    trainer.fit(model, datamodule=dataModule)

# def test_model():
#     model = MobileNetV2Lightning.load_from_checkpoint(
#         checkpoint_path="/path/to/pytorch_checkpoint.ckpt",
#         hparams_file="/path/to/experiment/version/hparams.yaml",
#         map_location=None,
#     )

#     trainer = Trainer(...)
#     trainer.test(model)

# def predict():

#     model = MobileNetV2Lightning.load_from_checkpoint(PATH)
#     dataset = WikiText2()
#     test_dataloader = DataLoader(dataset)
#     trainer = L.Trainer()
#     pred = trainer.predict(model, dataloaders=test_dataloader)

#     return pred

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--lr", default=0.005)
    parser.add_argument("--last_drop", default=0.2)
    parser.add_argument("--max_epochs", default=20)
    parser.add_argument("--patience", default=5)
    parser.add_argument("--mode", default=None)
    args = parser.parse_args()
    train_model(args)







