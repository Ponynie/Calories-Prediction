from data_module import ImageDataModule
from model import MobileNetV2Lightning
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
import constant
import os

def main(hparams):
    
    batch_size = hparams['batch_size']
    lr = hparams['lr']
    last_drop = hparams['last_drop']
    max_epochs = hparams['max_epochs']
    patience = hparams['patience']
    
    num_classes = constant.num_classes
    random_state = constant.random_state
    num_workers = constant.num_workers
    transform = constant.transform
    val_split = constant.val_split
    test_split = constant.test_split
    
    dataModule = ImageDataModule(data_dir='data', batch_size=batch_size, num_workers=num_workers, transform=transform, val_split=val_split, test_split=test_split, random_state=random_state)
    model = MobileNetV2Lightning(num_classes=num_classes, lr=lr, last_drop=last_drop)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    
    trainer = Trainer(devices='auto', 
                      accelerator='auto', 
                      max_epochs=max_epochs,
                      logger=True, 
                      enable_checkpointing=True,
                      callbacks=[early_stopping])
    
    trainer.fit(model, datamodule=dataModule)
    
    
if __name__ == '__main__':
    hparams = {
        'batch_size': 64,
        'lr': 0.001,
        'last_drop': 0.2,
        'max_epochs': 20,
        'patience': 5
    }
    main(hparams)
