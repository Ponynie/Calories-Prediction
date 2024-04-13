from data_module import ImageDataModule
from model import MobileNetV2Lightning
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning import Trainer
import properties as pt

def main(hparams):
    
    batch_size = hparams['batch_size']
    lr = hparams['lr']
    last_drop = hparams['last_drop']
    max_epochs = hparams['max_epochs']
    patience = hparams['patience']
    
    num_classes = pt.num_classes
    random_state = pt.random_state
    num_workers = pt.num_workers
    transform = pt.transform
    val_split = pt.val_split
    test_split = pt.test_split
    data_dir = pt.data_dir
    
    dataModule = ImageDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, transform=transform, val_split=val_split, test_split=test_split, random_state=random_state)
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
    
if __name__ == '__main__':
    hparams = {
        'batch_size': 64,
        'lr': 0.005,
        'last_drop': 0.2,
        'max_epochs': 20,
        'patience': 5
    }
    main(hparams)
