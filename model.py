import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torchvision.models import mobilenet_v2

class MobileNetV2Lightning(pl.LightningModule):
    def __init__(self, num_classes, lr=0.001, last_drop=0.2, lr_decay=True, gamma=0.90):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.gamma = gamma
        self.lr_decay = lr_decay
        self.model = mobilenet_v2(weights='DEFAULT')
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=last_drop, inplace=False),
            nn.Linear(self.model.last_channel, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer