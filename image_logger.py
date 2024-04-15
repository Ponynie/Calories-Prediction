from pytorch_lightning.callbacks import Callback
import torch
import wandb

class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32, class_names=None):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
        self.class_names = class_names
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{self.match_name(pred)}, Label:{self.match_name(y)}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })

    def match_name(self, n):
        if self.class_names:
            return self.class_names[n]
        return n
