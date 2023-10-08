import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from pytorch_utils.module import LightningModule
from network import ViewModel
  
    
class ViewQualityModel(LightningModule):
    def __init__(self, opt=None, **kwargs):
        super().__init__(opt, **kwargs)
        self.model = ViewModel(latent_size=self.opt.latent_size, mlp_layers=self.opt.mlp_layers, dropout=self.opt.dropout, sigmoid=False)
        self.binary_loss = torch.nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1 = BinaryF1Score()
                  
    def forward(self, batch, batch_idx, split):
        B = batch[0].shape[0]
        imagesA = batch[0]
        imagesB = batch[1]
        target = batch[3]
        
        logits = self.model(imagesA, imagesB) # (B, 1)    
        pred = logits.sigmoid()
        
        # loss        
        Loss   = self.binary_loss(logits, target)
                
        self.log_value("acc", self.accuracy(pred, target.round()), split, B)
        self.log_value("precision", self.precision(pred, target.round()) , split, B)
        self.log_value("recall", self.recall(pred, target.round()) , split, B)
        self.log_value("f1", self.f1(pred, target.round()), split, B)
        self.log_value("loss", Loss, split, B)
        
        return {'loss': Loss, 'pred': pred}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay, betas=[0.9, 0.98], eps=1.0e-6)
        if self.opt.lr_patience > 0:
            print("Adding learning rate scheduler on plateau")
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.opt.lr_patience, factor=self.opt.factor),
                'monitor': 'valid_loss'
            }
            return [optimizer], [scheduler]
        else:
            return optimizer

    