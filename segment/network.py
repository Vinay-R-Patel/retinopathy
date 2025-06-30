import torch
import torch.nnasnn
import segmentation_models_pytorchassmp


class ModularSegmentationModel(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", classes=3, segmentation_head="Segfor mer"):
        super().__init__()


model_class= getattr(smp, segmentation_head)

full_model= model_class(
encoder_name= encoder_name,
encoder_weights= encoder_weights,
in_channels=3,
classes= classes,
activation= None
)

self.encoder= full_model.encoder
self.decoder= full_model.decoder
self.segmentation_head= full_model.segmentation_head

def forward(self, x):
        encoder_features= self.encoder(x)
decoder_output= self.decoder(encoder_features)
segmentation_output= self.segmentation_head(decoder_output)
return segmentation_output


def create_model(cfg):

    segmentation_head= getattr(cfg.model,'segmentation_head','Segfor mer')

model= ModularSegmentationModel(
encoder_name= cfg.model.name,
encoder_weights= cfg.model.encoder_weights,
classes= cfg.model.num_classes,
segmentation_head= segmentation_head
)
return model