from torch import nn
import torch
from transformers import AutoImageProcessor, SwinModel
import torch
import cv2
from copy import deepcopy
from transformer import *
from position_encoding import *


class BioTransformer(nn.Module):
    def __init__(self, num_classes,d_model=768,nhead=8,num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,return_intermediate_dec=False):
        super().__init__()
        self.faces_encoder = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.fundus_encoder = deepcopy(self.faces_encoder)
        self.tongues_encoder = deepcopy(self.faces_encoder)

        self.face_pos=nn.Parameter(torch.randn(49,d_model))
        self.fundu_pos=nn.Parameter(torch.randn(49,d_model))
        self.tongue_pos=nn.Parameter(torch.randn(49,d_model))

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)

        self.decoderA = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec) #face as Q  fundu as KV
        self.decoderB= deepcopy(self.decoderA) # fundu as Q tongue as KV
        self.decoderC=deepcopy(self.decoderA) # tongue as Q face as KV

        self.classifier = nn.Linear(3*d_model, num_classes)
    
    def forward(self,faces,fundus,tongues):
        face_output=self.faces_encoder(**faces).last_hidden_state.permute(1,0,2)
        fundu_output=self.fundus_encoder(**fundus).last_hidden_state.permute(1,0,2)
        tongue_output=self.tongues_encoder(**tongues).last_hidden_state.permute(1,0,2)
        
        face_pos=self.face_pos.unsqueeze(0).repeat((face_output.shape[1],1,1)).permute(1,0,2)
        fundu_pos=self.fundu_pos.unsqueeze(0).repeat((fundu_output.shape[1],1,1)).permute(1,0,2)
        tongue_pos=self.tongue_pos.unsqueeze(0).repeat((tongue_output.shape[1],1,1)).permute(1,0,2)

        tgtA = torch.zeros_like(face_output)#[query]
        hsA = self.decoderA(tgtA, fundu_output, memory_key_padding_mask=None,
                        pos=fundu_pos, query_pos=face_output+face_pos).transpose(1,2)[0]
        
        tgtB = torch.zeros_like(fundu_output)#[query]
        hsB = self.decoderB(tgtB, tongue_output, memory_key_padding_mask=None,
                        pos=tongue_pos, query_pos=fundu_output+fundu_pos).transpose(1,2)[0]

        tgtC = torch.zeros_like(tongue_output)#[query]
        hsC = self.decoderC(tgtC, face_output, memory_key_padding_mask=None,
                        pos=face_pos, query_pos=tongue_output+tongue_pos).transpose(1,2)[0]
        
        hsA=torch.mean(hsA,dim=1)
        hsB=torch.mean(hsB,dim=1)
        hsC=torch.mean(hsC,dim=1)

        hs=torch.cat([hsA,hsB,hsC],dim=1)
        return self.classifier(hs)

        #print(face_output.shape)

if  __name__ == '__main__':
  b=1
  w=224
  h=224
  c=3
  faces=torch.rand(b,c,w,h).cuda()
  tongues=torch.rand(b,c,w,h).cuda()
  fundus = torch.rand(b,c,w,h).cuda()
  fundu=cv2.imread('image/eye.jpg')
  face=cv2.imread('image/face.jpg')
  tongue=cv2.imread('image/tongue.jpg')
  #print(faces,faces.shape)
  image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
  #model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
  fundus = image_processor(fundu, return_tensors="pt")
  faces = image_processor(face, return_tensors="pt")
  tongues = image_processor(tongue, return_tensors="pt")
  print(fundus['pixel_values'].shape)
  model=BioTransformer(100)
  
  result=model(faces,fundus,tongues)
  print(result.shape)