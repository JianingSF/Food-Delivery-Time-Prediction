import torch
import torch.nn as nn
from collections import OrderedDict
import re

#Model
class mlpModel(nn.Module):
    def __init__(self, inputSize, extraDefine, 
    	embDim4Category, embNum4Category, 
    	embDim4Protocol, embNum4Protocol, 
    	embDim4TotalItems, embNum4TotalItems,
    	embDim4DistinctItems, embNum4DistinctItems, 
    	dropout=0, useBilinear=False):
    	
        super(mlpModel, self).__init__()
        mod = OrderedDict()
        layer_num = 0
        if extraDefine is not None and extraDefine != "None":
        
            for ly in extraDefine.split("|"):
                if re.search("^\d+$", ly):
                    nnSize = int(ly)
                    mod["layer_%d"%layer_num] = nn.Linear(inputSize,nnSize)
                    inputSize = nnSize
                elif ly in ("R", "r"):
                    mod["layer_%d"%layer_num] = nn.ReLU()
                elif ly in ("S", "s"):
                    mod["layer_%d"%layer_num] = nn.Sigmoid()
                elif ly in ("T", "t"):
                    mod["layer_%d"%layer_num] = nn.Tanh()
                elif ly in ("Se", "se"):
                    mod["layer_%d"%layer_num] = nn.SELU()
                else:
                    raise ValueError("Only number or R|T|S|Se type is supported.")
                if ly in ("R", "r", "S", "s", "T", "t", "Se", "se") and dropout > 0:
                    layer_num = layer_num + 1
                    mod["layer_%d"%layer_num] = nn.Dropout(dropout)
                layer_num = layer_num + 1
        self.linear  = nn.Linear(inputSize,1)
        
        #use bilinear
        if useBilinear:
            self.Bilinear = nn.Bilinear(inputSize,inputSize,1, bias=False)
        else:
            self.Bilinear = None
        
        self.mod = nn.Sequential(mod)
        self.embedding1 = nn.Embedding(embNum4Category, embDim4Category)
        self.embedding2 = nn.Embedding(embNum4Protocol, embDim4Protocol)
        self.embedding3 = nn.Embedding(embNum4TotalItems, embDim4TotalItems)
        self.embedding4 = nn.Embedding(embNum4DistinctItems, embDim4DistinctItems)
        
    def forward(self, input):
    	#input[0]: continuous feature
    	#input[1-4]: discrete category feature
    	y1=self.embedding1.forward(input[1])
    	y2=self.embedding2.forward(input[2])
    	y3=self.embedding3.forward(input[3])
    	y4=self.embedding4.forward(input[4])
    	x=torch.cat((input[0],y1,y2,y3,y4),1)
    	out = self.mod(x)
    	if self.Bilinear is not None:
            out = self.linear(out) + self.Bilinear(out, out)
        else:
            out = self.linear(out)     	
        return out
