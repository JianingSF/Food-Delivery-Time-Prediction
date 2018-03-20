import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.stats as st
import random

from dataManager import dataManager
from model import mlpModel

def parseCmd():
    parser = argparse.ArgumentParser(description='Trainer for delivery time estimation')
    parser.add_argument("--trainData", type=str, default="train_data.csv", help="Training data csv file") 
    parser.add_argument("--validData", type=str, default="validation_data.csv", help="Training data csv file") 
    parser.add_argument("--modelDefine", default='None', help="extra definition of the model")
    parser.add_argument("--useBilinear", action="store_true", help="control to build model with polynomial terms")

    parser.add_argument("--dropout", default=0.0, type=float, help="dropout rate for the model") 
    parser.add_argument("--learningRate", type=float, default=0.05, help="learning rate")
    parser.add_argument("--batchSize", type=int, default=64, help="miniBatch size")
    parser.add_argument("--maxEpoch", type=int, default=30, help="maximum epoch size")
    
    parser.add_argument("--embDim4Category", type=int, default=10, help="embedding store category")
    parser.add_argument("--embDim4Protocol", type=int, default=5, help="embedding Protocol") 
    parser.add_argument("--embDim4TotalItems", type=int, default=10, help="embedding TotalItems")    
    parser.add_argument("--embDim4DistinctItems", type=int, default=10, help="embedding DistinctItems")
    
    parser.add_argument("--epochNumber4Valid", type=int, default=2, help="do validation every epochNumber4Valid epochs")
    parser.add_argument("--savedModel", type=str, default="savedModel.net", help="Save model")
    return parser.parse_args()

def main():
    opts = parseCmd()
    print(opts)
    torch.manual_seed(999)
    random.seed(999)
    embNum4Category=75
    embNum4Protocol=8
    embNum4TotalItems=30
    embNum4DistinctItems=20
    
    model = mlpModel(5+opts.embDim4Category+opts.embDim4Protocol+ opts.embDim4TotalItems + opts.embDim4DistinctItems, 
    	opts.modelDefine, 
    	opts.embDim4Category, embNum4Category, 
    	opts.embDim4Protocol, embNum4Protocol,
    	opts.embDim4TotalItems, embNum4TotalItems,
    	opts.embDim4DistinctItems, embNum4DistinctItems,
    	opts.dropout, opts.useBilinear)
    	
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=opts.learningRate)

    trainData = dataManager(opts.trainData, opts.batchSize)
    validData = dataManager(opts.validData, opts.batchSize, trainData , False)
    
    epoch = 1
    preValCost = 10000000
    doValid = False
    totalLoss=0
    epochNumber4Valid=opts.epochNumber4Valid
    while epoch <=opts.maxEpoch:
    	inputs, targets, targetOrig = trainData.nextBatch()
    	
        # forward
        out = model(inputs)
        loss = criterion(out, targets)
        totalLoss=totalLoss+loss.data[0]
          
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if trainData.isFinish():
        	if epoch%epochNumber4Valid == 0:
        		doValid = True
        	epoch = epoch + 1
        	
        #do validation
        if doValid:
        	doValid = False
        	inputData,targetData, targetOrig=validData.getValidationData()
        	valCost = doValidation(inputData, targetData, targetOrig, model, criterion, trainData)
        	#adjust learningRate when needed
        	if valCost[0].data[0]>preValCost and opts.learningRate>0.001:
        		opts.learningRate = opts.learningRate/2
        		optimizer.param_groups[0]['lr'] = opts.learningRate
        	preValCost = valCost[0].data[0]
        	
        	print('Epoch[{}/{}], Train MSE: {:.6f}, Validate MSE: {:.6f}, RMSE with real value: {:.2f}'.format(epoch-1, opts.maxEpoch, totalLoss/trainData.maxBatchNum/epochNumber4Valid, valCost[0].data[0], valCost[1]))
        	totalLoss=0
    saveModel(model, trainData, opts.savedModel)

def saveModel(model, trainData,savePath):
	#save model and feature parameters
    state={'model':model,
    	'featurePara':{
    	'targetGamma':trainData.targetGamma,
    	'subtotalGamma':trainData.subtotalGamma,
    	'minItemPriceGamma':trainData.minItemPriceGamma,
    	'maxItemPriceGamma':trainData.maxItemPriceGamma,
    	'availableDashesGamma':trainData.availableDashesGamma,
    	'totalOutstandingOrdersGamma':trainData.totalOutstandingOrdersGamma,
    	'lookUpCategory':trainData.lookUpCategory,
    	'lookUpProtocol':trainData.lookUpProtocol,
    	'lookUpTotalItems':trainData.lookUpTotalItems,
    	'lookUpDistinctItems':trainData.lookUpDistinctItems}}
    torch.save(state, savePath)
    	
def doValidation(inputData, targetData, targetOrig, model, criterion, trainData):
	model.eval()
	predict=model(inputData)
	mse_loss = criterion(predict, targetData)

	#reverse Gamma
	para=trainData.targetGamma

	model.train()
	
	#set the value <0 to 0 and >1 to 0.9999, so the value if meaning for probability
	def reverseCDF(data, gammaPara):
		fit_alpha,fit_loc,fit_beta=gammaPara[0],gammaPara[1],gammaPara[2]
		data=st.gamma.ppf(data, fit_alpha)* fit_beta +fit_loc
		return data
		
	predictNormalValue=reverseCDF(predict.data.clamp(0,0.9999).view(-1).numpy(),para)
	
	#RMSE of real value (predict - targetValidData)
	RMSE=np.sqrt(np.power(predictNormalValue-targetOrig.data.numpy(),2).mean())
	
	return (mse_loss, RMSE)

if __name__ == "__main__":
    main()
