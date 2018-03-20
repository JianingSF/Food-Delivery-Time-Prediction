import json
import csv
import datetime
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import random
import torch
from torch.autograd import Variable
import argparse
from model import mlpModel

class dataProcess():
    def __init__(self, dataPath, featurePara):
    	self.dataPath=dataPath
    	
    	#load feature para for prediction
    	self.targetGamma=featurePara['targetGamma']
		
    	self.lookUpCategory=featurePara['lookUpCategory']
    	self.lookUpProtocol=featurePara['lookUpProtocol']
    	self.lookUpTotalItems=featurePara['lookUpTotalItems']
    	self.lookUpDistinctItems=featurePara['lookUpDistinctItems']
    	
    	self.subtotalGamma=featurePara['subtotalGamma']
    	self.minItemPriceGamma=featurePara['minItemPriceGamma']
    	self.maxItemPriceGamma=featurePara['maxItemPriceGamma']
    	self.availableDashesGamma=featurePara['availableDashesGamma']
    	self.totalOutstandingOrdersGamma=featurePara['totalOutstandingOrdersGamma']
		    
    def normalizeByGamma(self, data, para):
    	#para: [fit_alpha, fit_loc, fit_beta]
    	fit_alpha=para[0]
    	fit_loc=para[1]
    	fit_beta=para[2]
    	data = (data  - fit_loc)/fit_beta
    	data=st.gamma.cdf(data, fit_alpha)
    	return data
		
    def getPredictionData(self):
		with open(self.dataPath,'rb') as fp:
			PredictionData = [ json.loads(x) for x in fp.readlines() if x.strip()!="" ]
		
		subtotal=[]
		min_item_price=[]
		max_item_price=[]
		
		availableDashes=[]
		totalOutstandingOrders=[]
		
		storeCategory=[]
		orderProtocol=[]
		totalItemsCategory=[]
		distinctItemsCategory=[]
		
		maxTotalItems=29
		maxDistinctItems=19
		
		targetPlus=[]
		deliveryID=[]
		#Unit test :'created_at') != 'NA'
		
		timeFormat='%Y-%m-%d %H:%M:%S'
		totalRecords=len(PredictionData)
	
		for i in range(totalRecords):	
			#process order features
			subtotal.append(int(PredictionData[i]['subtotal']))
			min_item_price.append(int(PredictionData[i]['min_item_price']))
			max_item_price.append(int(PredictionData[i]['max_item_price']))
			deliveryID.append(PredictionData[i]['delivery_id'])
			
			#process store features
			if PredictionData[i]['store_primary_category'] in self.lookUpCategory:
				storeCategory.append(self.lookUpCategory[PredictionData[i]['store_primary_category']])
			else:
				storeCategory.append(self.lookUpCategory['NA'])
			
			if PredictionData[i]['order_protocol'] in self.lookUpProtocol:
				orderProtocol.append(self.lookUpProtocol[PredictionData[i]['order_protocol']])
			else:
				orderProtocol.append(self.lookUpProtocol['NA'])
			
			#total_items and num_distinct_items from lookup table
			if PredictionData[i]['total_items'] in self.lookUpTotalItems:
				totalItemsCategory.append(self.lookUpTotalItems[PredictionData[i]['total_items']])
			else:
				totalItemsCategory.append(maxTotalItems)
			
			if PredictionData[i]['num_distinct_items'] in self.lookUpDistinctItems:
				distinctItemsCategory.append(self.lookUpDistinctItems[PredictionData[i]['num_distinct_items']])
			else:
				distinctItemsCategory.append(maxDistinctItems)
	
			if PredictionData[i]['total_onshift_dashers']!='NA' and PredictionData[i]['total_busy_dashers']!='NA':
				if int(PredictionData[i]['total_onshift_dashers'])-int(PredictionData[i]['total_busy_dashers'])<0:
					availableDashes.append(0)
				else:
					availableDashes.append((int(PredictionData[i]['total_onshift_dashers'])-int(PredictionData[i]['total_busy_dashers'])))
			else:
				availableDashes.append(0) 
			
			if PredictionData[i]['total_outstanding_orders']!='NA':
				totalOutstandingOrders.append(int(PredictionData[i]['total_outstanding_orders']))
			else:
				totalOutstandingOrders.append(0)
			
			#targetPlus: time4StoreToConsumer+time4OrderPlace, which will be added into the prediction time for total time
			time4StoreToConsumer=int(PredictionData[i]['estimated_store_to_consumer_driving_duration']) if PredictionData[i]['estimated_store_to_consumer_driving_duration']!='NA' else 0
			time4OrderPlace=int(PredictionData[i]['estimated_order_place_duration']) if PredictionData[i]['estimated_order_place_duration']!='NA' else 0
			targetPlus.append(time4StoreToConsumer+time4OrderPlace)
		
		#Normalization based on Gamma para
		subtotal=self.normalizeByGamma(subtotal,self.subtotalGamma)
		min_item_price=self.normalizeByGamma(min_item_price, self.minItemPriceGamma)
		max_item_price=self.normalizeByGamma(max_item_price, self.maxItemPriceGamma)
	
		availableDashes=self.normalizeByGamma(availableDashes, self.availableDashesGamma)
		totalOutstandingOrders=self.normalizeByGamma(totalOutstandingOrders, self.totalOutstandingOrdersGamma)
		
		#convert to Tensor
		inputOrder=torch.from_numpy(np.array(zip(subtotal,min_item_price,max_item_price,availableDashes,totalOutstandingOrders),dtype=np.float32))
		inputStoreCategory=torch.from_numpy(np.array(storeCategory,dtype=np.int64))
		inputProtocol=torch.from_numpy(np.array(orderProtocol,dtype=np.int64))
		inputTotalItems=torch.from_numpy(np.array(totalItemsCategory,dtype=np.int64))
		inputDistinctItems=torch.from_numpy(np.array(distinctItemsCategory,dtype=np.int64))
		
		#convert to Variable		
		input=[Variable(inputOrder, requires_grad=False),
			Variable(inputStoreCategory, requires_grad=False),
			Variable(inputProtocol, requires_grad=False), 
			Variable(inputTotalItems, requires_grad=False), 
			Variable(inputDistinctItems, requires_grad=False)]
		#targetPlus=Variable(torch.from_numpy(np.array(time4StoreToConsumer+time4OrderPlace,dtype=np.int64)), requires_grad=False)
		#targetPlus=time4StoreToConsumer+time4OrderPlace
		
		return input,targetPlus, deliveryID

def parseCmd():
    parser = argparse.ArgumentParser(description='Predictor for delivery time estimation')
    parser.add_argument("--predictData", type=str, default="data_to_predict.json", help="Prediction data json file") 
    parser.add_argument("--savedModel", type=str,  default="savedModel.net", help="Saved model")
    parser.add_argument("--predictionResult", type=str, default="predictionResult.tsv", help="output result to outResult file")
    return parser.parse_args()

def main():
    opts = parseCmd()
    
    #load model, gamma para, lookup table
    savedModel=torch.load(opts.savedModel)
    model=savedModel['model']
    featurePara=savedModel['featurePara']
    
    predictionData = dataProcess(opts.predictData, featurePara)
    
    #process prediction data, then run saved model and get prediction
    input,targetPlus, deliveryID=predictionData.getPredictionData()
    model.eval()
    predict=model(input)
    
    #reverse Gamma to get real target value
    para=predictionData.targetGamma
    predictNormalValue=st.gamma.ppf(predict.data.numpy(), para[0])*para[2]+para[1]
    
    #totalTime: predicted value + argetPlus from other prediction systems
    totalTime=[]
    for i in range (len(targetPlus)):
    	totalTime.append(int(predictNormalValue[i][0])+targetPlus[i])

    assert(len(deliveryID) == len(totalTime))

    with open(opts.predictionResult, 'wb') as fp:
        fp.write('"delivery_id"\t"predicted_delivery_seconds"\n')
        fp.write("\n".join( ["%s\t%s"%(deliveryID[i], totalTime[i]) for i in range(len(totalTime)) ] ))
        fp.write("\n")
	
if __name__ == "__main__":
    main()
		

	

