from __future__ import division

import json
import csv
import datetime
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import random
import torch
from torch.autograd import Variable

class dataManager():
    def __init__(self, dataPath, batchSize, src_data_manager=None, do_shuffle=True):
    	
    	self.dataPath=dataPath
    	self.batchSize=batchSize
  	
    	self.input=torch.FloatTensor()
    	self.inputStoreCategory=torch.LongTensor()
    	self.inputProtocol=torch.LongTensor()
    	self.inputTotalItems=torch.LongTensor()
    	self.inputDistinctItems=torch.LongTensor()
    	
    	self.target=torch.FloatTensor()
    	self.targetOrig=torch.FloatTensor()
        self.do_shuffle=do_shuffle
    	
    	self.batchPos=0
    	self.maxBatchNum=0
    	self.batchIndex=[]
    	
        if src_data_manager is None:
    	    self.lookUpCategory={} 
    	    self.lookUpProtocol={} 
    	    self.lookUpTotalItems={} 
    	    self.lookUpDistinctItems={} 
    	    
    	    #store Gamma para for prediction
    	    self.targetGamma=[] 
    	    
    	    self.subtotalGamma=[] 
    	    self.minItemPriceGamma=[] 
    	    self.maxItemPriceGamma=[] 
    	    self.availableDashesGamma=[] 
    	    self.totalOutstandingOrdersGamma=[] 
        else:
    	    self.lookUpCategory=src_data_manager.lookUpCategory
    	    self.lookUpProtocol=src_data_manager.lookUpProtocol
    	    self.lookUpTotalItems=src_data_manager.lookUpTotalItems
    	    self.lookUpDistinctItems=src_data_manager.lookUpDistinctItems
    	    
    	    #store Gamma para for prediction
    	    self.targetGamma=src_data_manager.targetGamma
    	    
    	    self.subtotalGamma=src_data_manager.subtotalGamma
    	    self.minItemPriceGamma=src_data_manager.minItemPriceGamma
    	    self.maxItemPriceGamma=src_data_manager.maxItemPriceGamma
    	    self.availableDashesGamma=src_data_manager.availableDashesGamma
    	    self.totalOutstandingOrdersGamma=src_data_manager.totalOutstandingOrdersGamma
    	        	
    	self.getData(dataPath)

    def isFinish(self):
    	if self.batchPos >= self.maxBatchNum:
    		return True
    	return False
    
    def shuffle(self):
        if self.do_shuffle:
    	    random.shuffle(self.batchIndex)

    def getValidationData(self):
    	input_data=[Variable(self.input, requires_grad=False),
    			Variable(self.inputStoreCategory, requires_grad=False), 
    			Variable(self.inputProtocol, requires_grad=False),
    			Variable(self.inputTotalItems, requires_grad=False),
    			Variable(self.inputDistinctItems, requires_grad=False)]
    	
    	target=Variable(self.target,requires_grad=False)
    	targetOrig=Variable(self.targetOrig,requires_grad=False)
        return input_data, target, targetOrig
    	
    def nextBatch(self):
    	if self.isFinish():
    		self.batchPos=0
    		self.shuffle()
    	
    	input=self.input.narrow(0,self.batchIndex[self.batchPos]*self.batchSize, self.batchSize)
    	inputStoreCategory=self.inputStoreCategory.narrow(0,self.batchIndex[self.batchPos]*self.batchSize, self.batchSize)
    	inputProtocol=self.inputProtocol.narrow(0,self.batchIndex[self.batchPos]*self.batchSize, self.batchSize)
    	inputTotalItems=self.inputTotalItems.narrow(0,self.batchIndex[self.batchPos]*self.batchSize, self.batchSize)
    	inputDistinctItems=self.inputDistinctItems.narrow(0,self.batchIndex[self.batchPos]*self.batchSize, self.batchSize)
    			
    	input_data=[Variable(input, requires_grad=False),
    			Variable(inputStoreCategory, requires_grad=False), 
    			Variable(inputProtocol, requires_grad=False),
    			Variable(inputTotalItems, requires_grad=False),
    			Variable(inputDistinctItems, requires_grad=False)]
    	
    	target=Variable(self.target.narrow(0,self.batchIndex[self.batchPos]*self.batchSize, self.batchSize),requires_grad=False)
    	targetOrig=Variable(self.targetOrig.narrow(0,self.batchIndex[self.batchPos]*self.batchSize, self.batchSize),requires_grad=False)
    	
    	self.batchPos+=1
    
    	return input_data, target, targetOrig
    
    def fitGamma(self, data, gamma):
    	if len(gamma) !=0:
    		fit_alpha = gamma[0]
    		fit_loc = gamma[1]
    		fit_beta = gamma[2]
        else:
        	fit_alpha, fit_loc, fit_beta=st.gamma.fit(data) 
        	
        data = (data  - fit_loc)/fit_beta
        data=st.gamma.cdf(data, fit_alpha)
        return data, [fit_alpha, fit_loc, fit_beta]
		
    def getData(self, dataPath):
		data=[]
		target=[]
		
		subtotal=[]
		min_item_price=[]
		max_item_price=[]
		availableDashes=[]
		totalOutstandingOrders=[]
		
		storeCategory=[]
		orderProtocol=[]
		totalItemsCategory=[]
		distinctItemsCategory=[]
		
		#read data from .csv
		reader = csv.DictReader(open(dataPath, 'rb'))
		for line in reader:
			data.append(line)
		
		#remove records when created_at or actual_delivery_time is NA
		data = [d for d in data if d.get('created_at') != 'NA' and d.get('actual_delivery_time') != 'NA']
			
		timeFormat='%Y-%m-%d %H:%M:%S'
		totalRecords=len(data)
		
		#use to define the total category
		maxTotalItems=29
		maxDistinctItems=19
		#build category lookup table for training data
		if len(self.lookUpCategory)==0:
			curIndex=0
			for i in range(totalRecords):
				if data[i]['store_primary_category'] not in self.lookUpCategory:
					self.lookUpCategory[data[i]['store_primary_category']]=curIndex
					curIndex+=1
			curIndex=0
			for i in range(totalRecords):
				if data[i]['order_protocol'] not in self.lookUpProtocol:
					self.lookUpProtocol[data[i]['order_protocol']]=curIndex
					curIndex+=1
					
			#predefine category for TotalItems and DistinctItems	
			for i in range (1,maxTotalItems+1):
				self.lookUpTotalItems[str(i)]=i
			self.lookUpTotalItems['NA']=0
		
			for i in range (1,maxDistinctItems+1):
				self.lookUpDistinctItems[str(i)]=i
			self.lookUpDistinctItems['NA']=0

		for i in range(totalRecords):
			#Time features:
			start = datetime.datetime.strptime(data[i]['created_at'], timeFormat)
			end = datetime.datetime.strptime(data[i]['actual_delivery_time'], timeFormat)
			
			time4StoreToConsumer=int(data[i]['estimated_store_to_consumer_driving_duration']) if data[i]['estimated_store_to_consumer_driving_duration']!='NA'  else 0
			time4OrderPlace=int(data[i]['estimated_order_place_duration']) if data[i]['estimated_order_place_duration']!='NA' else 0
			
			#Calculate target in seconds
			target.append((end-start).seconds-time4StoreToConsumer-time4OrderPlace)
	
			#Order features
			subtotal.append(int(data[i]['subtotal']))
			min_item_price.append(int(data[i]['min_item_price']))
			max_item_price.append(int(data[i]['max_item_price']))
			
			#Market features
			if data[i]['total_onshift_dashers']!='NA' and data[i]['total_busy_dashers']!='NA':
				if int(data[i]['total_onshift_dashers'])-int(data[i]['total_busy_dashers'])<0:
					availableDashes.append(0)
				else:
					availableDashes.append((int(data[i]['total_onshift_dashers'])-int(data[i]['total_busy_dashers'])))
			else:
				availableDashes.append(0)
			
			if data[i]['total_outstanding_orders']!='NA':
				totalOutstandingOrders.append(int(data[i]['total_outstanding_orders']))
			else:
				totalOutstandingOrders.append(0)
			
			#Store features from lookup table
			if data[i]['store_primary_category'] in self.lookUpCategory:
				storeCategory.append(self.lookUpCategory[data[i]['store_primary_category']])
			else:
				storeCategory.append(self.lookUpCategory['NA'])
				
			orderProtocol.append(self.lookUpProtocol[data[i]['order_protocol']])
			
			#Order features: total_items and num_distinct_items from lookup table
			if data[i]['total_items'] in self.lookUpTotalItems:
				totalItemsCategory.append(self.lookUpTotalItems[data[i]['total_items']])
			else:
				totalItemsCategory.append(maxTotalItems)
			
			if data[i]['num_distinct_items'] in self.lookUpDistinctItems:
				distinctItemsCategory.append(self.lookUpDistinctItems[data[i]['num_distinct_items']])
			else:
				distinctItemsCategory.append(maxDistinctItems)
		
		#Normalization for Gamma distribution: for training, calculate gamma para then calculate CDF, for validation, use exited gamma para to calculate CDF
		subtotal, self.subtotalGamma=self.fitGamma(subtotal, self.subtotalGamma)
		min_item_price, self.minItemPriceGamma=self.fitGamma(min_item_price, self.minItemPriceGamma)
		max_item_price, self.maxItemPriceGamma=self.fitGamma(max_item_price, self.maxItemPriceGamma)
		availableDashes, self.availableDashesGamma=self.fitGamma(availableDashes, self.availableDashesGamma)
		totalOutstandingOrders, self.totalOutstandingOrdersGamma=self.fitGamma(totalOutstandingOrders, self.totalOutstandingOrdersGamma)
		
		targetOrig=torch.from_numpy(np.array(target,dtype=np.float32))
		target, self.targetGamma=self.fitGamma(target, self.targetGamma)
		
		self.input=torch.from_numpy(np.array(zip(subtotal,min_item_price,max_item_price,availableDashes,totalOutstandingOrders),dtype=np.float32))
		self.inputStoreCategory=torch.from_numpy(np.array(storeCategory,dtype=np.int64))
		self.inputProtocol=torch.from_numpy(np.array(orderProtocol,dtype=np.int64))
		self.inputTotalItems=torch.from_numpy(np.array(totalItemsCategory,dtype=np.int64))
		self.inputDistinctItems=torch.from_numpy(np.array(distinctItemsCategory,dtype=np.int64))
    	
		self.target=torch.from_numpy(np.array(target,dtype=np.float32))
		self.targetOrig=torch.from_numpy(np.array(targetOrig,dtype=np.float32))
			
		#Batch related
		self.maxBatchNum=int(len(self.input)/self.batchSize)
		self.batchIndex = [ i for i in range(self.maxBatchNum)]