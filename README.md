# DNN-regression-for-total-delivery-time-estimation
The target problem can be defined as a regression problem for prediction based on supervised learning. This work uses Python and PyTorch to build linear/polynomial regression and DNN regression models to predict total delivery duration seconds. The **ModelDesign.pdf** includes details in Data Pre-processing, Feature Extraction & Normalization, Model design, Experiments,Conclusion and Future work, which can be used together with submitted code files.

* dataManager.py:  For Data Pre-processing, Feature Extraction & Normalization
* model.py: For model design
* buildModel.py: Run experiments

**Part1: Build Models**

* Run dataPrepare.sh to re-split historical_data.csv into train_data.csv and validation_data.csv. (You will get different split each time considering data shuffle during split)
* Run dataManager.py to build models, the new model will be saved after training for prediction.
      
**Part2: Prediction Output**

Three input parameters of prediction.py:

  * predictionData: .json file to predict
  * savedModel: saved model and normalization parameters from the part one
  * predictionResult: .tsv file of prediction output
  
Output will be saved with columns of delivery_id and  predicted_delivery_seconds.


