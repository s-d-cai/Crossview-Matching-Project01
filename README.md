###### # Ground-to-Aerial Image Geo-Localization Experiments
###### # Siam-FCANet18 and Siam-FCANet34
###### # Files for experiments on CVUSA dataset
###### # Python 2.7 (3.6 and 3.7 are also working) and Pytorch 0.4.0 or later versions
###### # Link to trained models (and some generated feature vectors): 
###### # Link to download CVUSA dataset: 
###### # https://www.dropbox.com/sh/yi3kkygzbw7hd2o/AABIQfnvi7UVjCTDf0jjHFsJa?dl=0
###### or https://pan.baidu.com/s/1ZRv-912C7SMaK7IcR8iGhA (code: x1g9)

###### # For training on CVUSA, Please directly run "train_CVUSA_01.py" with corrected paths to dataset and the model. Models trained by VH dataset are recommended to be used as initialization. 
###### # For evaluation, two ways are both working: (1) run "train_CVUSA_01.py" with setting (epoch>-1) in the "### ranking test" sub-section, (2) run "Feature_vectors_generation.py" to get the features and then run "RankingTest.py" or "RankingTest_ForBigMat.py" to output the results at corresponding metrics (the parameter "length" is used to control the metrics, i.e., recall at top-k or top k%)
###### # Please note that: when applying HER-loss to train un-normalized models (i.e., embedding features are not normalized), sometimes the trainig process may crash with "NaN" occurs in the very beginning, because values of some variables in the loss function are getting too big under the current initializations of some learning layers or the learning rate is too big. If this situation happens, please re-run the "train_CVUSA_01.py". BTW, the new version of HER-loss is robust to this case and it will be uploaded later...

###### #Evaluation on CVUSA 
###### Recall@ Top 1%: 98.457% (Net18), 98.345% (Net34 old model)
###### Recall@ Top-1:  49.459% (Net18), 42.863% (Net34 old model)
