import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import  csv
import pandas as pd
import joblib




##############load csv as numpy
train = np.loadtxt("train.csv",delimiter=",", dtype=str) 
test = np.loadtxt("test.csv",delimiter=",", dtype=str)


##########################

a = 15
X = train[:, :a] ##### train data without "last" element
y=train[:,a] ######### the last element: underclocking as result
######split dataset into train and test training size:0.9 test size:0.1 random state:100
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=100)
forest = RandomForestClassifier(n_estimators=100, random_state=100)###n_estimator:divide into 100 subtree and merge,random state:100
forest.fit(X_train,y_train)


id = test[:,0]####id
predictions = forest.predict(test) ########predicted underclocking

result = np.transpose([id,predictions]) #####merge id and predictions  
[print(result)] #######print result
df = pd.DataFrame(result, columns =['id','underclocking'])#####convert numpy into Panda.Dataframe

submisssion = df.to_csv("submission.csv",index = False) ###write submission.csv
joblib.dump(forest, "./random_forest.joblib")### saving trained model for someday to load 

# with open("result.csv", 'w') as csvfile: 
#         csvwriter = csv.writer(csvfile) 
#         csvwriter.writerow("result") 
#         csvwriter.writerows(predictions)
    
