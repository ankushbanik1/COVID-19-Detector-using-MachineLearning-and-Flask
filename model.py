import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn.linear_model import LinearRegression,LogisticRegression
df=pd.read_csv('data.csv')
def data_split(data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)* ratio)
    test_indecis=shuffled[:test_set_size]
    train_indecis=shuffled[test_set_size:]
    
    return data.iloc[train_indecis],data.iloc[test_indecis]
train,test=data_split(df,0.2)
x_train=train[['fever','bodypain','age','runnyNose','diffbreath','titeFlue']].to_numpy()
x_test=test[['fever','bodypain','age','runnyNose','diffbreath','titeFlue']].to_numpy()
y_train=train[['infectionProb']].to_numpy().reshape(2000,)
y_test=test[['infectionProb']].to_numpy().reshape(499,)
clf=LogisticRegression()
clf.fit(x_train,y_train)
input_feature=[107,1,43,1,1,1]
infprob=clf.predict_proba([input_feature])[0][1]    

file=open('my_model.pkl','wb')
pickle.dump(clf,file)