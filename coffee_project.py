#%%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# get dummies
#coffee = pd.read_csv('merged_data_cleaned.csv')
#print(coffee.columns)

#coffee_Species = pd.get_dummies(coffee['Species'])
#print(coffee_Species)


coffee = pd.read_csv('merged_data_cleaned.csv')
print(coffee.columns)
df1 = pd.DataFrame(coffee['Acidity'])
df2 = pd.DataFrame(coffee['Body'])
df3 = pd.DataFrame(coffee['Balance'])
frames = [df1,df2,df3]
Merge = pd.concat(frames , axis = 'columns')
print(coffee['Acidity'].shape , coffee['Body'].shape , coffee['Balance'].shape)
Merge[Merge==np.inf]=np.nan
Merge.fillna(Merge.mean(), inplace=True)
print(Merge)
print(Merge.shape)
y = coffee['Species'].values
X = Merge.values
print(X.shape)
lb = LabelEncoder()
jim = lb.fit_transform(y)
print(jim)

X_train,X_test,Y_train,Y_test = train_test_split(X,jim,test_size = 0.2,random_state = 42)
Reg = LogisticRegression()
Reg.fit(X_train,Y_train)
pred = Reg.predict([[9,6,6]])
print(pred)
