import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
#importing libraries


#%%
df=pd.read_csv("voice.csv")
#importing dataset

#%% count of male and female
sns.countplot(y=df.label ,data=df)
plt.xlabel("count")
plt.ylabel("gender")
plt.show()

#%% count of null values
df.isnull().sum(axis = 0)

#%%
df.label=[1 if i=="male" else 0 for i in df.label]
#male=1 and female=0
x=df.drop(["label"],axis=1).values 
y=df["label"].values.reshape(-1,1)



#%%
x_train,x_test,y_train,y_test=train_test_split(
        x,y,random_state=42,test_size=0.20)
#train and test split
#%% Scaling features
sc=StandardScaler()
sc.fit(x_train)
x_train=sc.transform(x_train)
x_test=sc.transform(x_test)

#%% finding most accurate n_neighbors value
scores=[]
for i in range(1,15):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    scores.append(knn.score(x_test,y_test))
#%% plot n_neighbors accuracy scores
plt.plot(range(1,15),scores) 
plt.xlabel("k values")
plt.ylabel("knn scores(accuracy)")    
plt.show()
#%% The best result is captured at k = 7
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print("{} nn score: {}".format(
        7,knn.score(x_test,y_test)))




#%%