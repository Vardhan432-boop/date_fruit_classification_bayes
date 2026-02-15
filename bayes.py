import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,recall_score,classification_report,confusion_matrix
df=pd.read_csv("Date_Fruit_Datasets.csv")
print(df.shape)
X=df.drop('Class',axis=1)
X=X.select_dtypes(include='number')
y=df['Class']
X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=0.4,random_state=42,stratify=y)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp)
X_train=np.array(X_train)
X_val=np.array(X_val)
X_test=np.array(X_test)
classes=np.unique(y_train)
means={}
covs={}
priors={}
for c in classes:
    x_c=X_train[y_train==c]
    means[c]=np.mean(x_c,axis=0)
    covs[c]=np.cov(x_c,rowvar=False)
    priors[c]=len(x_c)/len(X_train)
def predict(X):
    preds=[]
    for x in X:
        scores=[]
        for c in classes:
            likelihood=multivariate_normal(
                mean=means[c],
                cov=covs[c],
                allow_singular=True
            ).pdf(x)
            posterior=likelihood*priors[c]
            scores.append(posterior)
        preds.append(classes[np.argmax(scores)])
    return preds
y_pred_val=predict(X_val)
print(classification_report(y_val,y_pred_val))
y_pred_test=predict(X_test)
print(classification_report(y_test,y_pred_test))
matrix=confusion_matrix(y_test,y_pred_test)
plt.figure(figsize=(12, 10))
sns.heatmap(matrix, cmap='coolwarm', center=0)
plt.title("confusion matrix Heatmap")
plt.show()
model=GaussianNB()
model.fit(X_train,y_train)
print(classification_report(y_test,model.predict(X_test)))