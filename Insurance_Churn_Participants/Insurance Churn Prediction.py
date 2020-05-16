
import pandas as pd
df_train=pd.read_csv("Train.csv")
df_test=pd.read_csv("Test.csv")


#import
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#preprocessing
scaler=StandardScaler()
X=df_train.drop('labels', axis=1)
y=df_train['labels']

#training
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=.35, stratify=y, random_state=42)
scaler.fit_transform(X_train)
scaler.transform(X_test)
rfc=RandomForestClassifier()
param = { 'max_depth' : [2, 4, 6, 8, 10, 12],
           'n_estimators' : [50, 100, 150, 200, 250]
         }
gcv=GridSearchCV(rfc, param_grid = param, cv=5)
gcv.fit(X_train, y_train)

#prediction and accuracy
y_pred=gcv.best_estimator_.predict(X_test)
print(accuracy_score(y_test, y_pred))


print(gcv.score(X_train, y_train))
print(gcv.score(X_test, y_test))


sub=pd.DataFrame(gcv.best_estimator_.predict(df_test))


sub.to_excel(r"Submission.xlsx", index=False)






