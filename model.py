

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cufflinks as cf
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')





emp_data = pd.read_csv("employee-salary-data.csv")

emp_data.shape



"""## Feature Selection

"""

from sklearn.ensemble import ExtraTreesClassifier

#Feature selection
X = emp_data.drop(['yearly_increment'],axis=1)
X = X.select_dtypes(exclude=[object])
Y = emp_data['yearly_increment'].values

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()



X = emp_data.drop(['hired', 'service_year', 'nvq_level', 'length_of_service', 'employee_status','customer_satisfaction_level', 'monthly_basic_salary', 'yearly_increment'], axis=1).values
y = emp_data['yearly_increment'].values

# split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)

print(X.shape, X_train.shape, X_test.shape)



"""## Fitting Models

"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)



from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  #for classification report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score



svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
print(accuracy_score(y_test, pred_svc))

svc_cm1 = confusion_matrix(y_test, pred_svc)
print(svc_cm1)

sns.heatmap(svc_cm1, annot=True,cmap='BuPu', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] )
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.title('Confusion Matrix for Support Vector Machine\n')
plt.show()



param_dist = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

svc_hyper = RandomizedSearchCV(svc, param_distributions=param_dist, verbose=2, cv=5, random_state=42, n_iter=10, scoring='accuracy', n_jobs=-1)
svc_hyper.fit(X_train, y_train)

svc_hyper.best_params_

svc_hyper.best_score_

svc_classifier = SVC(C=1000, gamma=0.01, kernel='rbf')
svc_classifier.fit(X_train, y_train)
y_pred_svc= svc_classifier.predict(X_test)



svc_cm = confusion_matrix(y_test, y_pred_svc)
print(svc_cm)

sns.heatmap(svc_cm, annot=True,cmap='Blues', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] )
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.title('Confusion Matrix for Support Vector Machine\n')
plt.show()


import pickle
#save model
pickle.dump(svc_classifier, open('model.pkl', 'wb'))

# load model
employee_model = pickle.load(open('model.pkl', 'rb'))

#Predict the output

print(employee_model.predict([[2,280,40]]))





