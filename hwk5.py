import sklearn
import math
import pandas as pd
import numpy as np
import plotnine as p9
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Datasets spam and zip
zip_df = pd.read_csv("zip.test.gz",sep=" ",header=None)
spam_df = pd.read_csv("spam.data",sep=" ",header=None)

zip_label_col_num = 0
zip_label_vec = zip_df.iloc[:,zip_label_col_num]

# Remove non-binary entires from zip df
is_01 = zip_label_vec.isin([0,1])
zip_df = zip_df.loc[is_01,:]

is_label_col = zip_df.columns == zip_label_col_num
zip_features = zip_df.iloc[:,~is_label_col]
zip_labels = zip_df.iloc[:,is_label_col]

spam_label_col_num=57
is_spam_label_col = spam_df.columns == spam_label_col_num
spam_features = spam_df.iloc[:,~is_spam_label_col]
spam_labels = spam_df.iloc[:,is_spam_label_col]

data_dict = {
    "zip" : (zip_features, zip_labels),
    "spam" : (spam_features, spam_labels)
}

list_of_accuracy_rows = []

# Refactor spam set to be scaled
'''
spam_features, spam_labels = data_dict.pop("spam")
n_spam_rows, n_spam_features = spam_features.shape
spam_mean = spam_features.mean().to_numpy().reshape(1, n_spam_features)
spam_stf = spam_features.std().to_numpy().reshape(1,n_spam_features)
spam_scaled = (spam_features-spam_mean)/spam_stf
data_dict["spam_scaled"] = (spam_scaled, spam_labels)
'''

# MyLogReg class
class MyLogReg:
   def __init__(self, max_iterations=2000, step_size=0.0001, 
                validation_features=None, validation_labels=None):
      self.max_iterations = max_iterations
      self.step_size = step_size
      self.validation_features = validation_features
      self.validation_labels = validation_labels

   def fit(self, X, y):
      self.subtrain_features = X
      self.subtrain_labels = y
      
      scores_df_list = []

      data_nrow, data_ncol = X.shape
      weight_vec = np.repeat(0.0, data_ncol).reshape(data_ncol,1)
      data_mat = X.to_numpy()

      if(self.validation_features is not None):
         self.validation_labels = self.validation_labels.to_numpy().reshape(-1,1)
         val_nrow, val_ncol = self.validation_features.shape
         val_weight_vec = np.repeat(0.0, val_ncol).reshape(val_ncol,1)
         val_mat = self.validation_features.to_numpy()

      y = y.to_numpy().reshape(-1,1)

      for iteration in range(self.max_iterations):
         pred_vec = np.matmul(data_mat, weight_vec).reshape(data_nrow,1)
         label_pos_neg_vec = np.where(y==1,1,-1)
         grad_loss_wrt_pred=-label_pos_neg_vec/(1+np.exp(label_pos_neg_vec*pred_vec))
         loss_vec = np.log(1+np.exp(-label_pos_neg_vec*pred_vec))
         
         # Compute validation loss if validation matrix is stored
         if(self.validation_features is not None):
            val_pred_vec = np.matmul(val_mat, weight_vec).reshape(-1, 1)
            val_label_pos_neg_vec = np.where(self.validation_labels == 1, 1, -1)
            val_loss_vec = np.log(1 + np.exp(-val_label_pos_neg_vec * val_pred_vec))

            scores_row = pd.DataFrame({
               "iteration":[iteration],
               "set_name":"validation",
               "loss_value":[val_loss_vec.mean()]
            }, index=[0])
            scores_df_list.append(scores_row)

         scores_row = pd.DataFrame({
            "iteration":[iteration],
            "set_name":"subtrain",
            "loss_value":[loss_vec.mean()]
         }, index=[0])
         scores_df_list.append(scores_row)

         grad_loss_wrt_weight = np.matmul(data_mat.T, grad_loss_wrt_pred)
         weight_vec -= self.step_size * grad_loss_wrt_weight
      
      scores_df = pd.concat(scores_df_list)

      self.coef_ = weight_vec
      self.intercept_ = pred_vec

      return scores_df

   def decision_function(self, X):
      return np.matmul(X, self.coef_) + self.intercept_

   def predict(self, X):
      scores_vec = self.decision_function(X)
      predictions_vec = np.where(scores_vec > 0, 1, 0)
      return predictions_vec


class MyLogRegCV:
   def __init__(self, cv):
      self.cv = cv

   def fit(self, X, y):
      kf = KFold(n_splits=self.cv, shuffle=True, random_state=1)
      enum_obj = enumerate(kf.split(X))

      for fold_number, index_tuple in enum_obj:
         
         zip_obj = zip(["train","test"], index_tuple)
         split_data_dict = {}

         for set_name, set_indices in zip_obj:
            split_data_dict[set_name] = (
                  X.iloc[set_indices,:],
                  y.iloc[set_indices,0])
         
      subtrain_features, subtrain_labels = split_data_dict["train"]
      test_features, test_labels = split_data_dict["test"]

      log_reg = MyLogReg(validation_features=test_features, 
                         validation_labels=test_labels)
      self.scores_ = log_reg.fit(subtrain_features, subtrain_labels)

      print(self.scores_)

      validation_df = self.scores_[self.scores_["set_name"] == "validation"]
      min_row = validation_df[validation_df["loss_value"] == 
                              validation_df["loss_value"].min()]
      self.best_iterations = min_row["iteration"].values[0]

      self.lr = MyLogReg(max_iterations=self.best_iterations)
      self.lr.fit(subtrain_features, subtrain_labels)
   
   def decision_function(self, X):
      return self.lr.decision_function(X)
   
   def predict(self, X):
      return self.lr.predict(X)


test = MyLogRegCV(cv=2)
test.fit(zip_features,zip_labels)
#test.predict(zip_features)


for data_name, (data_features, data_labels) in data_dict.items():
    
   kf = KFold(n_splits=3, shuffle=True, random_state=1)
   print("Data set: " + data_name)
   enum_obj = enumerate(kf.split(data_features))

   for fold_number, index_tuple in enum_obj:
      
      print("Fold number: " + str(fold_number))
      zip_obj = zip(["train","test"], index_tuple)
      split_data_dict = {}

      for set_name, set_indices in zip_obj:
         split_data_dict[set_name] = (
               data_features.iloc[set_indices,:],
               data_labels.iloc[set_indices,0])
         print("Set name: " + set_name)
      
      train_features, train_labels = split_data_dict["train"]

      #Grid search
      clf = GridSearchCV(KNeighborsClassifier(), 
                        {"n_neighbors":[k+1 for k in range(20)]})
      clf.fit(train_features, train_labels)

      print("Best hyperparameter value (n_neighbors): " +\
            str(clf.best_params_["n_neighbors"]))
      
      #Linear model
      linear = make_pipeline(StandardScaler(), 
                           LogisticRegressionCV(cv=2, max_iter=1000))
      linear.fit(train_features, train_labels)

      most_freq_label = train_labels.value_counts().index[0]
      test_features, test_labels = split_data_dict["test"]
    
      #Custom linear model (gradient descent)
      my_log_reg = MyLogRegCV(cv = 2)
      my_log_reg.fit(zip_features, zip_labels)

      pred_dict = {
         "featureless":np.repeat(most_freq_label, len(test_labels)),
         "GridSearchCV + KNeighborsClassifier":clf.predict(test_features),
         "LogisticRegressionCV":linear.predict(test_features),
         "MyLogRegCV":my_log_reg.predict(test_features)
      }

      for algorithm, pred_vec in pred_dict.items():
         print("Algorithm: " + algorithm)
         is_correct = pred_vec == test_labels
         list_of_accuracy_rows.append(pd.DataFrame({
               "test_accuracy_percent":is_correct.mean(),
               "algorithm":[algorithm],
               "fold_number":[fold_number],
               "data_set":data_name
         }))

output_data = pd.concat(list_of_accuracy_rows)
print(output_data)

output_plot = p9.ggplot()+\
   p9.geom_point(
      p9.aes(
            x="test_accuracy_percent",
            y="algorithm",
      ),
      data = output_data
   )+\
   p9.facet_grid(". ~ data_set")

print(output_plot)