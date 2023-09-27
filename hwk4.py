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

neighbors = 5

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

class MyKNN:
    def __init__(self, n_neighbors):
        """store n_neighbors as attribute"""
        self.n_neighbors = n_neighbors
    def fit(self, X, y):
        """store data."""
        self.train_features = X
        self.train_labels = y
    def decision_function(self, X):
        """compute vector of predicted scores.
        Larger values mean more likely to be in positive class."""
        test_nrow, test_ncol = X.shape
        list_of_scores = []
        for test_index in range(test_nrow):
            test_feature_vector = X.iloc[test_index,:].to_numpy().reshape(1,test_ncol)
            diff_mat = test_feature_vector-self.train_features
            square_diff_mat = diff_mat.to_numpy()**2
            distance_vec = square_diff_mat.sum(axis=1)
            nearest_indices = distance_vec.argsort()[:self.n_neighbors]
            nearest_labels = self.train_labels.iloc[nearest_indices]
            list_of_scores.append(nearest_labels.mean())
        return np.array(list_of_scores)
    def predict(self, X):
        """compute vector of predicted labels."""
        np.where(self.decision_function(X)>0.5,1,0)
        
    
class MyCV:
    def __init__(self, estimator, param_grid, cv):
        """estimator: learner instance
        param_grid: list of dictionaries
        cv: number of folds"""
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.best_params_ = {}
    def fit_one(self, param_dict, X, y):
        """Run self.estimator.fit on one parameter combination"""
        for param_name, param_value in param_dict.items():
            setattr(self.estimator,param_name,param_value)
        self.estimator.fit(X,y)
    def fit(self, X, y):
        """cross-validation for selecting the best dictionary in param_grid"""
        validation_df_list = []

        train_nrow, train_ncol = X.shape
        times_to_repeat = int(math.ceil(train_nrow/self.cv))
        fold_id_vec = np.tile(np.arange(self.cv), times_to_repeat)[:train_nrow]

        for validation_fold in range(self.cv):
            is_split = {
                "subtrain":fold_id_vec != validation_fold,
                "validation":fold_id_vec == validation_fold
            }
            split_data_dict = {}
            zip_obj = zip(["subtrain","validation"], is_split)
            for set_name, set_indices in zip_obj:
                split_data_dict[set_name] = (
                    X.iloc[set_indices,:],
                    y.iloc[set_indices,0])
                print("Set name: " + set_name)

            for param_dict in self.param_grid:
                self.fit_one(param_dict, *split_data_dict["subtrain"])
                X_valid, y_valid = split_data_dict["validation"]
                pred_valid = self.estimator.predict(X_valid)
                is_correct = pred_valid == y_valid
                validation_row = pd.DataFrame({
                    "validation_fold":[validation_fold],
                    "accuracy_percent":[is_correct.mean()],
                    "param_dict":[param_dict]
                }, index=[0])
                validation_df_list.append(validation_row)
        validation_df = pd.concat(validation_df_list)
        mean_valid_acc = validation_df.groupby("param_dict")["accuracy_percent"].mean()
        best_index = mean_valid_acc.argmax()
        mean_valid_acc.index[best_index]
        best_param_dict = "" #TODO
        self.fit_one(best_param_dict, X, y)
    def predict(self, X):
        self.estimator.predict(X)


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
                           LogisticRegressionCV(cv=neighbors, max_iter=1000))
      linear.fit(train_features, train_labels)

      most_freq_label = train_labels.value_counts().index[0]
      test_features, test_labels = split_data_dict["test"]
    
      #Custom nearest neighbors model
      cv_knn = MyCV(estimator=MyKNN(n_neighbors=neighbors), param_grid=[
          {"n_neighbors":n_neighbors} for n_neighbors in range(20)], 
          cv=neighbors)
      cv_knn.fit(train_features, train_labels)

      pred_dict = {
         "featureless":np.repeat(most_freq_label, len(test_labels)),
         "GridSearchCV + KNeighborsClassifier":clf.predict(test_features),
         "LogisticRegressionCV":linear.predict(test_features),
         "MyCV + MyKNN":cv_knn.predict(test_features)
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