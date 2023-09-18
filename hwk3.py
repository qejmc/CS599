import sklearn
import pandas as pd
import numpy as np
import plotnine as p9
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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

      #Grid search
      clf = GridSearchCV(KNeighborsClassifier(), 
                        {"n_neighbors":[k+1 for k in range(20)]})
      train_features, train_labels = split_data_dict["train"]
      clf.fit(train_features, train_labels)

      print("Best hyperparameter value (n_neighbors): " +\
            str(clf.best_params_["n_neighbors"]))
      
      #Linear model
      linear = make_pipeline(StandardScaler(), 
                           LogisticRegressionCV(cv=5, max_iter=1000))
      linear.fit(train_features, train_labels)

      most_freq_label = train_labels.value_counts().index[0]
      test_features, test_labels = split_data_dict["test"]

      pred_dict = {
         "featureless":np.repeat(most_freq_label, len(test_labels)),
         "nearest neighbors":clf.predict(test_features),
         "linear model":linear.predict(test_features)
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