import sklearn
import math
import pandas as pd
import numpy as np
import plotnine as p9
import torch
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
spam_features, spam_labels = data_dict.pop("spam")
n_spam_rows, n_spam_features = spam_features.shape
spam_mean = spam_features.mean().to_numpy().reshape(1, n_spam_features)
spam_stf = spam_features.std().to_numpy().reshape(1,n_spam_features)
spam_scaled = (spam_features-spam_mean)/spam_stf
data_dict["spam_scaled"] = (spam_scaled, spam_labels)


class CSV(torch.utils.data.Dataset):
   def __init__(self, features, labels):
      self.features = features
      self.labels = labels
   
   def __getitem__(self, item):
      return self.features[item,:], self.labels[item]
   
   def __len__(self):
      return len(self.labels)
   

class TorchModel(torch.nn.Module):
   def __init__(self, units_per_layer):
      super(TorchModel, self).__init__()
      seq_args = []
      second_to_last = len(units_per_layer) - 1
      for layer_i in range(second_to_last):
         next_i = layer_i + 1
         layer_units = units_per_layer[layer_i]
         next_units = units_per_layer[next_i]
         seq_args.append(torch.nn.Linear(layer_units, next_units))

         if layer_i < second_to_last:
            seq_args.append(torch.nn.ReLU())
      self.stack = torch.nn.Sequential(*seq_args)
   def forward(self, features):
      return self.stack(features)


class TorchLearner:
   def __init__(self, hyper_params, units_per_layer, val_features=None, 
                val_labels=None):
      self.hyper_params = hyper_params
      self.max_epochs = hyper_params["max_epochs"]
      self.batch_size = hyper_params["batch_size"]
      self.step_size = hyper_params["step_size"]
      self.lr = hyper_params["learning_rate"]
      self.units_per_layer = units_per_layer
      self.model = TorchModel(units_per_layer)
      self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
      self.loss_fun = torch.nn.BCEWithLogitsLoss()
      self.val_features = val_features
      self.val_labels = val_labels.reshape(-1, 1)
   
   # X=batch_features, y=batch_labels
   def take_step(self, X, y):
      predictions = self.model(X)
      self.optimizer.zero_grad()
      loss_value = self.loss_fun(predictions, y)
      loss_value.backward()
      self.optimizer.step()

      return loss_value
   
   # X=subtrain_features, y=subtrain_labels
   def fit(self, X, y):
      ds = CSV(X, y)
      dl = torch.utils.data.DataLoader(
         ds, batch_size=self.batch_size, shuffle=True
      )

      loss_df_list = []

      for epoch in range(self.max_epochs):
         #Subtrain
         self.model.train()

         for batch_features, batch_labels in dl:
            loss = self.take_step(batch_features, batch_labels)
         
         loss_row = pd.DataFrame({
            "epoch":[epoch],
            "set_name":"subtrain",
            "loss_value":[loss.item()]
         }, index=[0])
         loss_df_list.append(loss_row)

         if(self.val_features is not None):
            #Validation
            self.model.eval()
            val_loss = self.take_step(self.val_features, self.val_labels)
            val_loss_row = pd.DataFrame({
               "epoch":[epoch],
               "set_name":"validation",
               "loss_value":[val_loss.item()]
            }, index=[0])
            loss_df_list.append(val_loss_row)

      loss_df = pd.concat(loss_df_list)

      return loss_df
   
   # X=test_features
   def decision_function(self, X):
      #Validation
      self.model.eval()
      pred = self.model(X)

      return pred.detach().numpy()

   # X=test_features
   def predict(self, X):
      scores_tensor = torch.from_numpy(self.decision_function(X)).float()
      class_tensor = torch.where(scores_tensor > 0.5, 1, 0)
      
      return class_tensor.detach().numpy()  


class TorchLearnerCV:
   def __init__(self, val_features=None, val_labels=None, cv=2):
      self.cv = cv
      self.val_features = val_features
      self.val_labels = val_labels
   
   # X=train_features
   def fit(self, X, y, layers_list):
      if(len(layers_list) > 3):
         model = "Deep Learning"
      else:
         model = "Linear"

      input_tensor = torch.from_numpy(X.to_numpy()).float()
      output_tensor = torch.from_numpy(y.to_numpy()).float()

      print(type(X))
      print(type(y))
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
      val_features, val_labels = split_data_dict["test"]
         
      train_input_tensor = torch.from_numpy(subtrain_features.to_numpy()).float()
      train_output_tensor = torch.from_numpy(subtrain_labels.to_numpy()).float()

      val_input_tensor = torch.from_numpy(val_features.to_numpy()).float()
      val_output_tensor = torch.from_numpy(val_labels.to_numpy()).float()

      hyper_params = {
         "learning_rate": 0.01,
         "max_epochs": 100, 
         "batch_size": 1,
         "step_size": 0.01
      }

      torch_val = TorchLearner(hyper_params, layers_list, val_input_tensor, 
                               val_output_tensor)

      val_results = torch_val.fit(train_input_tensor, 
                                  train_output_tensor.unsqueeze(1))

      val_df = val_results[val_results["set_name"] == "validation"]
      min_row = val_df[val_df["loss_value"] == val_df["loss_value"].min()]
      linear_best_epochs = min_row["epoch"].values[0]
      linear_best_loss = min_row["loss_value"].values[0]
      print("Best epochs: " + str(linear_best_epochs) + " with a loss of " + 
            str(linear_best_loss))
      
      if(linear_best_epochs != 0):
         hyper_params["max_epochs"] = linear_best_epochs
      
      torch_run = TorchLearner(hyper_params, layers_list, self.val_features, 
                               self.val_labels)
      
      run_results = torch_run.fit(input_tensor, output_tensor)

      train_plot = p9.ggplot()+\
      p9.geom_point(
         p9.aes(
               x="epoch",
               y="loss_value",
               fill="set_name"
         ),
         data = run_results
      )+\
      p9.ggtitle(
         model + ": fold " + str(fold_number) + ", dataset " + data_name
      )
      print(train_plot)

      return torch_run


# Main loop over spam and zip data
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
      test_features, test_labels = split_data_dict["test"]

      train_nrow, train_ncol = train_features.shape
      input_tensor = torch.from_numpy(data_features.to_numpy()).float()
      output_tensor = torch.from_numpy(data_labels.to_numpy()).float()

      val_input_tensor = torch.from_numpy(test_features.to_numpy()).float()
      val_output_tensor = torch.from_numpy(test_labels.to_numpy()).float()

      torch_linear = TorchLearnerCV(val_input_tensor, 
                                    val_output_tensor).fit(data_features, 
                                          data_labels, [train_ncol, 100, 1])
      torch_deep = TorchLearnerCV(val_input_tensor, 
                                  val_output_tensor).fit(data_features, 
                                          data_labels, [train_ncol, 10, 50, 100, 1])
      
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

      print("test")
      print((np.repeat(most_freq_label, len(test_labels))))
      pred_dict = {
         "featureless":np.repeat(most_freq_label, len(test_labels)),
         "GridSearchCV + KNeighborsClassifier":clf.predict(test_features),
         "LogisticRegressionCV":linear.predict(test_features),
         "TorchLinear":torch_linear.predict(val_input_tensor).flatten(),
         "TorchDeep":torch_deep.predict(val_input_tensor).flatten()
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