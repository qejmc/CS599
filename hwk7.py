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
   

class Node:
   def __repr__(self):
      return "%s(%s)"%(self.__class__.__name__, self.value.shape)


class InitialNode(Node):
   def __init__(self, value):
      self.value = value

   def backward(self):
      #print("%s Done!"%(self))
      pass


class Operation(Node):
   # Slides say no init method, but homework says to implement it
   def __init__(self, *node_list):
      self.node_list = node_list
      for input_name, node in zip(self.input_names, node_list):
         setattr(self, input_name, node)
      self.value = self.forward()

   def backward(self):
      gradients = self.gradient()
      for parent_node, grad in zip(self.input_names, gradients):
         if grad is not None and parent_node.value.shape != grad.shape:
            raise ValueError("Value%s not same shape as grad%s"%(
                             str(parent_node.value.shape),
                             str(grad.shape)))
         parent_node.grad = grad
         #print("in %s.backward, calling %s.backward"%(self, parent_node))
         parent_node.backward()


class mm(Operation):
   def __init__(self, feature_node, weight_node):
      self.input_names = [feature_node, weight_node]
      self.value = self.forward()

   def gradient(self):
      feature_node, weight_node = self.input_names
      # features X: b x p
      # weights W: p x u = p x 1
      # pred A: b x u = b x 1
      # where b is batch size, p is number of input features, u is number of 
      # outputs
      # grad_A(b x u) W(u x p)
      return [
         np.matmul(self.grad, weight_node.value.T),
         np.matmul(feature_node.value.T, self.grad)]
   
   def forward(self):
      feature_node, weight_node = self.input_names
      return np.matmul(feature_node.value, weight_node.value)


class relu(Operation):
   def __init__(self, pred_node):
      self.input_names = [pred_node]
      self.value = self.forward()

   def gradient(self):
      pred_node = self.input_names[0]
      # Gradients the same if A > 0, else 0
      pred_grad = np.where(pred_node.value > 0, self.grad, 0)
      return [pred_grad, None]
   
   def forward(self):
      pred_node = self.input_names[0]
      # H = A if A > 0, else 0
      return np.where(pred_node.value > 0, pred_node.value, 0)
   

class logistic_loss(Operation):
   def __init__(self, pred_node, output_node):
      self.input_names = [pred_node, output_node]
      # In order for np log function, data must be between -1 and 1
      self.output_vec = output_node.value
      if not ((self.output_vec==1) | (self.output_vec==-1)).all():
         raise ValueError("Labels must be [-1, 1]")
      self.value = self.forward()

   def gradient(self):
      pred_node, output_node = self.input_names
      output_node.value = output_node.value.reshape(-1, 1)
      pred_grad = -output_node.value/(
         1+np.exp(output_node.value * pred_node.value)
      )
      return [pred_grad, None]
   
   def forward(self):
      pred_node, output_node = self.input_names
      return np.log(1+np.exp(-self.output_vec*pred_node.value))


class AutoMLP:
   def __init__(self, max_epochs, batch_size, step_size, units_per_layer):
      """Store hyper-parameters as attributes, then initialize
      weight_node_list attribute to a list of InitialNode instances."""
      self.max_epochs = max_epochs
      self.batch_size = batch_size
      self.step_size = step_size
      self.units_per_layer = units_per_layer
      self.weight_node_list = []
      self.val_features = None
      self.val_labels = None
      n_col = units_per_layer[0]
      for node in range(len(units_per_layer) - 1):
         if(node == 0):
            weight_node = InitialNode(
               np.repeat(0.1, n_col).reshape(n_col, 1)
            )
         self.weight_node_list.append(weight_node)

   # X = batch_features
   def get_pred_node(self, X):
      """return node of predicted values for feature matrix X"""
      feature_node = InitialNode(X)

      for layer in range(len(self.weight_node_list)):
         a_node = mm(feature_node, self.weight_node_list[layer])
         h_node = relu(a_node)

      return h_node
   
   # X = batch_features, y = batch_labels
   def take_step(self, X, y):
      """call get_pred_node, then instantiate logistic_loss, call its
      backward method to compute gradients, then for loop over
      weight_node_list (one iteration of gradient descent).
      """
      train_nrow = y.shape

      label_vec = np.where(y==1, 1, -1).reshape(train_nrow)
      label_node = InitialNode(label_vec)
      pred_node = self.get_pred_node(X)
      self.final_node = logistic_loss(pred_node, label_node)
      self.final_node.backward()

      for weight_node in self.weight_node_list:
         weight_node.value -= weight_node.grad * self.step_size

   # X = subtrain_features, y = subtrain_labels
   def fit(self, X, y):
      """Gradient descent learning of weights"""
      X_tensor = torch.from_numpy(X.to_numpy()).float()
      y_tensor = torch.from_numpy(y.to_numpy()).float()
      ds = CSV(X_tensor, y_tensor)
      dl = torch.utils.data.DataLoader(
         ds, batch_size=self.batch_size, shuffle=True
      )
      loss_df_list = []
      for epoch in range(self.max_epochs):
         for batch_features, batch_labels in dl:
            self.take_step(batch_features.numpy(), batch_labels.numpy())
         loss_row = pd.DataFrame({
         "epoch":[epoch],
         "set_name":"subtrain",
         "loss_value":[np.mean(self.final_node.value)]
         }, index=[0])
         if self.val_features is not None:
            self.take_step(self.val_features.to_numpy(), 
                           self.val_labels.to_numpy())
            val_loss_row = pd.DataFrame({
            "epoch":[epoch],
            "set_name":"validation",
            "loss_value":[np.mean(self.final_node.value)]
            }, index=[0])
            loss_df_list.append(val_loss_row)
         loss_df_list.append(loss_row)
      self.loss_df = pd.concat(loss_df_list)

   # X = test_features
   def decision_function(self, X):
      """Return numpy vector of predicted scores"""
      pred_node = self.get_pred_node(X)
      return pred_node.value

   # X = test_features
   def predict(self, X):
      """Return numpy vector of predicted classes"""
      pred_vec = self.decision_function(X)
      class_vec = np.where(pred_vec > 0.5, 1, 0)
      
      return class_vec


class AutoGradLearnerCV:
   def __init__(self, max_epochs, batch_size, step_size, units_per_layer):
      self.batch_size = batch_size
      self.step_size = step_size
      self.units_per_layer = units_per_layer
      self.subtrain_model = AutoMLP(max_epochs, batch_size, step_size, 
                                    units_per_layer)
   
   def fit(self, X, y):
      """cross-validation for selecting the best number of epochs"""
      if(len(self.units_per_layer) > 2):
         model = "Deep Learning"
      else:
         model = "Linear"

      y = y.to_frame()
      kf = KFold(n_splits=2, shuffle=True, random_state=1)
      enum_obj = enumerate(kf.split(X))

      for f_number, index_tuple in enum_obj:
         
         zip_obj = zip(["train","test"], index_tuple)
         split_data_dict = {}

         for set_name, set_indices in zip_obj:
            split_data_dict[set_name] = (
                  X.iloc[set_indices,:],
                  y.iloc[set_indices,0])
      
      subtrain_features, subtrain_labels = split_data_dict["train"]
      val_features, val_labels = split_data_dict["test"]

      self.subtrain_model.val_features = val_features
      self.subtrain_model.val_labels = val_labels
      self.subtrain_model.fit(subtrain_features, subtrain_labels)

      results = self.subtrain_model.loss_df

      train_plot = p9.ggplot()+\
      p9.geom_point(
         p9.aes(
               x="epoch",
               y="loss_value",
               fill="set_name"
         ),
         data = results
      )+\
      p9.ggtitle(
          model + ": fold " + str(fold_number) + ", dataset " + data_name
      )
      print(train_plot)

      val_df = results[results["set_name"] == "validation"]
      min_row = val_df[val_df["loss_value"] == val_df["loss_value"].min()]
      best_epochs = min_row["epoch"].values[0] + 1
      best_loss = min_row["loss_value"].values[0]
      print("Best epochs: " + str(best_epochs) + " with a loss of " + 
            str(best_loss))
      
      self.train_model = AutoMLP(best_epochs, self.batch_size, self.step_size, 
                                 self.units_per_layer)
      self.train_model.fit(X, y)

   def predict(self, X):
      return self.train_model.predict(X)

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

      auto = AutoGradLearnerCV(100, 10, 0.1, [train_ncol, 1])
      auto.fit(train_features, train_labels)

      auto_deep = AutoGradLearnerCV(100, 10, 0.1, [train_ncol, 10, 5, 5, 1])
      auto_deep.fit(train_features, train_labels)
      
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

      pred_dict = {
         "featureless":np.repeat(most_freq_label, len(test_labels)),
         "GridSearchCV + KNeighborsClassifier":clf.predict(test_features),
         "LogisticRegressionCV":linear.predict(test_features),
         "AutoGradLearnerCV_Linear":auto.predict(test_features).reshape(-1),
         "AutoGradLearnerCV_Deep":auto_deep.predict(test_features).reshape(-1)
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