import sklearn
import math
import pandas as pd
import numpy as np
import plotnine as p9
import torch
import torchvision
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Datasets spam and zip
zip_df = pd.read_csv("zip.test.gz",sep=" ",header=None)

zip_label_col_num = 0
zip_label_vec = zip_df.iloc[:,zip_label_col_num]

is_label_col = zip_df.columns == zip_label_col_num
zip_features = zip_df.iloc[:,~is_label_col]
zip_labels = zip_df.iloc[:,is_label_col]
feature_nrow, feature_ncol = zip_features.shape

list_of_accuracy_rows = []

# MNIST dataset
ds = torchvision.datasets.MNIST(
    root="~/teaching/cs499-599-fall-2022/data", 
    download=True,
    transform=torchvision.transforms.ToTensor(),
    train=False)
dl = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
for mnist_features, mnist_labels in dl:
    pass
mnist_features = pd.DataFrame(mnist_features.flatten(start_dim=1).numpy())
mnist_labels = pd.DataFrame(mnist_labels.numpy())

data_dict = {
    "zip" : (zip_features, zip_labels),
    "MNIST" : (mnist_features, mnist_labels)
}

# Number of classes in dataset - this is 10 for both datasets
n_classes = 10

class TorchModel(torch.nn.Module):
    def __init__(self, units_per_layer):
        super(TorchModel, self).__init__()
        seq_args = []
        second_to_last = len(units_per_layer)-1
        for layer_i in range(second_to_last):
            next_i = layer_i+1
            layer_units = units_per_layer[layer_i]
            next_units = units_per_layer[next_i]
            seq_args.append(torch.nn.Linear(layer_units, next_units))
            if layer_i < second_to_last-1:
                seq_args.append(torch.nn.ReLU())
        self.stack = torch.nn.Sequential(*seq_args)

    def forward(self, features):
        return self.stack(features)
    

class CSV(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, item):
        return self.features[item,:], self.labels[item]
    
    def __len__(self):
        return len(self.labels)
    

class TorchLearner:
    def __init__(
            self, units_per_layer, step_size=0.015, 
            batch_size=20, max_epochs=150):
        self.max_epochs = max_epochs
        self.batch_size=batch_size
        self.model = TorchModel(units_per_layer)
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=step_size)
        
    def fit(self, split_data_dict):
        ds = CSV(
            split_data_dict["subtrain"]["X"], 
            split_data_dict["subtrain"]["y"])
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True)
        train_df_list = []
        for epoch_number in range(self.max_epochs):
            for batch_features, batch_labels in dl:
                self.optimizer.zero_grad()
                loss_value = self.loss_fun(
                    self.model(batch_features), batch_labels)
                loss_value.backward()
                self.optimizer.step()
            for set_name, set_data in split_data_dict.items():
                pred_vec = self.model(set_data["X"])
                set_loss_value = self.loss_fun(pred_vec, set_data["y"])
                train_df_list.append(pd.DataFrame({
                    "set_name":[set_name],
                    "loss":float(set_loss_value),
                    "epoch":[epoch_number]
                }))
        self.train_df = pd.concat(train_df_list)
    
    def predict(self, test_features):
        return self.model(test_features)
    

class TorchLearnerCV:
    def __init__(self, n_folds = 3, units_per_layer=[feature_ncol,1]):
        self.units_per_layer = units_per_layer
        self.n_folds = n_folds

    def fit(self, train_features, train_labels):
        train_nrow, train_ncol = train_features.shape
        times_to_repeat=int(math.ceil(train_nrow/self.n_folds))
        fold_id_vec = np.tile(torch.arange(self.n_folds), 
                              times_to_repeat)[:train_nrow]
        np.random.shuffle(fold_id_vec)
        cv_data_list = []
        for validation_fold in range(self.n_folds):
            is_split = {
                "subtrain":fold_id_vec != validation_fold,
                "validation":fold_id_vec == validation_fold
                }
            split_data_dict = {}
            for set_name, is_set in is_split.items():
                set_y = train_labels[is_set]
                split_data_dict[set_name] = {
                    "X":train_features[is_set,:],
                    "y":set_y}
            learner = TorchLearner(self.units_per_layer)
            learner.fit(split_data_dict)
            cv_data_list.append(learner.train_df)
        self.cv_data = pd.concat(cv_data_list)
        self.train_df = self.cv_data.groupby(["set_name",
                                              "epoch"]).mean().reset_index()
        valid_df = self.train_df.query("set_name=='validation'")
        best_epochs = valid_df["loss"].argmin()

        if(best_epochs == 0):
            best_epochs = 1

        self.min_df = valid_df.query("epoch==%s"%best_epochs)
        self.final_learner = TorchLearner(self.units_per_layer, 
                                          max_epochs=best_epochs)
        self.final_learner.fit({"subtrain":{"X":train_features,
                                            "y":train_labels}})

    def predict(self, test_features):
        return self.final_learner.predict(test_features)


# Main loop
# Splits into train / test for each dataset
for data_name, (data_features, data_labels) in data_dict.items():

    # Split into 3 folds
    kf = KFold(n_splits=3, shuffle=True)
    print("Data set: " + data_name)
    enum_obj = enumerate(kf.split(data_features))
    
    # Loop through each fold
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
        test_nrow, test_ncol = test_features.shape

        # features tensor - train
        input_tensor = torch.from_numpy(
            train_features.to_numpy()
        ).float()

        # labels tensor - train
        output_tensor = torch.from_numpy(
            train_labels.to_numpy()
        ).long()
        
        # features tensor - test
        test_input_tensor = torch.from_numpy(
            test_features.to_numpy()
        ).float()

        # labels tensor - test
        test_output_tensor = torch.from_numpy(
            test_labels.to_numpy()
        ).long()

        # Featureless
        # Predict most frequent train label
        most_freq_label = train_labels.value_counts().index[0]
        test_features, test_labels = split_data_dict["test"]

        # Nearest Neighbors + GridSearchCV
        nearest_model = GridSearchCV(KNeighborsClassifier(), 
                        {"n_neighbors":[k+1 for k in range(20)]})
        nearest_model.fit(train_features, train_labels)
        
        # Linear model (LassoCV)
        linear = make_pipeline(StandardScaler(), 
                           LogisticRegressionCV(cv=2, max_iter=1000))
        linear.fit(train_features, train_labels)
        
        # TorchLearnerCV_linear
        torch_linear = TorchLearnerCV(units_per_layer=[train_ncol, n_classes])
        torch_linear.fit(input_tensor, output_tensor)
        torch_linear_plot = p9.ggplot()+\
        p9.geom_line(
            p9.aes(
                x="epoch",
                y="loss",
                color="set_name"
            ),
            data=torch_linear.train_df)+\
        p9.geom_point(
            p9.aes(
                x="epoch",
                y="loss",
                color="set_name"
            ),
            data=torch_linear.min_df)+\
        p9.ggtitle(
            "Torch linear: fold " + str(fold_number) + ", dataset " + data_name
        )
        print(torch_linear_plot)

        # TorchLearnerCV_deep
        torch_deep = TorchLearnerCV(units_per_layer=[train_ncol, 100, 50, 
                                                     n_classes])
        torch_deep.fit(input_tensor, output_tensor)
        torch_deep_plot = p9.ggplot()+\
        p9.geom_line(
            p9.aes(
                x="epoch",
                y="loss",
                color="set_name"
            ),
            data=torch_deep.train_df)+\
        p9.geom_point(
            p9.aes(
                x="epoch",
                y="loss",
                color="set_name"
            ),
            data=torch_deep.min_df)+\
        p9.ggtitle(
            "Torch deep: fold " + str(fold_number) + ", dataset " + data_name
        )
        print(torch_deep_plot)

        # Determine test square loss values
        pred_dict = {
            "Featureless":np.repeat(most_freq_label, len(test_labels)),
            "KNClassifier+GridSearchCV":nearest_model.predict(test_features),
            "LogisticRegressionCV":linear.predict(test_features),
            "TorchLinear":torch.argmax(torch_linear.predict(test_input_tensor), 
                                       dim=1).numpy(),
            "TorchDeep":torch.argmax(torch_deep.predict(test_input_tensor), 
                                       dim=1).numpy()
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