import sklearn
import math
import pandas as pd
import numpy as np
import plotnine as p9
import torch
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LassoCV

# Use forestfires and airfoil datasets
data_info_dict = {
    "forest_fires":("forestfires.csv",",",True),
    "air_foil":("airfoil_self_noise.tsv","\t",False),
}
data_dict = {}

list_of_accuracy_rows = []
list_of_det_rows = []

# Scale and normalize data and data labels
for data_name,(file_name,sep,log_trans) in data_info_dict.items():

    data_df = pd.read_csv(file_name,sep=sep,header=0)
    data_nrow, data_ncol = data_df.shape
    label_col_num = data_ncol-1
    data_label_vec = data_df.iloc[:,label_col_num]

    # Log transform the forest fires dataset
    if log_trans:
        data_label_vec = np.log(data_label_vec+1)

    label_sd = math.sqrt(data_label_vec.var())
    standard_label_vec = (
        data_label_vec-data_label_vec.mean()
    )/label_sd
    is_feature_col = (
        np.arange(data_ncol) != label_col_num
    ) & (
        data_df.dtypes != "object"
    )

    data_features = data_df.loc[:,is_feature_col]
    feature_nrow, feature_ncol= data_features.shape
    feature_mean = data_features.mean().to_numpy().reshape(1,feature_ncol)
    feature_std = data_features.std().to_numpy().reshape(1,feature_ncol)
    feature_scaled = (data_features-feature_mean)/feature_std

    data_dict[data_name] = (feature_scaled, standard_label_vec.to_frame())

    print(type(feature_scaled))
    print(type(standard_label_vec.to_frame()))


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
            self, units_per_layer, step_size=0.005, 
            batch_size=20, max_epochs=100):
        self.max_epochs = max_epochs
        self.batch_size=batch_size
        self.model = TorchModel(units_per_layer)
        self.loss_fun = torch.nn.MSELoss()
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
        ).float().reshape(train_nrow, 1)
        
        # features tensor - test
        test_input_tensor = torch.from_numpy(
            test_features.to_numpy()
        ).float()

        # labels tensor - test
        test_output_tensor = torch.from_numpy(
            test_labels.to_numpy()
        ).float().reshape(test_nrow, 1)

        # Featureless
        # Predict mean of train label values
        mean_value = torch.mean(input_tensor)
        featureless_tensor = np.repeat(mean_value, len(test_labels))

        # Nearest Neighbors + GridSearchCV
        nearest_model = GridSearchCV(KNeighborsRegressor(), 
                        {"n_neighbors":[k+1 for k in range(20)]})
        nearest_model.fit(train_features, train_labels)
        
        # Linear model (LassoCV)
        linear = LassoCV(cv=2, max_iter=1000)
        linear.fit(train_features, train_labels)
        
        # TorchLearnerCV_linear
        torch_linear = TorchLearnerCV(units_per_layer=[train_ncol, 1])
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
        torch_deep = TorchLearnerCV(units_per_layer=[train_ncol, 100, 1])
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
            "Featureless":featureless_tensor,
            "KNeighborsRegressor+GridSearchCV":nearest_model.predict(test_features),
            "LassoCV":linear.predict(test_features),
            "TorchLinear":torch_linear.predict(test_input_tensor),
            "TorchDeep":torch_deep.predict(test_input_tensor)
        }

        for algorithm, pred_vec in pred_dict.items():
            print("Algorithm: " + algorithm)
            
            square_loss = torch.square(torch.from_numpy(test_labels.to_numpy()) 
                                       - pred_vec)
            mean_square_loss = square_loss.detach().numpy().mean()

            list_of_accuracy_rows.append(pd.DataFrame({
                "square_loss":mean_square_loss,
                "algorithm":[algorithm],
                "fold_number":[fold_number],
                "data_set":data_name
            }))

            sum_squares = torch.square(square_loss.mean() - pred_vec)
            r_squared = 1 - (square_loss.mean() / sum_squares.mean())

            list_of_det_rows.append(pd.DataFrame({
                "r_squared":r_squared.detach().numpy().mean(),
                "algorithm":[algorithm],
                "fold_number":[fold_number],
                "data_set":data_name
            }))

output_data = pd.concat(list_of_accuracy_rows)
det_data = pd.concat(list_of_det_rows)
print(output_data)

output_plot = p9.ggplot()+\
   p9.geom_point(
      p9.aes(
            x="square_loss",
            y="algorithm",
      ),
      data = output_data
   )+\
   p9.facet_grid(". ~ data_set")

print(output_plot)

det_plot = p9.ggplot()+\
   p9.geom_point(
      p9.aes(
            x="r_squared",
            y="algorithm",
      ),
      data = det_data
   )+\
   p9.facet_grid(". ~ data_set")

print(det_plot)
