from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from config.config import task_type, client_tree_num, client_num,  partition_col_index
from utils.fever_data_pre_processing import fever_preprocess_split, do_fl_partitioning
from utils.xgb_util import TreeDataset, construct_tree
from evaluation.server_evaluation import start_experiment


def main():
    data = pd.read_csv('data/fever.csv')
    X_train, y_train, X_test, y_test = fever_preprocess_split(data, 0.2)

    trainset = TreeDataset(np.array(X_train, copy=True), np.array(y_train, copy=True))
    testset = TreeDataset(np.array(X_test, copy=True), np.array(y_test, copy=True))

    # Global Tree
    global_tree = construct_tree(X_train, y_train, client_tree_num, task_type)
    preds_train = global_tree.predict(X_train)
    preds_test = global_tree.predict(X_test)

    # Results
    result_train = mean_squared_error(y_train, preds_train)
    result_test = mean_squared_error(y_test, preds_test)
    print("Global XGBoost Training MSE: %f" % result_train)
    print("Global XGBoost Testing MSE: %f" % result_test)

    # Client Trees for Comparison
    client_trees_comparison = []
    trainloaders, _, testloader = do_fl_partitioning(
        trainset, testset, pool_size=client_num, batch_size="whole", val_ratio=0.0
    )

    for i, trainloader in enumerate(trainloaders):
        for local_dataset in trainloader:
            local_X_train, local_y_train = local_dataset[0], local_dataset[1]
            tree = construct_tree(local_X_train, local_y_train, client_tree_num, task_type)
            client_trees_comparison.append(tree)

            print("Local Client Training Data Size: ", len(local_X_train))

            # Predicting
            preds_train = client_trees_comparison[-1].predict(local_X_train)
            preds_test = client_trees_comparison[-1].predict(X_test)

            # Results
            result_train = mean_squared_error(local_y_train, preds_train)
            result_test = mean_squared_error(y_test, preds_test)
            print("Local Client %d XGBoost Training MSE: %f" % (i, result_train))
            print("Local Client %d XGBoost Testing MSE: %f" % (i, result_test))

    # Staring federated experiment
    start_experiment(
        task_type=task_type,
        trainset=trainset,
        testset=testset,
        num_rounds=5,
        client_tree_num=client_tree_num,
        client_pool_size=client_num,
        num_iterations=500,
        batch_size=64,
        fraction_fit=1.0,
        min_fit_clients=1,
        val_ratio=0.0,
    )


if __name__ == "__main__":
    main()
