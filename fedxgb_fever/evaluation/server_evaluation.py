from xgboost import XGBClassifier, XGBRegressor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import flwr as fl

from utils.cnn import CNN
from utils.cnn import test
from flower.flower_client import tree_encoding_loader, FL_Client
from flower.flower_server import FL_Server
from utils.fever_data_pre_processing import do_fl_partitioning

import functools
from flwr.server.strategy import FedXgbNnAvg
from flwr.server.app import ServerConfig
from typing import Dict, List, Tuple, Union
from flwr.common import Parameters, Scalar
from flwr.server.client_manager import SimpleClientManager
from flwr.server.history import History
from flwr.common import parameters_to_ndarrays
from config.config import client_num

def print_model_layers(model: nn.Module) -> None:
    print(model)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def serverside_eval(
    server_round: int,
    parameters: Tuple[
        Parameters,
        Union[
            Tuple[XGBClassifier, int],
            Tuple[XGBRegressor, int],
            List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
        ],
    ],
    config: Dict[str, Scalar],
    task_type: str,
    testloader: DataLoader,
    batch_size: int,
    client_tree_num: int,
    client_num: int,
) -> Tuple[float, Dict[str, float]]:
    """An evaluation function for centralized/serverside evaluation over the entire test set."""
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = CNN()
    # print_model_layers(model)

    model.set_weights(parameters_to_ndarrays(parameters[0]))
    model.to(device)

    trees_aggregated = parameters[1]
    testloader = tree_encoding_loader(
        testloader, batch_size, trees_aggregated, client_tree_num, client_num
    )
    loss, result, _ = test(
        task_type, model, testloader, device=device, log_progress=False
    )

    if task_type == "BINARY":
        print(
            f"Evaluation on the server: test_loss={loss:.4f}, test_accuracy={result:.4f}"
        )
        return loss, {"accuracy": result}
    elif task_type == "REG":
        print(f"Evaluation on the server: test_loss={loss:.4f}, test_mse={result:.4f}")
        return loss, {"mse": result}


def start_experiment(
    task_type: str,
    trainset: Dataset,
    testset: Dataset,
    num_rounds: int = 5,
    client_tree_num: int = 50,
    client_pool_size: int = 5,
    num_iterations: int = 100,
    fraction_fit: float = 1.0,
    min_fit_clients: int = 2,
    batch_size: int = 32,
    val_ratio: float = 0.1,
) -> History:
    client_resources = {"num_cpus": 0.5}  # 2 clients per CPU

    # Partition the dataset into subsets reserved for each client.
    # - 'val_ratio' controls the proportion of the (local) client reserved as a local test set
    # (good for testing how the final model performs on the client's local unseen data)
    trainloaders, valloaders, testloader = do_fl_partitioning(
        trainset,
        testset,
        batch_size="whole",
        pool_size=client_pool_size,
        val_ratio=val_ratio,
    )
    print(
        f"Data partitioned across {client_pool_size} clients"
        f" and {val_ratio} of local dataset reserved for validation."
    )

    # Configure the strategy
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        print(f"Configuring round {server_round}")
        return {
            "num_iterations": num_iterations,
            "batch_size": batch_size,
        }

    # FedXgbNnAvg
    strategy = FedXgbNnAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit if val_ratio > 0.0 else 0.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=client_pool_size,  # all clients should be available
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=(lambda r: {"batch_size": batch_size}),
        evaluate_fn=functools.partial(
            serverside_eval,
            task_type=task_type,
            testloader=testloader,
            batch_size=batch_size,
            client_tree_num=client_tree_num,
            client_num=client_num,
        ),
        accept_failures=False,
    )

    print(
        f"FL experiment configured for {num_rounds} rounds with {client_pool_size} client in the pool."
    )
    print(
        f"FL round will proceed with {fraction_fit * 100}% of clients sampled, at least {min_fit_clients}."
    )

    def client_fn(cid: str) -> fl.client.Client:
        """Creates a federated learning client"""
        if val_ratio > 0.0 and val_ratio <= 1.0:
            return FL_Client(
                task_type,
                trainloaders[int(cid)],
                valloaders[int(cid)],
                client_tree_num,
                client_pool_size,
                cid,
                log_progress=False,
            )
        else:
            return FL_Client(
                task_type,
                trainloaders[int(cid)],
                None,
                client_tree_num,
                client_pool_size,
                cid,
                log_progress=False,
            )

    # Start the simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        server=FL_Server(client_manager=SimpleClientManager(), strategy=strategy),
        num_clients=client_pool_size,
        client_resources=client_resources,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    print(history)

    return history