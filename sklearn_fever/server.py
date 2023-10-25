import flwr as fl
import utils
from flwr.common import NDArrays, Scalar
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from typing import Dict, Optional, Tuple
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateRes,
    FitRes
)
import json

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LinearRegression):
    """Return an evaluation function for server-side evaluation."""

    #X_test, y_test = utils.generate_synthetic_data(200, 4, [0.81, 0.4, 0.1, 0.05])
    _, (X_test, y_test) = utils.load_fever()

    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        utils.set_model_params(model, parameters)
        loss = mean_absolute_error(y_test, model.predict(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LinearRegression()
    # utils.set_initial_params(model)
    strategy = AggregateCustomMetricStrategy(
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round
    )
    hist = fl.server.start_server(server_address="localhost:8081", strategy=strategy, config=fl.server.ServerConfig(num_rounds=3))
    print(hist)
