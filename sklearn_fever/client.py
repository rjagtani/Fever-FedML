import warnings
import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import utils
import flwr as fl


class SyntheticClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.X_train, self.y_train = x_train, y_train
        self.X_test, self.y_test = x_test, y_test
        self.is_trained = False

    def get_parameters(self, config):
        if not self.is_trained:
            # Return some default values or None if the model is not trained
            return [np.zeros((self.X_train.shape[1], )), np.zeros((1, ))]
        # Get model parameters as a list of numpy ndarrays
        coefs = [self.model.coef_]
        intercepts = [self.model.intercept_]
        return coefs + intercepts

    def set_parameters(self, weights):
        # Set model parameters from a list of numpy ndarrays
        self.model.coef_ = weights[0]
        self.model.intercept_ = weights[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = self.model.score(self.X_test, self.y_test)
        # fi_dict = {}
        # try:
        #     imputer = sage.MarginalImputer(model, self.X_test[100:200])
        #     estimator = sage.KernelEstimator(imputer, 'mse')
        #     sage_values = estimator(self.X_test, self.y_test, batch_size=4, thresh=0.04, bar=True)
        #     feature_names = ['trip_distance(km)', 'city', 'motor_way', 'country_roads', 'month']
        #     fi_dict = dict(zip(feature_names, sage_values.values))
        # except Exception as e:
        #     print(f"An error occurred while creating fi_dict: {e}")
    #sage_values.plot(np.array(feature_names))
    #print(fi_dict)
    # Use negative accuracy as a makeshift "loss" since lower is better in Flower
        return -accuracy, len(self.X_test), {"accuracy": accuracy}


if __name__ == "__main__":
    #x_train, y_train = utils.generate_synthetic_data(800, 4, [0.81, 0.4, 0.1, 0.05])
    #x_test, y_test = utils.generate_synthetic_data(200, 4, [0.81, 0.4, 0.1, 0.05])
    (x_train, y_train), (x_test, y_test) = utils.load_fever()
    model = utils.create_sklearn_model()
    client = SyntheticClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client(server_address="localhost:8081", client=client)



