import numpy as np

from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle


class ScalarFieldRegressionModel:
    """ Regression model for scalar field.

    Attributes:
        order: order of the scalar field regression model.
        model: sklearn.linear_model.BayesianRidge for the linear regression.
        features: proxy feature data.
        polynomial_features: sklearn.preprocessing.PolynomialFeatures class for the target order.
    """

    def __init__(self, order=1):
        """

        Args:
            order: order of the scalar field regression model.
        """
        self.order = order
        self.polynomial_features = PolynomialFeatures(order)

        self.features = {}

        self.model = BayesianRidge()
        self.I_shape = None

    def set_feature(self, key, feature):
        """ Set proxy feature with the given key.

        Args:
            key: feature id.
            feature: (h, w) image data.
        """
        self.features[key] = feature

    def set_alpha(self, A):
        """ Set alpha mask to get the target region.

        Args:
            A: (h, w) alpha mask image.
        """
        self.A = A

    def W_u(self):
        """ Compute feature matrix W_u for the given order.

        Returns:
            W_u: (h*w, #poly(N_F, order)) feature matrix.
        """
        W_u = []

        for key, feature_k in self.features.items():
            W_u.append(feature_k.flatten())

        W_u = np.array(W_u).T

        W_u = self.polynomial_features.fit_transform(W_u)

        return W_u

    def fit(self, I_dash, A=None):
        """ Fit color/length/width field model for the target image.

        Args:
            I_dash: (h, w, d) target color/length/width image.
            A: (h, w) alpha mask image.
        """
        W_u_constraints, I_constraints = self.constraints(I_dash, A)
        self.fit_constraints(W_u_constraints, I_constraints)
        return

    def constraints(self, I_dash, A=None):
        """ Return the model constraints for multi-exemplars.

        Args:
            I_dash: (h, w, d) target color/length/width image.
            A: (h, w) alpha mask image.

        Returns:
            W_u_constraints: constraints for feature matrix.
            I_constraints: constraints for the target color/length/width.
        """
        self.I_shape = I_dash.shape
        h, w = I_dash.shape[:2]
        num_data = h * w

        W_u = self.W_u()
        I_dash_flat = I_dash.reshape(num_data, -1)

        if A is not None:
            A_flat = A.flatten()
            W_u = W_u[A_flat > 0.5, :]
            I_dash_flat = I_dash_flat[A_flat > 0.5, :]

        num_samples = self.num_samples
        if W_u.shape[0] > num_samples:
            W_u_constraints = shuffle(W_u, random_state=0, n_samples=num_samples)
            I_constraints = shuffle(I_dash_flat, random_state=0, n_samples=num_samples)
        else:
            W_u_constraints = W_u
            I_constraints = I_dash_flat
        return W_u_constraints, I_constraints

    def fit_constraints(self, W_u_constraints, I_constraints):
        """ Fit length/width field model for the given constraints.

        Args:
            W_u_constraints: constraints for feature matrix.
            I_constraints: constraints for the target length/width.
        """
        self.model.fit(W_u_constraints, I_constraints)
        return

    def predict(self, I_shape=None):
        """ predict length/width field using the model.

        Args:
            I_shape: target image size.
        Returns:
            I_tilde: (h, w, d) predicted length/width image.
        """
        if I_shape is None:
            I_shape = self.I_shape
        W_u = self.W_u()

        I_tilde_flat = self.model.predict(W_u)

        I_tilde = I_tilde_flat.reshape(I_shape)

        return I_tilde


class ColorFieldRegressionModel:
    """ Regression model for color field.

    Attributes:
        order: order of the color field regression model.
        model: sklearn.linear_model.BayesianRidge for the linear regression.
        features: proxy feature data.
        polynomial_features: sklearn.preprocessing.PolynomialFeatures class for the target order.
    """

    def __init__(self, order=1):
        """

        Args:
            order: order of the color field regression model.
        """
        self.order = order
        self.models = []

        for ci in range(4):
            model = ScalarFieldRegressionModel(self.order)
            self.models.append(model)
        self.I_shape = None

    def set_feature(self, key, feature):
        """ Set proxy feature with the given key.

        Args:
            key: feature id.
            feature: (h, w) image data.
        """
        for model in self.models:
            model.set_feature(key, feature)

    def fit(self, I_dash, A=None):
        """ Fit color/length/width field model for the target image.

        Args:
            I_dash: (h, w, d) target color/length/width image.
            A: (h, w) alpha mask image.
        """
        self.I_shape = I_dash.shape
        h, w, cs = I_dash.shape

        I_tilde = np.array(I_dash)
        phi = []

        for ci in range(cs):
            model = self.models[ci]
            I_tilde_i, phi_i = model.fit(I_dash[:, :, ci].reshape(h, w), A)
            I_tilde[:, :, ci] = I_tilde_i
            phi.append(phi_i)

        return I_tilde, phi

    def predict(self, I_shape=None):
        """ predict length/width field using the model.

        Args:
            I_shape: target image size.
        Returns:
            I_tilde: (h, w, d) predicted length/width image.
        """
        if I_shape is None:
            I_shape = self.I_shape

        h, w, cs = I_shape

        I_tilde = np.zeros(I_shape)

        for ci in range(cs):
            I_tilde_i = self.models[ci].predict(I_shape[:2])
            I_tilde[:, :, ci] = I_tilde_i
        return I_tilde


class NearestNeighborModel:
    """ Nearest neighbor model for color/length/width field.

    Attributes:
        model: sklearn.neighbors.KNeighborsRegressor for the nearest neighbor model.
        features: proxy feature data.
        num_samples: number of sampling constraints.
    """

    def __init__(self, num_samples=2000):
        """

        Args:
            num_samples: number of sampling constraints.
        """
        self.features = {}
        self.model = KNeighborsRegressor()

        self.num_samples = num_samples

    def set_feature(self, key, feature):
        """ Set proxy feature with the given key.

        Args:
            key: feature id.
            feature: (h, w) image data.
        """
        self.features[key] = feature

    def clean_internal(self):
        """ Clean internal data to save the model as pickle data.

        Note:
            Init the large internal attribute (features).
        """
        self.features = {}

    def W_u(self):
        """ Compute feature matrix W_u.

        Returns:
            W_u: (h*w, N_F) feature matrix.
        """
        W_u = []

        for key, feature_k in self.features.items():
            W_u.append(feature_k.flatten())

        W_u = np.array(W_u).T
        return W_u

    def fit(self, I_dash, A=None):
        """ Fit color/length/width field model for the target image.

        Args:
            I_dash: (h, w, d) target color/length/width image.
            A: (h, w) alpha mask image.
        """
        W_u_constraints, I_constraints = self.constraints(I_dash, A)
        self.fit_constraints(W_u_constraints, I_constraints)
        return

    def constraints(self, I_dash, A=None):
        """ Return the model constraints for multi-exemplars.

        Args:
            I_dash: (h, w, d) target color/length/width image.
            A: (h, w) alpha mask image.

        Returns:
            W_u_constraints: constraints for feature matrix.
            I_constraints: constraints for the target color/length/width.
        """
        self.I_shape = I_dash.shape
        h, w = I_dash.shape[:2]
        num_data = h * w

        W_u = self.W_u()
        I_dash_flat = I_dash.reshape(num_data, -1)

        if A is not None:
            A_flat = A.flatten()
            W_u = W_u[A_flat > 0.5, :]
            I_dash_flat = I_dash_flat[A_flat > 0.5, :]

        num_samples = self.num_samples
        if W_u.shape[0] > num_samples:
            W_u_constraints = shuffle(W_u, random_state=0, n_samples=num_samples)
            I_constraints = shuffle(I_dash_flat, random_state=0, n_samples=num_samples)
        else:
            W_u_constraints = W_u
            I_constraints = I_dash_flat
        return W_u_constraints, I_constraints

    def fit_constraints(self, W_u_constraints, I_constraints):
        """ Fit color/length/width field model for the given constraints.

        Args:
            W_u_constraints: constraints for feature matrix.
            I_constraints: constraints for the target color/length/width.
        """
        self.model.fit(W_u_constraints, I_constraints)
        return

    def predict(self, I_shape=None):
        """ predict color/length/width field using the model.

        Args:
            I_shape: target image size.
        Returns:
            I_tilde: (h, w, d) predicted color/length/width image.
        """
        if I_shape is None:
            I_shape = self.I_shape

        W_u = self.W_u()

        I_tilde_flat = self.model.predict(W_u)
        I_tilde = I_tilde_flat.reshape(I_shape)

        return I_tilde
