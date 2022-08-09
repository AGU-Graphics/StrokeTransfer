# MIT License
#
# Copyright (c) 2022  Hideki Todo, Kunihiko Kobayashi, Jin Katsuragi, Haruna Shimotahira, Shizuo Kaji, Yonghao Yue
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

from util.logger import getLogger
from util.normalize.norm import normalize_vector_image

logger = getLogger(__name__)

CANONICAL_NAMES = ["$I_{\\parallel}$", "$I_{\\perp}$",
                   "$N^V_{\\parallel}$", "$N^V_{\\perp}$",
                   "$o^{s}_{\\parallel}$", "$o^{s}_{\\perp}$"]

FEATURE_NAMES = ['$I_d(p)$', '$I_s(p)$', '$I_{\\nabla_2}(p)$',
                 '$K(p)$', '$H(p)$', '$D_S (p)$',
                 '$N_0$', '$N_1$', '$N_2$']



class VectorFieldRegressionModel:
    """ Regression model for vector field.

    Attributes:
        order: order of the vector field regression model.
        model: sklearn.linear_model.BayesianRidge for the linear regression.
        canonical_sections: canonical section data.
        features: proxy feature data.
        polynomial_features: sklearn.preprocessing.PolynomialFeatures class for the target order.
    """

    def __init__(self, order=1):
        """

        Args:
            order: order of the vector field regression model.
        """
        logger.debug(f"order={order}")
        self.order = order
        self.model = BayesianRidge()

        self.canonical_sections = {}
        self.canonical_section_shape = None
        self.features = {}

        self.polynomial_features = PolynomialFeatures(order)

    def set_orientation(self, key, canonical_section):
        """ Set canonical section with the given key.

        Args:
            key: canonical section id.
            canonical_section: (h, w, d) image data.
        """
        self.canonical_sections[key] = canonical_section
        self.canonical_section_shape = canonical_section.shape

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
            Init the large internal attribute (canonical_sections, features).
        """
        self.canonical_sections = {}
        self.features = {}

    def set_alpha(self, A):
        """ Set alpha mask to get the target region.

        Args:
            A: (h, w) alpha mask image.
        """
        self.A = A

    def A_hat(self):
        """ Compute canonical section matrix A_hat.

        Returns:
            A_hat: (d*h*w, N_A) canonical section matrix.
        """
        A_hat = []

        for key, A_i in self.canonical_sections.items():
            A_hat.append(A_i.flatten())

        A_hat = np.array(A_hat)
        A_hat = A_hat.T

        return A_hat

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

    def A_hat_W_u(self):
        """ Compute A_hat W_u matrix for linear regression model.

        Returns:
            AW: (d*h*w, N_A*#poly(N_F, order)) A_hat W_u matrix for linear regression model.
        """
        A_hat = self.A_hat()

        if self.order == 0:
            return A_hat

        W = self.W_u()

        AW = []

        for j in range(A_hat.shape[1]):
            for k in range(W.shape[1]):
                W_k = np.repeat(W[:, k], 4).flatten()
                AW.append(W_k * A_hat[:, j])

        AW = np.array(AW).T
        return AW

    def constraints(self, u_dash, A=None):
        """ Return the model constraints for multi-exemplars.

        Args:
            u_dash: (h, w, d) target orientation image.
            A: (h, w) alpha mask image.

        Returns:
            AW_constraints: constraints for A_hat W_u matrix.
            u_constraints: constraints for the target orientation.
        """
        AW_constraints = self.A_hat_W_u()
        logger.debug(f"AW.shape={AW_constraints.shape}")

        if A is not None:
            A_flat = np.repeat(A.flatten(), 4).flatten()
            u_dash_flat = u_dash.flatten()
            return AW_constraints[A_flat > 0.5, :], u_dash_flat[A_flat > 0.5]
        else:
            return AW_constraints, u_dash.flatten()

    def fit_constraints(self, AW_constraints, u_constraints):
        """ Fit vector field model for the given constraints.

        Args:
            AW_constraints: constraints for A_hat W_u matrix.
            u_constraints: constraints for the target orientation.

        """
        self.model.fit(AW_constraints, u_constraints)
        phi = self.model.coef_
        self.phi = phi
        return

    def fit(self, u_dash, A=None):
        """ Fit vector field model for the target orientations.

        Args:
            u_dash: (h, w, d) target orientation image.
            A: (h, w) alpha mask image.
        """
        AW_constraints, u_constraints = self.constraints(u_dash, A)
        self.fit_constraints(AW_constraints, u_constraints)
        u_fit = self.predict()

        return u_fit, self.phi

    def predict(self):
        """ predict vector field using the model.

        Returns:
            u_tilde: (h, w, d) predicted orientation image.
        """
        AW = self.A_hat_W_u()

        u_flat = self.model.predict(AW)

        u = u_flat.reshape(self.canonical_section_shape)
        u[:, :, :3] = normalize_vector_image(u[:, :, :3])

        return u

    def compute_weight_map(self):
        """ Compute weight map for visualizing local weight distribution.

        Returns:
            weight_map: (h, w) weight image data in weight_map[key].
        """
        phi = self.phi
        A = self.A

        phi = phi.reshape(len(self.canonical_sections.items()), -1)
        weight_map = {}

        canonical_sections = self.canonical_sections
        features = self.features

        for i, c_key in enumerate(canonical_sections.keys()):
            w_sum = np.zeros_like(A)
            for j, f_key in enumerate(features.keys()):
                Xj = features[f_key]
                Wi = phi[i, j + 1] * Xj
                w_sum += Wi

            w_sum += phi[i, 0]
            weight_map[c_key] = w_sum
        return weight_map

    def model_matrix(self):
        """ Return the model matrix for 1st-order model.

        Returns:
            phi: (N_A, N_F + 1) matrix data.

        Note:
            Only support 1st-order mode.
        """
        phi = self.phi
        phi = phi.reshape(len(self.canonical_sections.items()), -1)

        canonical_keys = self.canonical_sections.keys()
        feature_keys = self.features.keys()

        phi_matrix = np.zeros((len(CANONICAL_NAMES), len(FEATURE_NAMES) + 1))

        for i, canonical in enumerate(canonical_keys):
            k = CANONICAL_NAMES.index(canonical)
            phi_matrix[k, -1] = phi[i, 0]
            for j, feature in enumerate(feature_keys):
                l = FEATURE_NAMES.index(feature)
                phi_matrix[k, l] = phi[i, j + 1]

        return phi_matrix


def vf_scale_with_alpha(V, A):
    V_ = np.array(V)

    for ci in range(3):
        V_[:, :, ci] *= A
    return V_


def proj_vf(u, N):
    udN = np.einsum("ijk,ijk->ij", u[:, :, :3], N[:, :, :3])
    u_proj = u - np.einsum("ij,ijk->ijk", udN, N)
    u_proj[:, :, 3] = u[:, :, 3]
    return u_proj