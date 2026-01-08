import numpy as np
import scipy.linalg


class KalmanFilter(object):
    """A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space is ``[x, y, w, h, vx, vy, vw, vh]``,
    containing the bounding box center position (x, y), width (w), height (h),
    and their respective velocities. Object motion follows a constant velocity
    model. The bounding box location (x, y, w, h) is taken as a direct
    observation of the state space (linear observation model).

    Attributes:
        chi2inv95 (dict): A table for the 0.95 quantile of the chi-square
            distribution with N degrees of freedom (N=1,...,9). Used as a
            Mahalanobis gating threshold. Values are from MATLAB/Octave's
            ``chi2inv`` function. This is a class-level attribute.
    """

    chi2inv95 = {  # Documented in class docstring
        1: 3.8415,
        2: 5.9915,
        3: 7.8147,
        4: 9.4877,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919,
    }
    """:meta private:"""

    def __init__(self):
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create track from unassociated measurement.

        Args:
            measurement (np.ndarray): Bounding box coordinates (x, y, w, h)
                with center position (x, y), width w, and height h.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                The mean vector (8 dimensional) and covariance matrix (8x8
                dimensional) of the new track. Unobserved velocities are
                initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean: np.ndarray = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step.

        Args:
            mean (np.ndarray): The 8 dimensional mean vector of the object
                state at the previous time step.
            covariance (np.ndarray): The 8x8 dimensional covariance matrix
                of the object state at the previous time step.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                The mean vector and covariance matrix of the predicted state.
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project state distribution to measurement space.

        Args:
            mean (np.ndarray): The state's mean vector (8 dimensional array).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                The projected mean and covariance matrix of the given state estimate.
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def multi_predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (np.ndarray): The Nx8 dimensional mean matrix of the object
                states at the previous time step.
            covariance (np.ndarray): The Nx8x8 dimensional covariance matrices
                of the object states at the previous time step.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                The mean matrix and covariance matrix of the predicted states.
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = np.array([np.diag(sqr_i) for sqr_i in sqr])
        mean = np.dot(mean, self._motion_mat.T)
        covariance_pred = np.zeros_like(covariance)
        for i in range(mean.shape[0]):
            covariance_pred[i] = self._motion_mat @ covariance[i] @ self._motion_mat.T
        covariance = covariance_pred + motion_cov

        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter correction step.

        Args:
            mean (np.ndarray): The predicted state's mean vector (8 dimensional).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (np.ndarray): The 4 dimensional measurement vector
                (x, y, w, h), where (x, y) is the center position, w the
                width, and h the height of the bounding box.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                The measurement-corrected state distribution (mean and covariance).
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain: np.ndarray = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean: np.ndarray = mean + np.dot(innovation, kalman_gain.T)
        new_covariance: np.ndarray = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from ``chi2inv95``.
        If ``only_position`` is False, the chi-square distribution has 4
        degrees of freedom, otherwise 2.

        Args:
            mean (np.ndarray): Mean vector over the state distribution (8 dimensional).
            covariance (np.ndarray): Covariance of the state distribution (8x8 dimensional).
            measurements (np.ndarray): An Nx4 dimensional matrix of N measurements,
                each in format (x, y, w, h) where (x, y) is the bounding box
                center position, w the width, and h the height.
            only_position (bool, optional): If True, distance computation is done
                with respect to the bounding box center position only.
                Defaults to False.
            metric (str, optional): Metric to use for distance computation.
                Can be "maha" (Mahalanobis) or "gaussian". Defaults to "maha".

        Returns:
            np.ndarray: An array of length N, where the i-th element contains the
                squared Mahalanobis distance (if metric is "maha") or sum of
                squared differences (if metric is "gaussian") between
                (mean, covariance) and ``measurements[i]``.

        Raises:
            ValueError: If an invalid distance metric is provided.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
            )
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError("invalid distance metric")