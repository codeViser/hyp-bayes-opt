import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from scipy.stats import norm
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        # keep track of all the x, f(x), v(x) we received from the environment
        self.x_rec = np.empty((0,domain.shape[0]), float)
        self.fx_rec = np.empty((0), float)
        self.vx_rec = np.empty((0), float)

        self.speedr_mean = 1.5
        self.speedr_coeff = 1

        # belief of the black box function
        self.gpr_kern = kernels.ConstantKernel(0.5) * kernels.Matern(length_scale=0.5, nu=2.5) + kernels.WhiteKernel(noise_level=0.15)
        self.gpr = GaussianProcessRegressor(kernel=self.gpr_kern, n_restarts_optimizer=10, normalize_y=True, random_state=SEED)

        # belief for the speed function
        self.speedr_kern = kernels.ConstantKernel(np.sqrt(2)) * kernels.Matern(length_scale=0.5, nu=2.5) + kernels.WhiteKernel(noise_level=0.0001)
        self.speedr = GaussianProcessRegressor(kernel=self.speedr_kern, n_restarts_optimizer=10, normalize_y=True, random_state=SEED)
        # pass


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        return self.optimize_acquisition_function()
        # raise NotImplementedError

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        fx_samples = self.gpr.predict(self.x_rec)
        fx_opt = np.max(fx_samples)
        mu1_x, sig1_x = self.gpr.predict(np.atleast_2d(x), return_std=True)
        gamma1_x = float(fx_opt - mu1_x)/sig1_x

        a1_x = sig1_x * (gamma1_x * norm.cdf(gamma1_x) + norm.pdf(gamma1_x))

        speedx_samples = self.speedr.predict(self.x_rec)
        speedx_opt = np.max(speedx_samples)
        speedx_opt += self.speedr_mean
        mu2_x, sig2_x = self.speedr.predict(np.atleast_2d(x), return_std=True)
        mu2_x+= self.speedr_mean
        gamma2_x = float(speedx_opt - mu2_x)/sig2_x

        a2_x = sig2_x * (gamma2_x * norm.cdf(gamma2_x) + norm.pdf(gamma2_x))

        a_x = a1_x*a2_x

        if mu2_x > SAFETY_THRESHOLD:
            return a_x
        else:
            return -1*a_x

    @ignore_warnings(category=ConvergenceWarning)
    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        self.x_rec = np.append(self.x_rec, np.atleast_2d(x), axis=0)
        self.fx_rec = np.append(self.fx_rec, np.atleast_1d(f), axis=0)
        self.vx_rec = np.append(self.vx_rec, np.atleast_1d(v-self.speedr_mean), axis=0)
        # Would update the posterior of the assumptions here to make even \
        # better recommendations using acquisition the next time
        self.gpr.fit(self.x_rec, self.fx_rec)
        self.speedr.fit(self.x_rec, self.vx_rec)
        # raise NotImplementedError

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        fx_filtered = self.fx_rec[self.vx_rec + self.speedr_mean > SAFETY_THRESHOLD]
        x_filtered = self.x_rec[self.vx_rec + self.speedr_mean > SAFETY_THRESHOLD]
        ind = np.argmax(fx_filtered)
        return x_filtered[ind]
        # raise NotImplementedError


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()