import numpy as np

###############################################################################
# GENERAL UTILITY FUNCTIONS ###################################################
###############################################################################

###############################################################################
# name: inv
# args: M - a thing to invert
# rets: inverse of M
# desc: a function to abstract inversion for scalars and matrices
###############################################################################
def inv(M):
  return np.linalg.inv(M) if np.size(M) > 1 else 1.0 / M

###############################################################################
# name: solve
# args: A - as in A x = y, solving for x
#       y - as in A x = y, solving for y
# rets: solution A \ y
# desc: performs A \ y, but allows scalars
###############################################################################
def solve(A, y):
  return np.linalg.solve(A, y) if np.size(A) > 1 else y / A

###############################################################################
# name: is_observable
# args: x - state or predicted state
#       t - current time in appropriate units (probably an integer)
#       T - time step
#       Alin_fn - dynamics jacobian wrt to state
#       Clin_fn - observation jacobian wrt to state
# rets: the rank of the observability matrix
# desc: 
###############################################################################
def is_observable(x, t, T, Alin_fn, Clin_fn):
  dim = np.size(x)
  A, C = Alin_fn(x, t, T), Clin_fn(x, t, T)
  OBS = np.stack([C @ np.linalg.matrix_power(A, i) for i in range(dim)], 0)
  return np.linalg.matrix_rank(OBS)

###############################################################################
###############################################################################
###############################################################################



###############################################################################
# FILTERS AND THIER PARAMETER CLASSES #########################################
###############################################################################
#-----------------------------------------------------------------------------#
class EKF_parameters:
  def __init__(self, Alin_fn, Clin_fn):
    self.Alin_fn = Alin_fn
    self.Clin_fn = Clin_fn
###############################################################################
# name: EKF
# args: z - composite vector of actual and predicted state and state covariance
#       t - current time in appropriate units (probably an integer)
#       T - time step
#       f_fn - dynamics function, takes (state, t, T, use_noise)
#       Q - state transition noise covariance matrix
#       obs_fn - observation function, takes (state, t, T, use_noise)
#       R - observation covariance matrix
#       xn - when running from a pre-computed trajectory, the next actual state
#       EKF_params - EKF parameters with Alin_fn and Clin_fn
# rets: composite vector of NEXT actual, predicted state and state covariance
# desc: 
###############################################################################
def EKF(z, t, T, f_fn, Q, obs_fn, R, EKF_params, xn=None, yn=None):
  dim = np.shape(Q)[0]
  x, xt, St = z[0:dim], z[dim:2*dim], z[2*dim:].reshape((dim, dim))
  At, Ct = EKF_params.Alin_fn(xt, t, T), EKF_params.Clin_fn(xt, t, T)

  xn = f_fn(x, t, T, noise=True) if xn is None else xn
  xtn = f_fn(xt, t, T, noise=False)

  yn = obs_fn(xn, t, T, noise=True) if yn is None else yn
  ytn = obs_fn(xtn, t, T, noise=False)

  # predict 
  Stn = At @ St @ At.T + Q

  # update
  xtn = xtn + (Stn @ Ct.T @ solve(Ct @ Stn @ Ct.T + R, yn - ytn)).reshape(-1)
  Stn = Stn - Stn @ Ct.T @ solve(Ct @ Stn @ Ct.T + R, Ct @ Stn)

  return np.concatenate([xn, xtn, Stn.reshape(-1)], 0)
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
class UKF_parameters:
  def __init__(self, alf, bet, lam=None):
    self.alf = alf
    self.bet = bet
    self.lam = lam
###############################################################################
# name: UKF
# args: z - composite vector of actual and predicted state and state covariance
#       t - current time in appropriate units (probably an integer)
#       T - time step
#       f_fn - dynamics function, takes (state, t, T, use_noise)
#       Q - state transition noise covariance matrix
#       obs_fn - observation function, takes (state, t, T, use_noise)
#       R - observation covariance matrix
#       xn - when running from a pre-computed trajectory, the next actual state
#       UKF_params - UKF parameters with alf, bet, lam hyperparameters
# rets: composite vector of NEXT actual, predicted state and state covariance
# desc: 
###############################################################################
def UKF(z, t, T, f_fn, Q, obs_fn, R, UKF_params, xn=None, yn=None):
  assert np.shape(Q)[0] == np.shape(Q)[1]; dim = np.shape(Q)[0]
  x, xt, St = z[0:dim], z[dim:2*dim], z[2*dim:].reshape((dim, dim))

  # propagte actual state
  xn = f_fn(x, t, T, noise=True) if xn is None else xn
  # observe
  yn = obs_fn(xn, t, T, noise=True) if yn is None else yn

  # predict
  L = dim
  bet = UKF_params.bet #bet = 2.0
  alf = UKF_params.alf #alf = 1e-3
  lam = alf**2 * (L + 0) - L if UKF_params.lam is None else UKF_params.lam
  M = np.linalg.cholesky((L + lam) * St)
  xp = xt * np.ones((2 * L + 1, np.size(xt)))
  xp[1:L+1] += M
  xp[L+1:] -= M

  wm = np.ones(2 * L + 1) * lam / (2  * (L + lam))
  wm[0] = lam / (L + lam)
  wm /= np.sum(wm)

  wc = np.ones(2 * L + 1) * lam / (2  * (L + lam))
  wc[0] = lam / (L + lam) + (1 - alf**2 + bet)
  wc /= np.sum(wc)

  xp = f_fn(xp, t, T, noise=True)

  xtn = np.sum(wm.reshape((-1, 1)) * xp, axis=0)
  Stn = (wc.reshape((-1, 1)) * (xp - xtn)).T @ (xp - xtn) + Q * T

  # update
  yp = obs_fn(xp, t, T, noise=True)
  ytn = np.sum(wm * yp)
  Styy = np.sum(wc * (yp - ytn) * (yp - ytn)) + R
  Stxy = np.sum(wc.reshape((-1, 1)) * (xp - xtn) * (yp - ytn).reshape((-1, 1)),
      axis=0)
  K = Stxy * inv(Styy)
  xtn = xtn + K * (yn - ytn)
  St = St - np.outer(K * Styy, K)

  return np.concatenate([xn, xtn, Stn.reshape(-1)], 0)
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
class PF_parameters:
  def __init__(self, particle_nb, sqrtQ):
    self.particle_nb = 10**3
    self.sqrtQ = sqrtQ
###############################################################################
# name: PF
# args: z - composite vector of actual and predicted state and state covariance
#       t - current time in appropriate units (probably an integer)
#       T - time step
#       f_fn - dynamics function, takes (state, t, T, use_noise)
#       Q - state transition noise covariance matrix
#       obs_fn - observation function, takes (state, t, T, use_noise)
#       R - observation covariance matrix
#       xn - when running from a pre-computed trajectory, the next actual state
#       PF_params - PF parameters with particle_nb and sqrtQ
# rets: composite vector of NEXT actual, predicted state and state covariance
# desc: 
###############################################################################
# particle filter
def PF(z, t, T, f_fn, Q, obs_fn, R, PF_params, xn=None, yn=None):
  assert np.shape(Q)[0] == np.shape(Q)[1]; dim = np.shape(Q)[0]
  sqrtQ = PF_params.sqrtQ
  N = int(PF_params.particle_nb * dim)
  x, xt, St = z[0:dim], z[dim:2*dim], z[2*dim:].reshape((dim, dim))

  # propagte actual state
  xn = f_fn(x, t, T, noise=True) if xn is None else xn
  # observe
  yn = obs_fn(xn, t, T, noise=True) if yn is None else yn

  # predict
  xp = np.random.randn(N * dim, dim) @ sqrtQ + xt

  w = np.exp(-0.5 * np.sum(((xp - xt) @ inv(St)) * (xp - xt), axis=1))
  w /= np.sum(w)

  xp = f_fn(xp, t, T, noise=False)

  xtn = np.sum(w.reshape((-1, 1)) * xp, axis=0)
  Stn = (w.reshape((-1, 1)) * (xp - xtn)).T @ (xp - xtn) + Q * T

  # update
  yp = obs_fn(xp, t, T, noise=False)
  ytn = np.sum(w * yp)
  Styy = np.sum(w * (yp - ytn) * (yp - ytn)) + R
  Stxy = np.sum(w.reshape((-1, 1)) * (xp - xtn) * (yp - ytn).reshape((-1, 1)),
      axis=0)
  K = Stxy * inv(Styy)
  xtn = xtn + K * (yn - ytn)
  St = St - np.outer(K * Styy, K)

  return np.concatenate([xn, xtn, Stn.reshape(-1)], 0)
#-----------------------------------------------------------------------------#
###############################################################################
###############################################################################
###############################################################################
