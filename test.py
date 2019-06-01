###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import time
from filters import *
###############################################################################


###############################################################################
dim = 3
Q = 0.1 * np.eye(dim)
sqrtQ = np.linalg.cholesky(Q)
R = 0.1
sqrtR = np.sqrt(R)
T = 1e-1
S0 = 1e-2 * np.eye(dim)
mu0 = np.zeros(dim)
###############################################################################


###############################################################################
u = lambda x, t: np.array([1, np.sin(t)])
def f(x, t, T, noise=True):
  if len(np.shape(x)) == 1:
    px, py, th = x
    v, phi = u(x, t)
    xn = np.array([px + T * v * np.cos(th), py + T * v * np.sin(th), th + T *
      phi])
    xn += noise * np.sqrt(T) * sqrtQ @ np.random.randn(np.size(x))
  elif len(np.shape(x)) == 2:
    px, py, th = x[:, 0], x[:, 1], x[:, 2]
    v, phi = u(x, t)
    xn = np.stack([px + T * v * np.cos(th), py + T * v * np.sin(th), th + T *
      phi], 1)
    xn += noise * np.sqrt(T) * np.random.randn(*np.shape(x)) @ sqrtQ
  else:
    raise ValueError("Only 1D or batch (2D) input supported for x")
  return xn

def obs(x, t, T, noise=True):
  if len(np.shape(x)) == 1:
    px, py, th = x
    y = np.sqrt(px**2 + py**2) + noise * sqrtR * np.random.randn()
    #y = px - py + 3 * th + noise * T * sqrtR * np.random.randn()
  elif len(np.shape(x)) == 2:
    px, py, th = x[:, 0], x[:, 1], x[:, 2]
    y = (np.sqrt(px**2 + py**2) + noise * sqrtR *
        np.random.randn(*np.shape(px)))
    #y = px - py + 3 * th + noise * T * sqrtR * np.random.randn(*np.shape(px))
  else:
    raise ValueError("Only 1D or batch (2D) input supported for x")
  return y

def Alin(x, t, T):
  px, py, th = x
  v, _ = u(x, t)
  return np.array([
    [1, 0, -T * v * np.sin(th)],
    [0, 1, T * v * np.cos(th)],
    [0, 0, 1]])

def Clin(x, t, T):
  px, py, th = x
  return np.array([[px, py, 0]]) / np.sqrt(px**2 + py**2)
  #norm = np.sqrt(px**2 + py**2)
  #return np.array([[px / norm, py / norm, 1]])
  #return np.array([[1, -1, 3]])
###############################################################################


# main routine ################################################################
if __name__ == "__main__":
  x0 = mu0 + sqrtQ @ np.random.randn(dim)
  xt0 = np.array([1, 1, 1])
  z0 = np.concatenate([x0, x0, S0.reshape(-1)], 0)

  tf = 40
  tspan = np.linspace(0, tf, int(tf / T))
  x = x0
  x_list = [x]
  y_list = [obs(x, tspan[0], T, noise=True)]
  observable_list = [is_observable(x, tspan[0], T, Alin, Clin)]
  for i in range(len(tspan) - 1):
    x = f(x, tspan[i], T, noise=True)
    y = obs(x, tspan[i], T, noise=True)
    x_list.append(x)
    y_list.append(y)
    observable_list.append(is_observable(x, tspan[i + 1], T, Alin, Clin))

  FILTERS = [EKF, UKF, PF]
  PARAMS = [EKF_parameters(Alin, Clin), UKF_parameters(1e-3, 2.0),
      PF_parameters(10**3, np.linalg.cholesky(Q))]
  NAMES = ["EKF", "UKF", "PF"]
  timef = [0.0 for _ in range(dim)]
  zf_list = [[z0] for i in range(dim)]
  for d in range(dim):
    z = z0
    t1 = time.time()
    for i in range(len(tspan) - 1):
      z = FILTERS[d](z, tspan[i], T, f, Q, obs, R, PARAMS[d], 
        xn=x_list[i + 1], yn=y_list[i + 1])
      zf_list[d].append(z)
    timef[d] = time.time() - t1
  print(timef)

  Zf = [np.stack(zf_list[d], 0) for d in range(dim)]

  COLORS = ["red", "black", "blue"]
  for d in range(dim):
    plt.figure(figsize=[10, 6])
    for i in range(3):
      plt.plot(tspan, Zf[d][:, i], color=COLORS[i], linestyle='-')
      plt.plot(tspan, Zf[d][:, i + dim], color=COLORS[i], linestyle=':')
    plt.legend(["$p_x$", "$\\tilde{p}_x$", "$p_y$", "$\\tilde{p}_y$",
      "$\\theta$", "$\\tilde{\\theta}$"])
    plt.title(NAMES[d])
    plt.xlabel("Time, t (1)")
    #plt.savefig(NAMES[d] + ".png", dpi=200)
  #plt.figure()
  #plt.plot(tspan, observable_list)
  plt.show()
###############################################################################
