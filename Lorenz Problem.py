# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RJ3wk-8vXFXKy4PueoMkO915o4Gbav6A
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
plt.rcParams["font.family"] = "serif"; plt.rcParams["font.size"] = 11

# system parameters
S = SIGMA = 10
R = RHO = 28
B = BETA = 8/3
s0 = S + 100 # initial estimate

# algorithm parameters
t0 = 0
tf = 100 # final time
dt = 0.0001 # forward Euler timestep
dt_obs = 0.1 # how often to observe solution
dt_par = 0.1 # how often to update param
p_tol = 0.0001 # tolerance for parameter switching

@njit
def get_ic(S, R, B):
  dt = 0.0001
  N = int(5/dt)+1

  x, y, z = xp, yp, zp = [60, 59.9, 10]

  for i in range(N):
    x += dt * S*(yp - xp)
    y += dt * (R*xp - yp - xp*zp)
    z += dt * (xp*yp - B*zp)
    xp, yp, zp = x, y, z

  return [x, y, z]

x, y, z = get_ic(S, R, B)

print(x, y, z)

@njit
def solve_lorenz(s0):
  t = np.arange(t0, tf, dt)
  dt_obs_int = int(dt_obs / dt)
  dt_par_int = int(dt_par / dt)

  N = t.size
  sol = np.empty((N, 3))
  ndg = np.empty((N, 3))
  par = np.empty(N)

  sol[0] = get_ic(S, R, B)
  ndg[0] = [0.1, 0.1, 0.1]
  par[0] = s0

  s = s0 # initialize parameter estimate
  for i in range(1, N):
    u = sol[i-1]
    v = ndg[i-1]

    if i % dt_par_int == 0: # update
      if abs(v[1] - u[0]) > p_tol:
        new_s = s * (v[1] - v[0]) / (v[1] - u[0])
        if new_s > 0: s = new_s

    if i % dt_obs_int == 0: # observe data
      v[0] = u[0] # direct insertion

    u_ = np.array([S*(u[1] - u[0]), R*u[0] - u[1] - u[0]*u[2], u[0]*u[1] - B*u[2]])
    v_ = np.array([s*(v[1] - v[0]), R*v[0] - v[1] - v[0]*v[2], v[0]*v[1] - B*v[2]])

    sol[i] = u + dt*u_
    ndg[i] = v + dt*v_
    par[i] = s

  return t, par, sol, ndg

t, par, sol, ndg = solve_lorenz(s0)

plt.figure(figsize=(8,3))
idx = np.arange(0, t.size, 1000) # plot every 1000 timesteps
plt.plot(t[idx], abs(ndg[idx,0]-sol[idx,0]), lw=1, label=r'$|u|$')
plt.plot(t[idx], abs(ndg[idx,1]-sol[idx,1]), lw=1, label=r'$|v|$')
plt.plot(t[idx], abs(ndg[idx,2]-sol[idx,2]), lw=1, label=r'$|w|$')
plt.plot(t, abs(par - S), 'k', lw=1, label=r'$|\tilde{\sigma}-\sigma|$')
plt.title('Evolution of absolute errors'); plt.xlabel('t'); plt.yscale('log')
plt.legend(); plt.grid(); plt.tight_layout()
plt.show()