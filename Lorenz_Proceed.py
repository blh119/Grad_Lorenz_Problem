 # -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 19:32:51 2023

@author: holli
"""

import numpy as np
import matplotlib.pyplot as plt

sigma = 10.
beta = 8/3
rho = 28.

mu0 = 100.
mu2 = 100. 


def linear_space(start, end, steps):
    
    step_length = (end - start) / steps

    even_spaced_nums = [start]
    
    # keeps track if we have reached 
    tracker = start
    
    # add values to the even_spaced_nums list
    while tracker < end:
        tracker = tracker + step_length
        even_spaced_nums.append(tracker)
        
    return np.array(even_spaced_nums) # return whole list as numpy array


def get_ic(x0, y0, z0): 
    
    dt = 0.0001 
    N = int(5/dt)+1
    
    x, y, z = xp, yp, zp = [x0, y0, z0]
        
    for i in range(N):     
        
        x += dt * sigma*(yp - xp)
        y += dt * (rho*xp - yp - xp*zp)
        z += dt * (xp*yp - beta*zp)
        xp, yp, zp = x, y, z
        
    return x, y, z


def get_ic_list(x0, y0, z0): 
    
    dt = 0.0001 
    N = int(10/dt)+1
    x, y, z = xp, yp, zp = [x0, y0, z0]

    x_list = [x]
    y_list = [y]
    z_list = [z]
        
    for i in range(N):     
        
        x += dt * sigma*(yp - xp)
        y += dt * (rho*xp - yp - xp*zp)
        z += dt * (xp*yp - beta*zp)
        xp, yp, zp = x, y, z

        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

    return x_list, y_list, z_list


def solve_lorenz_split_1(time_step, tf, init_x, init_y, init_z, last_sol_list, last_ndg_list, counter):
    
    dt = 0.0001
    
    sol = np.empty((time_step, 3))   
    ndg = np.empty((time_step, 3))
    g = np.empty(time_step)
    
    if counter == 0:
        
        init_x, init_y, init_z = get_ic(60., 59.9, 10.)
        
        sol[0] = np.array([sigma*(init_y - init_x), 
                           rho*init_x - init_y - init_x*init_z,
                           init_x*init_y - beta*init_z])
        
        
        ndg[0] = np.array([sigma*(init_y - init_x), 
                           rho*init_x - init_y - 0., # zero is just a placeholder
                           init_x*init_y - beta*init_z])
        g[0] = 0.
        
    else:
        
        sol[0] = np.array([last_sol_list[0,0], last_sol_list[0,1], last_sol_list[0,2]])
        ndg[0] = np.array([last_ndg_list[0,0], last_ndg_list[0,1], last_ndg_list[0,2]])
        g[0] = -last_ndg_list[0,0]*last_ndg_list[0,2]
        
    
    print("Time Final: ", tf, "Time Step: ", time_step)
    
    for i in range(1, time_step):
        
        u = sol[i-1]
        v = ndg[i-1]
        
        # set g value
        if counter == 0:
            g_ = 0.
            
        else:
    
            last_ndg = last_ndg_list[i-1]
            g_ = -last_ndg[0]*last_ndg[2]
        
        u_ = np.array([sigma*(u[1]-u[0]), rho*u[0]-u[1]-u[0]*u[2], u[0]*u[1]-beta*u[2]])
    
        # unknown solution
        v_ = np.array([sigma*(v[1]-v[0])-mu0*(v[0]-u[0]), rho*v[0]-v[1]+g_, v[0]*v[1]-beta*v[2]-mu2*(v[2]-u[2])])
        #v_ = f_ + np.array([, , 0])
        
        sol[i] = u + (dt*u_)
        ndg[i] = v + (dt*v_)
        g[i] = g_
    
    return [sol, ndg, g]
        
def nudged_system(tf, step):
    
    time_steps = np.arange(0, tf + 1, step)[::-1]
    time_steps = time_steps[time_steps != 0]
    
    sol_full_set = []
    ndg_full_set = []
    g_full_set = []
    
    # initial run
    
    counter = 0 # this helps keep track of our correct initial position
    
    list_indexer = int(time_steps[0]-time_steps[1])
    
    for time_step in time_steps:
        
        if counter == 0:
            
            init_x, init_y, init_z = get_ic(60., 59.9, 10.)
            
            current_sol, current_ndg, current_g = solve_lorenz_split_1(time_step, tf, init_x, init_y, init_z, np.nan, np.nan, counter)
            
            sol_full_set.append(current_sol)
            ndg_full_set.append(current_ndg)
            g_full_set.append(current_g)
            
            counter = counter + 1
            
        else:
            
            last_sol = sol_full_set[counter-1][list_indexer:]
            last_ndg = ndg_full_set[counter-1][list_indexer:]
            
            print("Time Step: ", time_step, "last ndg size: ", len(last_ndg))
            
            current_sol, current_ndg, current_g = solve_lorenz_split_1(time_step, tf, init_x, init_y, init_z, last_sol, last_ndg, counter)
            
            sol_full_set.append(current_sol)
            ndg_full_set.append(current_ndg)
            g_full_set.append(current_g)
            
            counter = counter + 1
            
        if counter == 1:
            return sol_full_set, ndg_full_set, g_full_set
        
    return sol_full_set, ndg_full_set, g_full_set


def get_endpoints(sol, ndg):
    
    sol_endpoints = []
    ndg_endpoints = []
    
    for i in range(len(sol)):
        sol_endpoints.append(sol[i][len(sol[i])-1])
        
    for i in range(len(ndg)):
        ndg_endpoints.append(ndg[i][len(sol[i])-1])
        
    return np.array(sol_endpoints), np.array(ndg_endpoints)

def plot_absolute_errors(sol_endpoints, ndg_endpoints, g_full_set):
    
    abs_u = abs(sol_endpoints[:,0]-ndg_endpoints[:,0])
    abs_v = abs(sol_endpoints[:,1]-ndg_endpoints[:,1])
    abs_w = abs(sol_endpoints[:,2]-ndg_endpoints[:,2])
    
    t = np.arange(0, len(abs_u))
    
    plt.figure(figsize=(8,3))
    plt.plot(t, abs_u, lw=1, label=r'$|u|$') 
    plt.plot(t, abs_v, lw=1, label=r'$|v|$')
    plt.plot(t, abs_w, lw=1, label=r'$|w|$')
    #plt.plot(self.t, self.delta_S, 'k', lw=1, label=r'$|\tilde{\sigma}-\sigma|$')
    plt.title('Evolution of absolute errors'); plt.xlabel('t'); plt.yscale('log')
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.show()

sol, ndg, g = nudged_system(100000, 1000)
sol_endpoints, ndg_endpoints = get_endpoints(sol, ndg)
plot_absolute_errors(sol_endpoints, ndg_endpoints, g)


plt.plot(np.arange(100000), sol[0][:, 0], label = "x")
plt.plot(np.arange(100000), sol[0][:, 1], label = "y")
plt.plot(np.arange(100000), sol[0][:, 2], label = "z")
plt.legend()
plt.show()
            
            
plt.plot(np.arange(91000), sol[9][:, 0], label = "x")
plt.plot(np.arange(91000), sol[9][:, 1], label = "y")
plt.plot(np.arange(91000), sol[9][:, 2], label = "z")
plt.legend()
plt.show()           
            
x_list, y_list, z_list = get_ic_list(60., 59.9, 10.)


plt.plot(np.arange(40000, 50002), x_list[40000:], label = "x")
plt.plot(np.arange(40000, 50002), y_list[40000:], label = "y")
plt.plot(np.arange(40000, 50002), z_list[40000:], label = "z")
plt.legend()
plt.show()   
        

plt.plot(np.arange(80000, 100002), x_list[80000:], label = "x")
plt.plot(np.arange(80000, 100002), y_list[80000:], label = "y")
plt.plot(np.arange(80000, 100002), z_list[80000:], label = "z")
plt.legend()
plt.show()  

plt.plot(np.arange(100002), x_list, label = "x")
plt.plot(np.arange(100002), y_list, label = "y")
plt.plot(np.arange(100002), z_list, label = "z")
plt.legend()
plt.show()


plt.plot(np.arange(1,101), sol_endpoints[:, 0], label = "x")
plt.plot(np.arange(1,101), sol_endpoints[:, 1], label = "y")
plt.plot(np.arange(1,101), sol_endpoints[:, 2], label = "z")
plt.legend()
plt.show()  
        
        