import time
import math

# Numpy and Scipy
import scipy as sc
import numpy as np

import scipy.integrate as si

import scipy.linalg as sl
import numpy.linalg as nl

from numpy import pi as pi
from scipy.optimize import fsolve as fsolve

# Matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

pylab.rcParams['figure.figsize'] = (8, 5)
pylab.rcParams['figure.dpi'] = 96
pylab.rcParams['savefig.dpi'] = 96

# Units: Imperial

#-------------------------------- General parameters 
#- Parameters at sea level: air density and thrust
TSL = 1000
rho0 = 1.10

#- Initial conditions: speed, angle of attack, 
#  Euler angle and air density at given altitude
V0 = 305
alpha0 = 0.
theta0 = 0.
zE0  = 0.
rho = 0.00238

#- Mass and inertia properties
mass = 746
Iy = 65000
gravity = 32.2

#-------------------------------- Stability derivatives
# These derivatives represent how the aerodynamic forces 
# change due to the flight dynamics unknowns. For example, 
# Zw represents the change in the vertical force due to the 
# vertical velocity w

#- Derivatives - Z direction
Zu = -159.64
Zu_dot = 0.

Zw = -328.24
Zw_dot = 0

Zq = 0
Zq_dot = 0

Zdelta = -16502.0

#- Derivatives - X direction
Xu = -26.26
Xu_dot = 0.

Xw = 79.82 
Xw_dot = 0

Xq = 0 
Xq_dot = 0

Xdelta = 0

#- Derivatives - Rotation
Mu = 0
Mu_dot = 0

Mw = -1014.0
Mw_dot = -36.4

Mq = -18135.0
Mq_dot = 0

Mdelta = -303575.0

#-------------------------------- Simulation options
# We are going to simulate the first 60 seconds. The solution to
# the differential equations is calculated each 0.1 seconds

simTime = [0, 600]
simStep = 0.01

# Horizontal and vertical velocity at equilibrium condition 
Ue = V0*np.cos(alpha0)
We = V0*np.sin(alpha0)

# Initial conditions. We're going to simulate an initial disturbance
# in the angular velocity q
state_vector0 = [0, 0, 0.1, 0]
elevator0 = 0
throttle0 = 0 

def getMatrices(Ue, We):
    # This function return the Jacobian matrix of the system 
    # and the other vectors related to control surfaces and thrust 
    # that we need for the integration
    
    # ---------
    I1 = np.array([[mass-Xu_dot, -Xw_dot, -Xq_dot, 0.],
                   [-Zu_dot, mass-Zw_dot, -Zq_dot, 0.],
                   [-Mu_dot, -Mw_dot, Iy-Mq_dot, 0.],
                   [      0,       0,        0, 1]])
    
    In = nl.inv(I1)

    
    # ---------
    An = np.array([[Xu, Xw, Xq, 0],
                   [Zu, Zw, Zq, 0],
                   [Mu, Mw, Mq, 0],
                   [0,0,0,0]])

    Kn = np.array([[0, 0, - mass * We, - mass * gravity * np.cos(theta0)],
                   [0, 0, + mass * Ue, - mass * gravity * np.sin(theta0)],
                   [0, 0, 0, 0],
                   [0, 0, 1, 0]])

    A = An + Kn
    
    # ---------
    D = np.array([Xdelta, Zdelta, Mdelta, 0])
    T = np.array([TSL * rho / rho0, 0, 0, 0])
    
    return np.dot(In, A), np.dot(In, D), np.dot(In, T)

Jac, Jac_D, Jac_T = getMatrices(V0, 0.)

print("Jacobian matrix = ")
print(Jac)
print

print("Eigenvalue analysis: ")
evals = nl.eig(Jac)[0]
for ev in evals:
    print("\t", ev)
print

#Standard ODE integration

def integrate(y0, tSpan, dt, update=None, callback=None, maxInnerSteps=10000):
    '''
    Integrate a set of first order differential equation. This function acts
    similarly to MATLAB's ode45 function.

    t, y, timing = integrate(y0, tSpan, dt, update, callback, maxInnerSteps)

    INPUT
      y0 = initial conditions (y at tBeg)
      tSpan = [tBeg, tEnd]
      dt = step size

      update = fn(t, y)
      callback = fn(t, y) called at every succeful integration

      maxInnerSteps = max number of inner iterations (default: 10000)

    OUTPUT
      t = list of timesteps
      y = solutions at timestep (one line per timestep)
      timing = elapsed time for the integration of the timestep
    '''

    tBegin = min(tSpan)
    tEnd = max(tSpan)

    history_t = []
    history_y = []

    integr = sc.integrate.ode(update)
    integr.set_integrator('dopri5', nsteps=maxInnerSteps)
    integr.set_initial_value(y0, tBegin)

    t_elapsed = []
    tic = time.time()

    history_t.append(tBegin)
    history_y.append(np.array(y0))

    while integr.t < tEnd:
        values = integr.integrate(integr.t + dt)

        if not integr.successful():
            raise Exception("Integration not successful")

        history_t.append(integr.t)
        history_y.append(values)

        if callback is not None:
            try:
                callback(integr.t, values)
            except:
                print("Error in the callback function")

        toc = time.time() - tic
        t_elapsed.append(toc)

    return np.array(history_t), np.array(history_y), np.array(t_elapsed)

def eqsOfMotion(time, state_vector, elevator, throttle, Ue, We):
    """
    This function return the first order derivative of the state vector at each timestep.
    The derivative is calculated using the Jacobian matrix evaluated earlier.
    """
    
    if np.any(np.isnan(state_vector)):
        raise ValueError("NaN!")
    
    Jac, Jac_D, Jac_T = getMatrices(Ue, We)

    state_vector_dot = np.dot(Jac, state_vector) \
        + Jac_D * elevator \
        + Jac_T * throttle 
           
    return state_vector_dot

print("Initial state vector:")
print(state_vector0)
print

print("Elevator = ", elevator0 * (180./np.pi))
print("Throttle = ", throttle0)
    
fn_integrate = lambda t, x: eqsOfMotion(t, x, elevator0, throttle0, Ue, We)
t_time, t_values, t_elapsed = integrate(np.array(state_vector0), simTime, simStep, fn_integrate, None)

print("Integration performed!")

names = ["u", "w", "q", "theta"]
titles = ["Horizontal speed", "Vertical speed", "Angular speed", "Rotation"]
units = ["[ft/s]", "[ft/s]", "[deg/s]", "[deg]"]
convertion = [1, 1, (180./np.pi), (180./np.pi)]

plt.figure()

for index in range(4):

    plt.figure(index +1)
    plt.plot(t_time, t_values[:,index]*convertion[index])
    plt.ylabel("{} {}".format(names[index], units[index]))
    plt.xlabel("Time [s]")
    plt.title(titles[index])
    plt.grid('on')

plt.show()

# The derivatives of the horizontal and vertical position in the inertial frame are calculated
xE_dot = +(Ue + t_values[:,0]) * np.cos(t_values[:,3]) + (We + t_values[:,1]) * np.sin(t_values[:,3])
zE_dot = +(Ue + t_values[:,0]) * np.sin(t_values[:,3]) - (We + t_values[:,1]) * np.cos(t_values[:,3])

# When these derivatives are needed at time steps not sampled, we have to use interpolation. 
# This is needed because the integration routine can choose time steps for which no evaluation of
# xE_dot or zE_dot is available
def int_path(t, x):
    xE_dot_temp = np.interp(t, t_time, xE_dot)
    zE_dot_temp = np.interp(t, t_time, zE_dot)
    return np.array([xE_dot_temp, zE_dot_temp])
    
path_time, path_values, _ = integrate(np.array([0,0]), simTime, simStep, int_path, None)

plt.figure()
plt.plot(path_values[:,0], path_values[:,1])
plt.xlabel('Horizontal translation [ft]')
plt.ylabel('Vertical translation [ft]')
plt.title('Flight path')
plt.grid()
plt.show()



