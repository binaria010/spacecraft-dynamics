"""
THis library contains ...

author: Juliana Osorio
date: 08/31/20
"""

import numpy as np
from scipy.linalg import *
from DCMmatrix import *
import math



# Some constants:
R_m = 3396.19  #Radius of Mars in km
mu = 42828.3  # Mars gravity constant in km^3/s^2
rgmo = 20424.2 #in km the orbit radius of a GMO 
rlmo = 3796.19 #in km the orbit radius of an LMO
hGMO = 17028.010000000002
hLMO = 400
Omega_GMO, i_GMO, theta0_GMO = 0, 0, 250  # in degrees. initial attitude for the mother satellite 
Omega_LMO, i_LMO, theta0_LMO = 20,30,60 # in degrees. Initial attitude for the nano-satellite 

"""
theta_dot = constant orbit rate of LMO (Low Mars Orbit ) orbit  theta_dot = sqrt{mu/r^3}in rad/sec where r is the radius orbit r= R_m + h with h the altitude of the satellite.
"""

def orbit_radius(h) :
    
    return R_m + h

def theta_dot(h):
    """
    this function computes the constant rate theta_dot of an LMO for a given radius orbit r = R_m +h
    
    Parameters
    ----------
    h : float --> altitude of the satellite
    
    Returns
    ---------
    theta_dot: float--> rate of the orbit in deg/sec
    """
    r = orbit_radius(h)
    return np.sqrt(mu/(r**3))

def pos_and_vel(h,euler_angles0,t):
    """
    This function computes the inertial position and velocity vectors at time t for a given altitude and initial 3-1-3 Euler angles 
    """
    #h = r - R_m
    r = orbit_radius(h)
    theta0 = euler_angles0[2]
    
    theta = (180/np.pi)*theta_dot(h)*t + theta0  # in deg
    euler_angles = (euler_angles0[0], euler_angles0[1], theta)
    C = DCMatrix(euler_angles,(3,1,3))

    r_N = r*C[0,:]
    r_dot_N = theta_dot(h)*r*C[1,:]
    
    return r_N, r_dot_N

def DCM_HN(t) : 
    
    h_LMO = 400
    (Omega_LMO, i_LMO, theta0_LMO) = (20, 30, 60) # in deg
    r_LMO, r_dot_LMO = pos_and_vel(h_LMO, (Omega_LMO, i_LMO, theta0_LMO), t)
    i_r = r_LMO/norm(r_LMO)
    
    i_h = np.cross(r_LMO, r_dot_LMO)
    i_h = i_h/norm(i_h)
    
    i_theta = np.cross(i_h,i_r)
    
    C = np.zeros((3,3))
    
    
    C[0,:] = i_r
    C[1,:] = i_theta
    C[2,:] = i_h
    
    return C

def R_sN() :
    """
    computes the DCM matrix from N to R_s, where R_s is the sun reference frame (the frame where the spacecraft solar panel axis b_3 points towards the sun, i.e n_2 direction)
    """
    
    R_s = np.array([[-1, 0, 0],[0,0,1], [0, 1, 0]])  # sun reference
    #H = DCM_HN(t)
    ## H^T is the matrix whose columns are the basis vectors of the Hill frame.
    #R_sH = R_s @ H.T
    
    #R_sN = R_sH@ H
    
    return R_s

def N_w_RsN() :
    
    return np.zeros(3)

## Nadir Pointing Orientation

def R_nN(t) :
    """
    computes the DCM matrix from N to R_n, where R_n is the nadir reference frame (the frame where the spacecraft sensor axis b_1 points towards the center of Mars)
    """
    HN = DCM_HN(t) # this gives me the matrix [i_r i_theta i_h]
    
    RnH = np.array([[-1, 0, 0],[0, 1, 0], [0, 0, -1]])
    
    RnN = RnH@HN
    
    return RnN

def N_w_RnN(t) :
    """
    computes the angular velocity w_{Rn/N} in the frame N:
    N^w_{RnN} = N^w_{Rn/H} + N^w_{H/N}  but N^w_{Rn/H}=0 then 
    N^w_{RnN} = N^w_{H/N} = (HN)^T ^Hw_{H/N} with ^Hw_{H/N} = thetadot*i_h
    
    """
    h = 400  # the altitude of the satellite (LMO orbit)
    HN = DCM_HN(t)
    RnN = R_nN(t)
    Rn_w_HN = -theta_dot(h)*np.array([0,0,1]) # angular vel of H/N in the Rn frame is -thetdot*r_3= thetadot*i_h in the H frame
    N_w_RnN = RnN.T@ Rn_w_HN  # angular vel of H/N in the N frame
    
    return N_w_RnN

## GMO Pointing orientation

def R_cN(t):
    
    """
    This function computes the DCM matrix of the GMO pointing orientation frame R_c
    """
    r_GMO, _ = pos_and_vel(hGMO, (Omega_GMO, i_GMO, theta0_GMO), t)
    r_LMO, _ = pos_and_vel(hLMO, (Omega_LMO, i_LMO, theta0_LMO), t)
    Delta_r = r_GMO - r_LMO
    r_1 = -Delta_r/norm(Delta_r,2)
    r_2 = np.cross(Delta_r, np.array([0,0,1]))
    r_2 = r_2/norm(r_2,2)
    r_3 = np.cross(r_1, r_2)
    
    RcN = np.zeros((3,3))
    RcN[0,:] = r_1
    RcN[1,:] = r_2
    RcN[2,:] = r_3
    
    return RcN

def N_w_RcN(t) :
    
    """
    This function computes the angular velocity of frame R_c relative to N in the N frame. For this we use the formula: RcNdot = -tilde(W)@RcN
    """
    
    dt = 0.001
    RcNdot = (R_cN(t + dt) - R_cN(t))/dt
    tildeW = -RcNdot@(R_cN(t).T)
    w = np.array([tildeW[2,1], tildeW[0,2], tildeW[1,0]])  # this gives w in the Rc frame 
    
    w = R_cN(t).T @ w
    
    return w
    

    
### Attitude Error Evaluation

def sigma_b_r(s_BN, s_RN):
    """
    This function computes the orientation of the body relative to the reference frame R:
    
    Parameters
    ---------
    s_BN: 1-d np.array: orientation sigma_B/N (orientation of body frame B relative to inertial N)
    s_RN: 1-d np.array: orientation sigma_R/N (orientation of reference frame R relative to inertial N)
    
    Returns
    -------
    s_BR : the orienatation of the body relative to reference
    """
    BN = MRP_to_DCM(s_BN)
    RN = MRP_to_DCM(s_RN)
    BR = BN@(RN.T)
    s_BR = DCM_to_MRP(BR)
    
    return BR,s_BR



def tracking_errors(s_BN, B_w_BN, RN, N_w_RN) :
    
    """
    this function computes the associated tracking errors s_{B/R} and B_w_{B/R} at some time t given at this time the values of: attitude states s_{B/N}, the angular velocity B_w_{B/N}, the dcm matrix RN and rates N_w_{R/N}
    
    Parameters
    ----------
    s_BN : 1-d numpy array-> attitude orientation of body relative to N
    B_w_BN : 1-d numpy array -> angular velocity of body frame B relative to N in the B frame. In radians!
    RN : 3x3 numpy array -> The DCM matrix from N frame to reference frame R
    N_w_RN : 1-d numpy array -> angular velocity of frame R relative to N in the N frame. In radians 
    
    Returns
    -------
    s_BR : 1-d numpy array -> attitude error
    B_w_BR : 1-2 numpy array -> rate error in radians!!!
    """
    
    s_RN = DCM_to_MRP(RN)  # this is the attitude of R relative to N, computed from the DCM matrix RN

    BN = MRP_to_DCM(s_BN) # DCM matrix of attitude s_BN
    
    BR,s_BR = sigma_b_r(s_BN, s_RN) # DCM matrix and attitude s_B/R
    
    #for the rates: B^w_B/R = B^w_B/N - B^w_R/N 
    B_w_RN = BN@ N_w_RN  # ang vel w_R/N in frame B in radians
    
    B_w_BR = B_w_BN - B_w_RN  # all in radians
    
    return s_BR, B_w_BR


##  Numerical Attitude Simulator

def order4(function, tval, x, step, *param):
    """
    this function computes the fourth order approximation of a solution of a differential equation whose r.h.s is given by function
    
    Parameters
    ---------
    function: a function that returns the r.h.s of the differential equation
    tval: double. time at which function is evaluated
    x: numpy array. Initial condition
    step: float. The delta t of the grid in the time interval.
    """
    
    k1 = function(tval, x, *param)
    k2 = function(tval + 0.5*step, x + 0.5*step*k1, *param)
    k3 = function(tval + 0.5*step, x + 0.5*step*k2, *param)
    k4 = function(tval + step, x + step*k3, *param)
    
    x = x +step*(k1 + 2*k2 +2*k3 + k4)/6
    
    return x

def RG4(function, tspan, x0, step, *param) :
    
    """
    this function implements a Runge-Kutta of order 4 to integrate a diff equation xdot = function, in the interval  tspan with time step = step.
    Returns a matrix whose columns are the history time of function x
    """
    t = np.arange(tspan[0],tspan[1] +step , step)
    nt = len(t)
    
    # initialization
    x = x0
    X = np.zeros((len(x0),nt))
    X[:,0] = x
    
    for i in range(1,nt) :
        
        x = order4(function, t[i], x, step, *param)
        if norm(x[0:3],2) >= 1:
            s = norm(x[0:3],2)**2
            x[0:3] = -x[0:3]/s
            
        X[:,i] = x
        
    return X

def w_BN_dot(t,w0, u, I):
    """
    computes the r.hs of the diff equ. for w_b/N_dot as a function of the control 
    """
    rhs = inv(I)@((-Matrix_tilde(w0) @ I) @ w0 + u)
    return rhs

def MRP_dot(t, sigma0, w):
    """
    computes r.h.s f the differential eq for sigma_B/N
    
    Parameters
    ----------
    t: double. time t
    sigma0: numpy array. initial condition for sigma_B/N
    w: numpy array. tha angular velocity w_B/N at time t
    
    Returns
    --------
    rhs: numpy array of size= w.shape. The r.h.s of the kinematic differential equation of the MRP's
    """
    B_sigma = 0.25*((1- norm(sigma0,2)**2)*np.eye(3) +2*Matrix_tilde(sigma0) +2*np.outer(sigma0,sigma0))
    rhs = B_sigma @ w
    return rhs

def xdot(t, x0, u, I):
    """
    this function computes the right hand side of Xdot for the state X = (s_BN, w_BN)
    """
    
    sigma0 = x0[0:3]
    w0 = x0[3:]
    rhs1 = MRP_dot(t, sigma0, w0)
    rhs2 = w_BN_dot(t, w0, u, I)
    rhs = np.hstack((rhs1, rhs2)) 
    
    return rhs

def controlPD(s_br, delw, k, P):
    
    u = -k*s_br - P*delw
    
    return u



def control_simulation(function, tspan, x0, reference, *param) :
    I = np.diag([10, 5, 7.5])
    P= 0.16666666666666666
    k = 0.005555555555555555
    step = 0.1
    t = np.arange(tspan[0],tspan[1], step)
    nt = len(t)
    
    # initialization
    x = x0
    X = np.zeros((len(x0),nt))
    X[:,0] = x
    
    if reference == 'sun' :
        # tracking errors: 
        RN = R_sN()
        N_w_RN = N_w_RsN()
        s_br, delw = tracking_errors(x[0:3], x[3:], RN, N_w_RN) 

    if reference == 'nadir' :
        # tracking errors:
        RN = R_nN(t[0])
        N_w_RN = N_w_RnN(t[0])
        s_br, delw = tracking_errors(x[0:3], x[3:], RN, N_w_RN)

    if reference == 'GMO' :
        # tracking errors:

        RN = R_cN(t[0])
        N_w_RN = N_w_RcN(t[0])
        s_br, delw = tracking_errors(x[0:3], x[3:], RN, N_w_RN)

    # constant control law every 1 second 
    u = controlPD(s_br, delw, k, P)
    
    U = np.zeros((3,nt))
    U[:,0] = u
    S_BR = np.zeros((3,nt))
    S_BR[:,0] = s_br
    
    for i in range(1,nt) :
            
        x = order4(function, t[i], x, step, u, I)
        
        if norm(x[0:3],2) >= 1:
            s = norm(x[0:3],2)**2
            x[0:3] = -x[0:3]/s
            
        X[:,i] = x
        
        if i%10 == 0 :
            if reference == 'sun' :
                # tracking errors: 
                RN = R_sN()
                N_w_RN = N_w_RsN()
                s_br, delw = tracking_errors(x[0:3], x[3:], RN, N_w_RN) 

            if reference == 'nadir' :
                # tracking errors:
                RN = R_nN(t[i])
                N_w_RN = N_w_RnN(t[i])
                s_br, delw = tracking_errors(x[0:3], x[3:], RN, N_w_RN)

            if reference == 'GMO' :
                # tracking errors:

                RN = R_cN(t[i])
                N_w_RN = N_w_RcN(t[i])
                s_br, delw = tracking_errors(x[0:3], x[3:], RN, N_w_RN)

            # constant control law every 1 second 
            u = controlPD(s_br, delw, k, P)
        U[:,i] = u
        S_BR[:,i] = s_br        
        
    return X,t,U, S_BR

def angle(v,w):
    a = math.acos(np.dot(v,w)/(norm(v)*norm(w)))*180/np.pi
    return a

def Is_sunlit(t):
    h = hLMO
    r, _ = pos_and_vel(h, (20,30,60),t)
    return r[1] >= 0

def comm_pos(t) :
    pos_lmo, _ = pos_and_vel(hLMO,(20,30, 60), t)
    pos_gmo, _ = pos_and_vel(hGMO, (0,0,250),t)
    return angle(pos_lmo, pos_gmo) <= 35

def switching_times():
    t = 0        
    while Is_sunlit(t) == True:
        t += 1
    tshade  = t
     
    tscience = tshade
    while not comm_pos(tscience) :
        tscience = tscience + 1
    
    tcom = tscience
    while comm_pos(tcom)==True and Is_sunlit(tcom)==False:
        tcom = tcom +1
        if tcom > 6500 :
            break
        
    if  Is_sunlit(tcom) == False:
        tscience2 = tcom
        while Is_sunlit(tscience2) == False :
            tscience2 += 1
            if tscience2 >6500 :
                break
    tsun = tscience2
    while Is_sunlit(tsun) ==True:
        tsun += 1
        if tsun> 6500:
            break
    
    return tshade, tscience, tcom, tscience2, tsun





def control_spacecraft() :


    # parameters for the simulation:
    #step = 0.01
    I = np.diag([10, 5, 7.5])
    tshade, tscience, tcom, tscience2, tsun = 1918, 3057, 4067, 5469, 6501
    s_BN0 = np.array([0.3, -0.4, 0.5])
    B_w_BN0 = np.array([1.0, 1.75, -2.20])*np.pi/180 #in radians. Initial condition for w_BN
    x0 = np.hstack((s_BN0, B_w_BN0)) # initial condition for state x


    X_sun, _ , U_sun, _ = control_simulation(xdot,(0,tshade),x0, 'sun', I)
    
    x1 = X_sun[:,-1] # new initial condition
    
    X_nadir, _ , U_nadir, _ = control_simulation(xdot, (tshade, tscience), x1, 'nadir', I)
    
    x2 = X_nadir[:,-1] # new initial condition
    
    X_comm, _ , U_comm, _ = control_simulation(xdot, (tscience,tcom), x2, 'GMO', I)
    
    x3 = X_comm[:,-1] # new initial condition
    
    X_nadir2, _ , U_nadir2, _ = control_simulation(xdot, (tcom, tscience2), x3, 'nadir', I)
    
    x4 = X_nadir2[:,-1] # new initial condition
    
    X_sun2, _, U_sun2, _ = control_simulation(xdot, (tscience2, 6500), x4, 'sun', I)
    
    X = np.hstack((X_sun, X_nadir, X_comm, X_nadir2, X_sun2))
    
    return X, X_sun, X_nadir, X_comm, X_nadir2, X_sun2
            
    
    
    
    

