import numpy as np
import scipy as sp
import scipy.linalg
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)


def estRun(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement):
    # In this function you implement your estimator. The function arguments
    # are:
    #  time: current time in [s] 
    #  dt: current time step [s]
    #  internalStateIn: the estimator internal state, definition up to you. 
    #  steeringAngle: the steering angle of the bike, gamma, [rad] 
    #  pedalSpeed: the rotational speed of the pedal, omega, [rad/s] 
    #  measurement: the position measurement valid at the current time step
    #
    # Note: the measurement is a 2D vector, of x-y position measurement.
    #  The measurement sensor may fail to return data, in which case the
    #  measurement is given as NaN (not a number).
    #
    # The function has four outputs:
    #  est_x: your current best estimate for the bicycle's x-position
    #  est_y: your current best estimate for the bicycle's y-position
    #  est_theta: your current best estimate for the bicycle's rotation theta
    #  internalState: the estimator's internal state, in a format that can be understood by the next call to this function

    # Example code only, you'll want to heavily modify this.
    # this internal state needs to correspond to your init function:
    r=0.425
    B=0.8
    # V=np.diag([0.004,0.001])
    V=np.diag([0.004,0.002])
    W=np.array([[1.05,1.478],[1.478,2.873]])
    V_mean = np.zeros((2,1))
    W_mean = np.array([[0],[0]])
    Last_state = np.array([[internalStateIn[0]],[internalStateIn[1]],[internalStateIn[2]]])
    Pm = internalStateIn[4]
    myColor = internalStateIn[3]

    if (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        # x = internalStateIn[0] + 5 * (r) * (pedalSpeed) * np.cos(internalStateIn[2])*dt
        # y = internalStateIn[1] + 5 * (r) * (pedalSpeed) * np.sin(internalStateIn[2])*dt
        # theta = internalStateIn[2] + 5 * (r) * (pedalSpeed) * np.tan(steeringAngle)*dt/(B)

        tmp_state = np.concatenate((Last_state,V_mean,W_mean),axis=0) 
        tmp_var = scipy.linalg.block_diag(Pm, V, W)
        s_new=np.zeros((14,7,1))
        s_xp=np.zeros((14,3,1))
        for i in range(7):
            s_new[i]=tmp_state+ (scipy.linalg.sqrtm(7*tmp_var))[:,i,np.newaxis]
            s_new[i+7]=tmp_state- (scipy.linalg.sqrtm(7*tmp_var))[:,i,np.newaxis]
        for i in range(14):
            s_xp[i,0,0] = s_new[i,0,0] + 5 * (r) * (pedalSpeed+s_new[i,3,0]) * np.cos(s_new[i,2,0])*dt
            s_xp[i,1,0] = s_new[i,1,0] + 5 * (r) * (pedalSpeed+s_new[i,3,0]) * np.sin(s_new[i,2,0])*dt
            s_xp[i,2,0] = s_new[i,2,0] + 5 * (r) * (pedalSpeed+s_new[i,3,0]) * np.tan(steeringAngle+s_new[i,4,0])*dt/(B)
        xp=np.mean(s_xp,axis=0)
        Pp=np.zeros((3,3))
        for i in range(14):
            Pp += (s_xp[i]-xp)@ (s_xp[i]-xp).T/14
        x = xp[0,0]
        y= xp[1,0]
        theta=xp[2,0]
        Pm=Pp



    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        
        tmp_state = np.concatenate((Last_state,V_mean,W_mean),axis=0) 
        tmp_var = scipy.linalg.block_diag(Pm, V, W)
        s_new=np.zeros((14,7,1))
        s_xp=np.zeros((14,3,1))
        for i in range(7):
            # print((scipy.linalg.sqrtm(9*tmp_var))[:,i,np.newaxis])
            s_new[i]=tmp_state+ (scipy.linalg.sqrtm(7*tmp_var))[:,i,np.newaxis]
            s_new[i+7]=tmp_state- (scipy.linalg.sqrtm(7*tmp_var))[:,i,np.newaxis]
        # print((scipy.linalg.sqrtm(7*tmp_var))[:,i,np.newaxis])
        for i in range(14):
            s_xp[i,0,0] = s_new[i,0,0] + 5 * (r) * (pedalSpeed+s_new[i,3,0]) * np.cos(s_new[i,2,0])*dt
            s_xp[i,1,0] = s_new[i,1,0] + 5 * (r) * (pedalSpeed+s_new[i,3,0]) * np.sin(s_new[i,2,0])*dt
            s_xp[i,2,0] = s_new[i,2,0] + 5 * (r) * (pedalSpeed+s_new[i,3,0]) * np.tan(steeringAngle+s_new[i,4,0])*dt/(B)
        xp=np.mean(s_xp,axis=0)
        Pp=np.zeros((3,3))
        
        for i in range(14):
            Pp += (s_xp[i]-xp)@ (s_xp[i]-xp).T/14
        # print(Pp)
        s_z=np.zeros((14,2,1))
        for i in range(14):
            s_z[i,0,0]= s_xp[i,0,0]+0.5*(B)*np.cos(s_xp[i,2,0])+s_new[i,5,0]
            s_z[i,1,0]= s_xp[i,1,0]+0.5*(B)*np.sin(s_xp[i,2,0])+s_new[i,6,0]

        zp=np.mean(s_z,axis=0)
        Pzz=np.zeros((2,2))
        for i in range(14):
            Pzz += (s_z[i]-zp)@ (s_z[i]-zp).T/14
        Pxz=np.zeros((3,2))
        for i in range(14):
            Pxz += (s_xp[i]-xp)@ (s_z[i]-zp).T/14

        K=Pxz@ np.linalg.inv(Pzz)
        z=np.array([[measurement[0]],[measurement[1]]])
        xm=xp+K@(z-zp)
        Pm=Pp-K@Pzz@K.T
        x = xm[0,0]
        y= xm[1,0]
        theta=xm[2,0]
        # print(K)
        

    #we're unreliable about our favourite colour: 
    if myColor == 'green':
        myColor = 'red'
    else:
        myColor = 'green'


    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function
    # at next run), must obviously be compatible with the format of
    # internalStateIn:
    internalStateOut = [x,
                     y,
                     theta, 
                     myColor,
                     Pm
                     ]

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut 


