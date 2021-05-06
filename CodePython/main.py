
# This is the main function, which will initilize your estimator, and run it using data loaded from a text file. 
#

## Changes:
# April 05: 
# - print run number to help keep track. 
# - corrected final error output to be in [-pi, pi]

import numpy as np
import matplotlib.pyplot as plt
from estRun import estRun
from estInitialize import estInitialize

#provide the index of the experimental run you would like to use.
# Note that using "0" means that you will load the measurement calibration data.
experimentalRun = 5

print('Loading the data file #', experimentalRun)
experimentalData = np.genfromtxt ('../data/run_{0:03d}.csv'.format(experimentalRun), delimiter=',')

#===============================================================================
# Here, we run your estimator's initialization
#===============================================================================
print('Running the initialization')
internalState, studentNames, estimatorType = estInitialize()

numDataPoints = experimentalData.shape[0]

#Here we will store the estimated position and orientation, for later plotting:
estimatedPosition_x = np.zeros([numDataPoints,])
estimatedPosition_y = np.zeros([numDataPoints,])
estimatedAngle = np.zeros([numDataPoints,])

print('Running the system')
dt = experimentalData[1,0] - experimentalData[0,0]
for k in range(numDataPoints):
    t = experimentalData[k,0]
    gamma = experimentalData[k,1]
    omega = experimentalData[k,2]
    measx = experimentalData[k,3]
    measy = experimentalData[k,4]
    
    #run the estimator:
    x, y, theta, internalState = estRun(t, dt, internalState, gamma, omega, (measx, measy))

    #keep track:
    estimatedPosition_x[k] = x
    estimatedPosition_y[k] = y
    estimatedAngle[k] = theta
    

print('Done running')
#make sure the angle is in [-pi,pi]
estimatedAngle = np.mod(estimatedAngle+np.pi,2*np.pi)-np.pi

posErr_x = estimatedPosition_x - experimentalData[:,5]
posErr_y = estimatedPosition_y - experimentalData[:,6]
angErr   = np.mod(estimatedAngle - experimentalData[:,7]+np.pi,2*np.pi)-np.pi

print('Final error: ')
print('   pos x =',posErr_x[-1],'m')
print('   pos y =',posErr_y[-1],'m')
print('   angle =',angErr[-1],'rad')

ax = np.sum(np.abs(posErr_x))/numDataPoints
ay = np.sum(np.abs(posErr_y))/numDataPoints
ath = np.sum(np.abs(angErr))/numDataPoints
score = ax + ay + ath
if not np.isnan(score):
    #this is for evaluation by the instructors
    print('average error:')

    print('   pos x =', ax, 'm')
    print('   pos y =', ay, 'm')
    print('   angle =', ath, 'rad')

    #our scalar score. 
    print('average score:',score)

#===============================================================================
# make some plots:
#===============================================================================
#feel free to add additional plots, if you like.
print('Generating plots')

figTopView, axTopView = plt.subplots(1, 1)
axTopView.plot(experimentalData[:,3], experimentalData[:,4], 'rx', label='Meas')
axTopView.plot(estimatedPosition_x, estimatedPosition_y, 'b-', label='est')
axTopView.plot(experimentalData[:,5], experimentalData[:,6], 'k:.', label='true')
axTopView.legend()
axTopView.set_xlabel('x-position [m]')
axTopView.set_ylabel('y-position [m]')

figHist, axHist = plt.subplots(5, 1, sharex=True)
axHist[0].plot(experimentalData[:,0], experimentalData[:,5], 'k:.', label='true')
axHist[0].plot(experimentalData[:,0], experimentalData[:,3], 'rx', label='Meas')
axHist[0].plot(experimentalData[:,0], estimatedPosition_x, 'b-', label='est')
# print(np.cov((estimatedPosition_x,estimatedPosition_y)))
# print(np.mean(estimatedPosition_x),np.mean(estimatedPosition_y))

axHist[1].plot(experimentalData[:,0], experimentalData[:,6], 'k:.', label='true')
axHist[1].plot(experimentalData[:,0], experimentalData[:,4], 'rx', label='Meas')
axHist[1].plot(experimentalData[:,0], estimatedPosition_y, 'b-', label='est')

axHist[2].plot(experimentalData[:,0], experimentalData[:,7], 'k:.', label='true')
axHist[2].plot(experimentalData[:,0], estimatedAngle, 'b-', label='est')

axHist[3].plot(experimentalData[:,0], experimentalData[:,1], 'g-', label='m')
axHist[4].plot(experimentalData[:,0], experimentalData[:,2], 'g-', label='m')

axHist[0].legend()

axHist[-1].set_xlabel('Time [s]')
axHist[0].set_ylabel('Position x [m]')
axHist[1].set_ylabel('Position y [m]')
axHist[2].set_ylabel('Angle theta [rad]')
axHist[3].set_ylabel('Steering angle gamma [rad]')
axHist[4].set_ylabel('Pedal speed omega [rad/s]')

print('Done')
plt.show()

