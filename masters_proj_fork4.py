#TO USE: 
    #UNCOMMENT EITHER UNCONTROLLED PENDULUM CART DYNAMICS, CONTROLLED PENDUKUM CART DYNAMICS, OR CONTROLLED PENDULUM CART DYNAMICS WITH MAX TIME STEP
    #ANIMATION ONLY WORKS PROPERLY WITH THE THE FIRST TWO OPTIONS ABOVE, WITH SETTINGS TAU = 20, INTERVAL =50, AND TIME_PTS = 400, THE ANIMATION MATCHES REAL TIME, I HAVEN'T MADE IT ROBUST TO OTHER SETTINGS
    #FOR THE 2 AND 3 OPTION CAN CHOOSE TO RUN WITH LQR, ARBITRARY POLES, OR NONLINEAR FEEDBACK

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.animation import FuncAnimation
import control

###########DEFINE INTEGRATION FUNCTIONS###############################################
def ctrl1_pen_cart_dyn(t, x, l, g, m1, m2, c, K, M, setpoint): 
    ##########THIS IS FOR CONTROLLER WITH FORCE FEEDBACK BUILT IN##########################
    #####################Adjust state for input into controller###################
    state = x.copy() #make a copy because I only want to change the state for input into the controller not for the integration
    state[0] = np.sign(state[0])*(abs(state[0])%(2*np.pi)) #make angle go from -2pi to 2pi (thought this might help controller stability)
    
    ##################Standard feedback control #############################
    F=np.matmul(M,setpoint)-np.matmul(K,state)
    
    #TRY REMOVING ALL NONLINEAR TERMS FROM THETA DDOT LEAVING ONLY LINEAR LQR SYSTEM
    #kx = np.matmul(K,state)
    #F = (l*(m1+m2*np.sin(state[0])**2)/np.cos(state[0]))*((g*np.sin(state[0])/l)-g*(m1+m2)*state[0]/(l*m1)-c*state[3]/(l*m1)-kx/l*m1)+c*state[3]+m2*g*np.sin(state[0])*np.cos(state[0])-m2*np.sin(state[0])*state[1]**2
    #SYSTEM IS MUCH MORE STABLE WHEN M1!=M2 FOR SOME REASON, DAMPING CONSTANT AND R ALSO HAVE AN EFFECT
    
    print(t)
    xprime = np.zeros(len(x))
    xprime[0] = x[1]
    xprime[1] = (g/l)*np.sin(x[0]) - (np.cos(x[0])/l) *  ((F+m2*np.sin(x[0])*(-g*np.cos(x[0])+l*x[1]**2)-c*x[3])/(m1+m2*np.sin(x[0])**2))
    xprime[2] = x[3]
    xprime[3] = (F+m2*np.sin(x[0])*(-g*np.cos(x[0])+l*x[1]**2)-c*x[3])/(m1+m2*np.sin(x[0])**2)
    return xprime

def ctrl2_pen_cart_dyn(t, x, l, g, m1, m2, c, F): 
    #######THIS IS FOR CONTROLLER WITH FORCE AS AN EXTERNAL PARAMETER############
    xprime = np.zeros(len(x))
    xprime[0] = x[1]
    xprime[1] = (g/l)*np.sin(x[0]) - (np.cos(x[0])/l) *  ((F+m2*np.sin(x[0])*(-g*np.cos(x[0])+l*x[1]**2)-c*x[3])/(m1+m2*np.sin(x[0])**2))
    xprime[2] = x[3]
    xprime[3] = (F+m2*np.sin(x[0])*(-g*np.cos(x[0])+l*x[1]**2)-c*x[3])/(m1+m2*np.sin(x[0])**2)
    return xprime

def pen_cart_dyn(t, x, l, g, m1, m2, c):   
    ########THIS IS FOR UNCONTROLLED CART-PENDULUM DYNAMICS################
    F=0
    xprime = np.zeros(len(x))
    xprime[0] = x[1]
    xprime[1] = (g/l)*np.sin(x[0])- (np.cos(x[0])/l) *  ((F+m2*np.sin(x[0])*(-g*np.cos(x[0])+l*x[1]**2)-c*x[3])/(m1+m2*np.sin(x[0])**2))
    xprime[2] = x[3]
    xprime[3] = (F+m2*np.sin(x[0])*(-g*np.cos(x[0])+l*x[1]**2)-c*x[3])/(m1+m2*np.sin(x[0])**2)
    return xprime

#Pendulum Constant Parameters
l = 1  #pendulum length in controller
g = 9.8     #gravity
m1 = 1   #cart mass
m2 = 1      #mass on pendulum
c = 1    #damping coefficient

#System matrices
A = np.array([[0,1,0,0],[g*(m1+m2)/(l*m1), 0, 0, c/(l*m1)],[0,0,0,1],[-m2*g/m1, 0, 0, -c/m1]])   #linearized dynamics matrix about vertical position
B = np.array([[0],[-1/(l*m1)], [0], [1/m1]])                                                    #linearized b input vector  

#Compute Full State Feedback Controller Values for a chosen set of Poles          
A_sq = np.matmul(A,A)       #precalculate A^2
A_cu = np.matmul(A_sq,A)    #precalculate A^3
omega_c = np.concatenate((B, np.matmul(A,B), np.matmul(A_sq,B),np.matmul(A_cu,B)),axis=1) #controllability matrix 
val,vect = sp.linalg.eig(A)         #compute eigenvalues of A matrix to get the poles/roots of the system
a_vals = np.poly(val)       #compute polynomial coefficients from roots to get "a" values
W = np.zeros([4,4])                 #define W matrix for defining T matrix
W[0,:] = np.flip(a_vals[0:4])
W[1,:] = np.append(np.flip(a_vals[0:3]), 0)  
W[2,:] = np.append(np.flip(a_vals[0:2]), [0,0])
W[3,:] = np.append(np.flip(a_vals[0:1]), [0,0,0])  
Tinv = np.matmul(omega_c,W)   #define T matrix which represents state transformation between current form and canonical controllability form 
T = np.linalg.inv(Tinv)         # invert Tinv to get T
poles = np.array([-3,-3,-3,-3]) # select pole/eigenvalues of choice for controlled system
alpha_vals = np.poly(poles)     #compute polynomial with those roots
Kc = np.flip(alpha_vals[1:5]-a_vals[1:5])   #compute Kc by difference between alpha values and a values,K values are numbered opposite alpha and a
K = np.matmul(Kc,T)                 #convert K from controllability matrix form system back to existing system
K = K.reshape(1,-1)                 #convert K from row to column or vice versa, i dont recall 

#####Compute LQR gains using built in function###################
Q = np.array([[1,0,0,0], [0, 1000, 0,0], [0,0,1,0], [0,0,0,1]])  #state error cost matrix for lqr
R = 1                               #control effort penalty for lqr
K, S, E = control.lqr(A, B, Q, R)   #use control library to find desired K                        

#Compute feedforward gains to adjust position setpoint properly
system = A-np.matmul(B,K)
kappa = np.matmul(np.linalg.inv(system),B)
M = np.matmul(np.linalg.inv(np.matmul(np.transpose(kappa),kappa)),np.transpose(kappa))
##########################################################################################################################################





#######RUN PENDULUM WITH NO CONTROLLER#############################################################
# tau = 20   #end time of integration
# interval = 50 #ms for each frame in animation
# time_pts = 400 #get results from 0 to 20s at 400 points
# frames = time_pts   #how many frames to give to animation
# tspan = [0,tau]  #time range to integrate over
# y0 = np.array([.2,0,0,0]) #initial state
# sol = sp.integrate.solve_ivp(pen_cart_dyn, tspan, y0, method='BDF', args=(l, g, m1, m2, c), t_eval=np.linspace(0,tau,round(time_pts)))
#########################################################################################################






# # ####################Run Control System at same rate as dynamics simulation, and allow solve_ivp to set time step################
tau = 20
interval = 50 #ms for each frame in animation
time_pts = 400 #get results from 0 to 20s at 400 points #this results in animation matching real time speed
frames = time_pts           #number of frames to animate
tspan = [0,tau]              #interval to integrate the equations
y0 = np.array([.2,0,1,0])  #initial state
setpoint = [0,0,0,0]    #setpoint, for the ability to change the equilibrium position of the pendulum-cart
sol = sp.integrate.solve_ivp(ctrl1_pen_cart_dyn, tspan, y0, method='RK45', args=(l, g, m1, m2, c, K, M, setpoint), t_eval=np.linspace(0,tau,round(time_pts)))

#Get LQR force history for plotting
F_history = -np.matmul(K,sol.y)

#Get nonlinear force history for plotting
#F_history = np.zeros([len(sol.t)])
# for b in range(0,len(sol.t)):
#     state = sol.y[:,b]
#     kx = np.matmul(K,state)
#     F = (l*(m1+m2*np.sin(state[0])**2)/np.cos(state[0]))*((g*np.sin(state[0])/l)-g*(m1+m2)*state[0]/(l*m1)-c*state[3]/(l*m1)-kx/l*m1)+c*state[3]+m2*g*np.sin(state[0])*np.cos(state[0])-m2*np.sin(state[0])*state[1]**2
#     F_history[b] = F
##############################################################################








#####Run control system slower than dynamics while enforcing a maximum allowed time step size###################################
# tau = 1 #number of seconds to simulate
# frames = 400 #frames to animate
# interval = int(1000*(tau/frames)) #time in ms each frame is shown in the animation
# dt = .0001  # time step for dynamics sim in seconds
# controldt = 1 #update frequency of control system in multiples of dt
# y0 = np.array([3.14,0,0,0]) #intial condition
# setpoint = [0,0,0,0]        #initial setpoint
# timesteps = round(tau/dt)  #number of timesteps
# solution = np.zeros([4,timesteps+1])  #variable to store solution
# solution[:,0] = y0  #store initial condition

# for i in range(0, timesteps):
    
#     if i%controldt == 0: #run control loop every controldt time steps
    
#         #Adjust theta state so that it goes from 0-2pi into controller
#         state = y0.copy()
#         state[0] = np.sign(state[0])*(abs(state[0])%(2*np.pi)) #make angle go from -2pi to 2pi
        
#         kx = np.matmul(K,state)
#         F = (l*(m1+m2*np.sin(state[0])**2)/np.cos(state[0]))*((g*np.sin(state[0])/l)-g*(m1+m2)*state[0]/(l*m1)-c*state[3]/(l*m1)-kx/l*m1)+c*state[3]+m2*g*np.sin(state[0])*np.cos(state[0])-m2*np.sin(state[0])*state[1]**2
#         #Basic LQR feedback
#         #F=np.matmul(K,-state)
            
#         #print(F)
#         print(i)

#     sol = sp.integrate.solve_ivp(ctrl2_pen_cart_dyn, [i*dt, i*dt+dt], y0, method='RK45', args=(l, g, m1, m2, c, F), t_eval=np.linspace(i*dt,i*dt+dt,2))
#     yn_1 = y0
#     y0 = sol.y[:,1]
#     solution[:,i+1] = sol.y[:,1]
   
# sol.y = solution
# sol.t = np.linspace(0,tau,timesteps)

# #Get LQR force history for plotting
# # F_history = -np.matmul(K,sol.y)

# #Get nonlinear force history for plotting
# F_history = np.zeros([len(sol.t)+1])
# for b in range(0,len(sol.t)+1):
#     state = sol.y[:,b]
#     kx = np.matmul(K,state)
#     F = (l*(m1+m2*np.sin(state[0])**2)/np.cos(state[0]))*((g*np.sin(state[0])/l)-g*(m1+m2)*state[0]/(l*m1)-c*state[3]/(l*m1)-kx/l*m1)+c*state[3]+m2*g*np.sin(state[0])*np.cos(state[0])-m2*np.sin(state[0])*state[1]**2
#     F_history[b] = F



#Throw away extra frames, only use for animation
#sol.y = sol.y[:,::interval]
#sol.t = sol.t[::interval]



##########Make Plots of state variables####################################
fig = plt.figure()
t = np.linspace(0,tau,len(sol.y[0,:]))
plt.plot(t,sol.y[0,:],label ="theta") 
plt.plot(t,sol.y[1,:],label ="theta_dot") 
plt.plot(t,sol.y[2,:],label ="x")
plt.plot(t,sol.y[3,:],label ="x_dot") 
#plt.plot(t,np.transpose(F_history),label = 'force')
plt.xlabel('Time (s)')
plt.ylabel('Rad,Rad/s,Meters,Meters/s')

plt.legend()
plt.title('State Variables')
plt.show

fig = plt.figure()
plt.plot(t,np.transpose(F_history))
plt.title('Force on Cart Over Time ')
plt.xlabel('Time (s)')
plt.ylabel('Newtons')
plt.show

##########################  ANIMATE SYSTEM  ##################################
fig = plt.figure()
ax = plt.axes(xlim=(-5,5), ylim=(-5, 5))
line1, = ax.plot([], [],'o-', lw=1)
line2, = ax.plot([], [], 'o-', lw=1)
line3, = ax.plot([], [], lw=.5)
line4, = ax.plot([], [], lw=.2)

def init():
    return line1,line2,line3,line4

def animate(i):

    x1 = [sol.y[2,i]+l*np.sin(sol.y[0,i])]
    y1 = [l*np.cos(sol.y[0,i])]
    line1.set_data(x1,y1)
    x2 = [sol.y[2,i]]
    y2 =[0]
    line2.set_data(x2,y2)
    x3 = np.linspace(x1,x2)
    y3 = np.linspace(y1,y2)
    line3.set_data(x3,y3)
    x4 = np.linspace(-50,50)
    y4 = np.zeros(np.shape(x4))
    line4.set_data(x4,y4)
    return line1,line2,line3,line4

anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, blit=True)
plt.show()

###############################EXTRA CODE##################################

#################################### ###MSE LOOP
# tot = 0
# for l in range(0, len(sol.t)+1):
#     diffsq = np.sum((sol1[:,l]-sol2[:,l])**2)
#     tot = diffsq+tot
# tot = tot/len(sol.t)
# print(tot)  
    
###################################################################################################################
#compute cost from cost function
# cost = 0
# for ind in range(0, len(sol.y[1,:])):
#     x1 = np.matmul(Q+np.matmul(np.transpose(K),K)*R, sol.y[:,ind])
#     x2 = np.matmul(np.transpose(sol.y[:,ind]),x1)
#     cost = cost+x2
# print(cost)
