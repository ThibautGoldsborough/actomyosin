

import fenics as fenics

import numpy as np


import os

import matplotlib.pyplot as plt


#Turn off plots:
plt.ioff()
fenics.set_log_level(40)


#Turn on plots
#plt.ion()

T = 6 # final time

num_steps = 2000 # number of time steps 

dt = T / num_steps
 




class PeriodicBoundary(fenics.SubDomain):
         # Left boundary is "target domain" G
         def inside(self, x, on_boundary):
             return bool(- fenics.DOLFIN_EPS < x[0] < fenics.DOLFIN_EPS and on_boundary)

         def map(self, x, y):
             y[0] = x[0] - 1


class Boundary(fenics.SubDomain):
    #def inside(self, x, on_boundary):
     #   tol = 1E-14
      #  return on_boundary and fenics.near(x[0], 0, tol)
    
    def inside(self, x, on_boundary):
        return bool(- fenics.DOLFIN_EPS < x[0] < fenics.DOLFIN_EPS and on_boundary)

    
    
    
mesh = fenics.IntervalMesh(10000,0.0,1.0)
vector_element = fenics.FiniteElement('P',fenics.interval,1)
single_element = fenics.FiniteElement('P',fenics.interval,1)
vector_element_rho_activity = fenics.FiniteElement('P',fenics.interval,1)
mixed_element = fenics.MixedElement(vector_element,single_element,vector_element_rho_activity)
V = fenics.FunctionSpace(mesh, mixed_element)
    
   
k=0.5

periodic_boundary_condition = Boundary()

boundary=Boundary()


bc_l = fenics.DirichletBC(V, fenics.Constant((0,0,0)), 'x[0] < DOLFIN_EPS')
bc_r = fenics.DirichletBC(V, fenics.Constant((0,0,1)), 'x[0] > 1- DOLFIN_EPS')

bcs=[bc_l,bc_r]

v,r,ra = fenics.TestFunctions(V)
full_trial_function = fenics.Function(V)
u, rho, rho_activity = fenics.split(full_trial_function)

full_trial_function_n = fenics.Function(V)
u_n, rho_n, rho_activity_n = fenics.split(full_trial_function_n)

u_initial = fenics.Constant(0.0)
rho_initial = fenics.Expression('1/k0', degree=2,k0 = k)
#rho_activity_initial=fenics.Expression('1/(0.1+x[0])/10', degree=2)
rho_activity_initial=fenics.Expression('1-1*x[0]', degree=2)


u_n = fenics.interpolate(u_initial, V.sub(0).collapse())
rho_n = fenics.interpolate(rho_initial, V.sub(1).collapse())
rho_activity_n = fenics.interpolate(rho_activity_initial, V.sub(2).collapse())


np.random.seed(0)
rho_n.vector().set_local(np.array(rho_n.vector())+1.0*(0.5-np.random.random(rho_n.vector().size())))
fenics.assign(full_trial_function_n, [u_n,rho_n,rho_activity_n])
u_n, rho_n, rho_activity_n = fenics.split(full_trial_function_n)




fenics.method=2
Z1=fenics.Constant(-14.5)
z2=fenics.Constant(0.001)

Z1=fenics.Constant(-8.5)
z2=fenics.Constant(0.1)


Z1=fenics.Constant(-11.5)
z2=fenics.Constant(0.1)


b=fenics.Constant(6)

ku0=fenics.Constant(0.5)
alpha=fenics.Constant(1)
kb=fenics.Constant(1)
d = fenics.Constant(0.14)
alpha = fenics.Constant(1.0)
c=fenics.Constant(0.05)
k = fenics.Constant(0.5)

eta=fenics.Constant(0.7)

def K(rho,rho_n,rho_activity):
    return ((Z1*rho)/(1+z2*rho)*rho_activity)


def chi(n):
    value_list=[1,1,0.001,0.0001]
    return fenics.Constant(value_list[n])





F = (-u_n*v*fenics.dx+ u*v*fenics.dx
     
     -rho_activity_n*ra*fenics.dx + rho_activity*ra*fenics.dx
     
    -dt*K(rho,rho_n,rho_activity)*chi(0)*v.dx(0)*fenics.dx
    +dt*(b+(K(rho,rho_n,rho_activity))*rho_activity*c*chi(1))*u.dx(0)*v.dx(0)*fenics.dx
    -dt*K(rho,rho_n,rho_activity)*c*c*chi(2)/2.0*u.dx(0)*u.dx(0)*v.dx(0)*fenics.dx
    +dt*K(rho,rho_n,rho_activity)*c*c*c*chi(3)/6.0*u.dx(0)*u.dx(0)*u.dx(0)*v.dx(0)*fenics.dx    
    +eta*u.dx(0)*v.dx(0)*fenics.dx-eta*u_n.dx(0)*v.dx(0)*fenics.dx
      
    -rho_n*r*fenics.dx+rho*r*fenics.dx
    +rho*u_n*r.dx(0)*fenics.dx-rho*u*r.dx(0)*fenics.dx 
      
     
    +dt*d*rho.dx(0)*r.dx(0)*fenics.dx
    +dt*k*fenics.exp(alpha*u.dx(0))*rho*r*fenics.dx
    -dt*r*fenics.dx
    +dt*c*u.dx(0)*r*fenics.dx)

     




iters=0
t = 0






actin,myosin=[],[]

rho_activity_grad=[]


plt.figure()

# problem = fenics.NonlinearVariationalProblem(F,full_trial_function,J=fenics.derivative(F,full_trial_function))

    
# solver = fenics.NonlinearVariationalSolver(problem)
# stype = 'newton'
# solver.parameters['nonlinear_solver']=stype
# sprms = solver.parameters[stype+'_solver']

# # Set maximum iterations:
# sprms['maximum_iterations'] = 100



for n in range(num_steps):
  
    # Update current time
    t += dt
    
    delta_t=dt
    
    
    J = fenics.derivative(F, full_trial_function)
    
    try:
    
        fenics.solve(F==0,full_trial_function,J=J)
        
    except:
        print('KABOOOM')
        break

    
    
   # solver.solve()
    
    vis_u, vis_rho, vis_rho_activity = full_trial_function.split()
      
        
    plt.close()
    
    a=fenics.plot(vis_rho)
    myosin.append(a[0].get_data()[1])
    


    plt.close()
      
    a=fenics.plot(1-c*vis_u.dx(0)) 
    actin.append(a[0].get_data()[1])
    
    
    plt.close()
      
    a=fenics.plot(vis_rho_activity) 
    rho_activity_grad.append(a[0].get_data()[1])
           
    
           
    
    iters+=1
    
    
    
    print('time is:',t)

    full_trial_function_n.assign(full_trial_function)
     
    

    
 #Break
 
plt.figure()
actin_arr=np.array(actin)  
plt.imshow(actin_arr,interpolation='nearest', aspect='auto')

plt.ylabel("Time (t)")
plt.xlabel("Distance (x/L)")   
plt.colorbar()
plt.title("Kymograph of the actin mesh density",pad=20)
#plt.xticks([0,20,40,60,80,100],[0,0.2,0.4,0.6,0.8,1])
plt.xticks([0,200,400,600,800,1000],[0,0.2,0.4,0.6,0.8,1])
plt.show()
    

    
plt.figure()
myosin_arr=np.array(myosin)  
plt.imshow(myosin_arr,interpolation='nearest', aspect='auto',vmin=0,vmax=10)

plt.ylabel("Time (t)")
plt.xlabel("Distance (x/L)")   
plt.colorbar()
plt.title("Kymograph of the myosin density profile",pad=20)
#plt.xticks([0,200,400,600,800,1000],[0,0.2,0.4,0.6,0.8,1])
#plt.xticks([0,20,40,60,80,100],[0,0.2,0.4,0.6,0.8,1])
plt.show()



plt.figure()
rho_activity_arr=np.array(rho_activity_grad)  
plt.imshow(rho_activity_arr,interpolation='nearest', aspect='auto')
plt.ylabel("Time (t)")
plt.xlabel("Distance (x/L)")   
plt.colorbar()
plt.title("Kymograph of the rho_activity mesh density",pad=20)
#plt.xticks([0,20,40,60,80,100],[0,0.2,0.4,0.6,0.8,1])
#plt.xticks([0,200,400,600,800,1000],[0,0.2,0.4,0.6,0.8,1])
plt.show()
    
 




experiment_arr=np.array(cross_section)  
plt.imshow(experiment_arr,interpolation='nearest', aspect='auto')
plt.ylabel("Time (minutes)")
plt.xlabel("Distance (x/L)")   


cbar = plt.colorbar()

cbar.set_label('Pixel intensity', rotation=270)

plt.yticks([0, 12, 24, 36, 48, 60, 72, 84],[0,1,2,3,4,5,6,7])
plt.xticks(  plt.xticks([0, 150, 300, 450, 600, 750],[0,0.2,0.4,0.6,0.8,1]))
plt.show()   





# iters=0 
# for i in range(len(actin)):
#     fig=plt.figure()
#     ax2 = plt.axes()
#     plt.ylim(0.9,2)
#     ax2.set_ylabel("Density")
#     ax2.set_xlabel("Distance (x/L)")   
#     plt.plot((actin_arr[i])/2+0.5,color="b",label="Actin")
#     plt.plot((myosin_arr[i])/100+1,color="r",label="Myosin")
#     plt.xticks([0,20,40,60,80,100],[0,0.2,0.4,0.6,0.8,1])
#     ax2.legend()
#    # plt.show()
    

#     fig.savefig('./pltmyosin/'+str(iters)+'.png')
    
#     iters+=1




plt.figure()
myosin_arr=np.array(myosin)  

p=myosin_arr[:1100].transpose()[::-1].transpose()
plt.imshow(p,interpolation='nearest', aspect='auto')

plt.ylabel("Time (dt)")
plt.xlabel("Distance (x/L)")   
plt.colorbar()

plt.xticks([0,2000,4000,6000,8000,10000],[0,0.2,0.4,0.6,0.8,1])
#plt.xticks([0,20,40,60,80,100],[0,0.2,0.4,0.6,0.8,1])
plt.show()




plt.figure()
actin_arr=np.array(actin)  

p=actin_arr[:1100].transpose()[::-1].transpose()
plt.imshow(p,interpolation='nearest', aspect='auto')

plt.ylabel("Time (dt)")
plt.xlabel("Distance (x/L)")   

plt.colorbar()



#plt.xticks([0,20,40,60,80,100],[0,0.2,0.4,0.6,0.8,1])
plt.xticks([0,2000,4000,6000,8000,10000],[0,0.2,0.4,0.6,0.8,1])
plt.show()
    








    
    
    
    
    