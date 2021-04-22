#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:35:59 2021

@author: thibautgold
"""


import fenics as fenics

import numpy as np


import matplotlib.pyplot as plt


#Turn off plots:
plt.ioff()
fenics.set_log_level(40)


#Turn on plots
#plt.ion()



final2=[]
for nx in [1000,4000,5000,6000,7000,8000,9000,10000,15000,20000,25000]:
    
         #Break
     
    T = 5 # final time
    
    num_steps = 500 # number of time steps 
    
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
    
        
    periodic_boundary_condition = PeriodicBoundary()
        
    mesh = fenics.IntervalMesh(nx,0.0,1.0)
    vector_element = fenics.FiniteElement('P',fenics.interval,1)
    single_element = fenics.FiniteElement('P',fenics.interval,1)
    mixed_element = fenics.MixedElement(vector_element,single_element)
    
    V = fenics.FunctionSpace(mesh, mixed_element,constrained_domain = periodic_boundary_condition)
        
       
    k=0.5
    
    periodic_boundary_condition = PeriodicBoundary()
    
    boundary=Boundary()
    
    bc = fenics.DirichletBC(V, fenics.Constant((0,0)), boundary)
    
    bc_l = fenics.DirichletBC(V, fenics.Constant((0,0)), 'x[0] < DOLFIN_EPS')
    bc_r = fenics.DirichletBC(V, fenics.Constant((0,0)), 'x[0] > 1- DOLFIN_EPS')
    
    bcs=[bc_l,bc_r]
    
    v,r = fenics.TestFunctions(V)
    full_trial_function = fenics.Function(V)
    u, rho = fenics.split(full_trial_function)
    full_trial_function_n = fenics.Function(V)
    u_n, rho_n = fenics.split(full_trial_function_n)
    u_initial = fenics.Constant(0.0)
    rho_initial = fenics.Expression('1/k0', degree=2,k0 = k)
    u_n = fenics.interpolate(u_initial, V.sub(0).collapse())
    rho_n = fenics.interpolate(rho_initial, V.sub(1).collapse())
    np.random.seed(0)
    
    xs=np.linspace(0,501,nx+1)
    ys=[np.cos(i)/2 for i in xs]

    rho_n.vector().set_local(np.array(rho_n.vector())+ys)

   # rho_n.vector().set_local(np.array(rho_n.vector()))#+1.0*(0.5-np.random.random(rho_n.vector().size())))
    fenics.assign(full_trial_function_n, [u_n,rho_n])
    u_n, rho_n = fenics.split(full_trial_function_n)
    
    
  # plt.plot( np.array(rho_n.vector()))#+1.0*(0.5-np.random.random(rho_n.vector().size())))
    

    
    
    
    fenics.method=2
    Z1=fenics.Constant(-10.5)
    z2=fenics.Constant(0.1)
    
    b=fenics.Constant(6)
    D=fenics.Constant(0.1)
    ku0=fenics.Constant(0.5)
    alpha=fenics.Constant(1)
    kb=fenics.Constant(1)
    d = fenics.Constant(0.15)
    alpha = fenics.Constant(1.0)
    c=fenics.Constant(0.05)
    k = fenics.Constant(0.5)
    
    eta=fenics.Constant(0.7)
    
    def K(rho,rho_n):
        return (Z1*rho)/(1+z2*rho)
    
    
    def chi(n):
        value_list=[1,1,0.001,0.0001]
        return fenics.Constant(value_list[n])
    
    
    
    
    
    F = (-u_n*v*fenics.dx+ u*v*fenics.dx
         
        -dt*K(rho,rho_n)*chi(0)*v.dx(0)*fenics.dx
        +dt*(b+(K(rho,rho_n))*c*chi(1))*u.dx(0)*v.dx(0)*fenics.dx
        -dt*K(rho,rho_n)*c*c*chi(2)/2.0*u.dx(0)*u.dx(0)*v.dx(0)*fenics.dx
        +dt*K(rho,rho_n)*c*c*c*chi(3)/6.0*u.dx(0)*u.dx(0)*u.dx(0)*v.dx(0)*fenics.dx    
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
    
    
    plt.figure()
    
    problem = fenics.NonlinearVariationalProblem(F,full_trial_function,J=fenics.derivative(F,full_trial_function))
    
        
    solver = fenics.NonlinearVariationalSolver(problem)
    stype = 'newton'
    solver.parameters['nonlinear_solver']=stype
    sprms = solver.parameters[stype+'_solver']
    
    # Set maximum iterations:
    sprms['maximum_iterations'] = 100
    
    
    
    for n in range(num_steps):
      
        # Update current time
        t += dt
        
        delta_t=dt
        
        
        J = fenics.derivative(F, full_trial_function)
    
        
        #fenics.solve(F==0,full_trial_function,J=J)
        
        solver.solve()
        
        vis_u, vis_rho = full_trial_function.split()
          
            
        plt.close()
        
        a=fenics.plot(vis_rho)
        myosin.append(a[0].get_data()[1])
        
    
    
        plt.close()
          
        a=fenics.plot(1-c*vis_u.dx(0)) 
        actin.append(a[0].get_data()[1])
               
        
        iters+=1
        
        
        
        print(nx,'time is:',t)
    
        full_trial_function_n.assign(full_trial_function)
         
        
    
        

    
    
      
        
    actin_arr=np.array(actin) 
    
    final2.append(actin_arr[-1,:])
    
         
        

    





 #Break
 
plt.figure()
actin_arr=np.array(actin)  
plt.imshow(actin_arr,interpolation='nearest', aspect='auto')

plt.ylabel("Time (t)")
plt.xlabel("Distance (x/L)")   
plt.colorbar()
plt.title("Kymograph of the actin mesh density",pad=20)
plt.xticks([0,20,40,60,80,100],[0,0.2,0.4,0.6,0.8,1])
#plt.xticks([0,200,400,600,800,1000],[0,0.2,0.4,0.6,0.8,1])
plt.show()
    



exact=final2[-1]
error=[]

lenfin=len(exact)
for i in final2:
    
    
    x1=np.linspace(0,len(i)-1,len(i))*lenfin/(len(i)-1)
    
    x2=np.linspace(0,lenfin-1,lenfin)
    


    y2=np.interp(x2,x1,i)
    
    plt.plot(y2,label=str(len(i)))
    plt.legend()


    
    error.append(sum(abs(exact-y2)))
    

plt.figure()

x=[1000,4000,5000,6000,7000,8000,9000,10000,15000]


plt.xlabel("Number of gridpoints (log scale)")
plt.ylabel("Error (log scale)")
plt.plot(np.log(x),np.log(error[:-1]),label='error')
plt.plot(np.log(x),np.log([1/i for i in x])+12,label='1/x')
plt.plot(np.log(x),np.log([1/(i**2) for i in x])+18.89,label='1/x**2')
plt.legend()







    
 
    