import fenics as fenics
import numpy as np
import matplotlib.pyplot as plt

fenics.set_log_level(40)


boundary_type='Neumann'
#boundary_type='Dirichlet'
#boundary_type='Periodic'



T = 5 # final time
num_steps = T*1000 # number of time steps 
dt = T / num_steps

mesh = fenics.IntervalMesh(1000,0.0,1.0)
vector_element = fenics.FiniteElement('P',fenics.interval,1)
single_element = fenics.FiniteElement('P',fenics.interval,1)
mixed_element = fenics.MixedElement(vector_element,single_element)




#BOUNDARIES

if boundary_type=='Periodic':
    class PeriodicBoundary(fenics.SubDomain):
             # Left boundary is "target domain" G
             def inside(self, x, on_boundary):
                 return bool(- fenics.DOLFIN_EPS < x[0] < fenics.DOLFIN_EPS and on_boundary)
    
             def map(self, x, y):
                 y[0] = x[0] - 1
    periodic_boundary_condition = PeriodicBoundary()
    
    V = fenics.FunctionSpace(mesh, mixed_element,constrained_domain = periodic_boundary_condition)

    Z1=fenics.Constant(-10)
    z2=fenics.Constant(0.1)
    b=fenics.Constant(6)
    alpha=fenics.Constant(1.0)
    d = fenics.Constant(0.15)
    c=fenics.Constant(0.05)
    k = fenics.Constant(0.5)
    eta=fenics.Constant(0.7)


if boundary_type=='Neumann':
    V = fenics.FunctionSpace(mesh, mixed_element)
    
    Z1=fenics.Constant(-10)
    z2=fenics.Constant(0.1)
    b=fenics.Constant(6)
    alpha=fenics.Constant(1.0)
    d = fenics.Constant(0.15)
    c=fenics.Constant(0.05)
    k = fenics.Constant(0.5)
    eta=fenics.Constant(0.7)
    
    

if boundary_type=='Dirichlet':
    
    Z1=fenics.Constant(-12)
    z2=fenics.Constant(0.2)
    b=fenics.Constant(6)
    alpha=fenics.Constant(1.0)
    d = fenics.Constant(0.01)
    c=fenics.Constant(0.05)
    k = fenics.Constant(0.5)
    eta=fenics.Constant(0.7)
    
    class Boundary(fenics.SubDomain):

        def inside(self, x, on_boundary):
            return bool(- fenics.DOLFIN_EPS < x[0] < fenics.DOLFIN_EPS and on_boundary)

    boundary=Boundary()
    V = fenics.FunctionSpace(mesh, mixed_element)
    bc = fenics.DirichletBC(V, fenics.Constant((0,0)), boundary)
    bc_l = fenics.DirichletBC(V, fenics.Constant((0,0)), 'x[0] < DOLFIN_EPS')
    bc_r = fenics.DirichletBC(V, fenics.Constant((0,0)), 'x[0] > 2- DOLFIN_EPS')
    bcs=[bc_l,bc_r]
    
    
    
    
v,r = fenics.TestFunctions(V)
full_trial_function = fenics.Function(V)
u, rho = fenics.split(full_trial_function)
full_trial_function_n = fenics.Function(V)
u_n, rho_n = fenics.split(full_trial_function_n)
u_initial = fenics.Constant(0.0)
rho_initial = fenics.Constant(2.0)
u_n = fenics.interpolate(u_initial, V.sub(0).collapse())
rho_n = fenics.interpolate(rho_initial, V.sub(1).collapse())

np.random.seed(0)
rho_n.vector().set_local(np.array(rho_n.vector())+1.0*(0.5-np.random.random(rho_n.vector().size())))
fenics.assign(full_trial_function_n, [u_n,rho_n])
u_n, rho_n = fenics.split(full_trial_function_n)





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

 


if boundary_type=='Dirichlet':
    problem = fenics.NonlinearVariationalProblem(F,full_trial_function,J=fenics.derivative(F,full_trial_function),bcs=bcs)

else:
    problem = fenics.NonlinearVariationalProblem(F,full_trial_function,J=fenics.derivative(F,full_trial_function))

solver = fenics.NonlinearVariationalSolver(problem)



t = 0
actin,myosin=[],[]

for n in range(num_steps):
  
    # Update current time
    t += dt
      
    J = fenics.derivative(F, full_trial_function)
    
    try:
    
        solver.solve()
        
    except:
        print('Error:  Unable to solve nonlinear system with NewtonSolver.')
        break

   
    
    
    vis_u, vis_rho = full_trial_function.split()
            
    a=fenics.plot(vis_rho)
    myosin.append(a[0].get_data()[1])

    plt.close()     
    a=fenics.plot(1-c*vis_u.dx(0)) 
    actin.append(a[0].get_data()[1])
    plt.close()
    
    print('time is:',t)
    
    full_trial_function_n.assign(full_trial_function)
     
 
plt.figure()
actin_arr=np.array(actin)  
plt.imshow(actin_arr,interpolation='nearest', aspect='auto')
plt.ylabel("Time (t)")
plt.xlabel("Distance (x/L)")   
plt.colorbar()
plt.title("Actin mesh density",pad=20)
plt.yticks([0,1000,2000,3000,4000,5000],[0,1,2,3,4,5])
#plt.xticks([0,200,400,600,800,1000],[0,0.2,0.4,0.6,0.8,1])
plt.show()
    
   
plt.figure()
myosin_arr=np.array(myosin)  
plt.imshow(myosin_arr,interpolation='nearest', aspect='auto')
plt.ylabel("Time (t)")
plt.xlabel("Distance (x/L)")   
plt.colorbar()
plt.title("Bound myosin density",pad=20)
plt.yticks([0,1000,2000,3000,4000,5000],[0,1,2,3,4,5])
#plt.xticks([0,200,400,600,800,1000],[0,0.2,0.4,0.6,0.8,1])
plt.show()
 


    