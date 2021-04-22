#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:34:51 2021

@author: thibautgold
"""



import fenics as fenics

import numpy as np

import mshr as mshr


import os

import matplotlib.pyplot as plt

import time as time


#Turn off plots:
#plt.ioff()
fenics.set_log_level(40)


#Turn on plots
#plt.ion()

T = 8# final time

num_steps = 2000# number of time steps 

dt = T / num_steps


 



class PeriodicBoundary(fenics.SubDomain):
         # Left boundary is "target domain" G
         def inside(self, x, on_boundary):
             return bool(- fenics.DOLFIN_EPS < x[0] < fenics.DOLFIN_EPS and on_boundary)

         def map(self, x, y):
             y[0] = x[0] - 1
             y[1] = x[1]

class PeriodicBoundary(fenics.SubDomain):
            # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
            return bool((fenics.near(x[0], 0) or fenics.near(x[1], 0)) and 
                    (not ((fenics.near(x[0], 0) and fenics.near(x[1], 1)) or 
                            (fenics.near(x[0], 1) and fenics.near(x[1], 0)))) and on_boundary)
    
        def map(self, x, y):
            if fenics.near(x[0], 1) and fenics.near(x[1], 1):
                y[0] = x[0] - 1.
                y[1] = x[1] - 1.
            elif fenics.near(x[0], 1):
                y[0] = x[0] - 1.
                y[1] = x[1]
            else:   # near(x[1], 1)
                y[0] = x[0]
                y[1] = x[1] - 1.
   

mesh = fenics.UnitSquareMesh(20,20)

ny = 20 
nx=25


domain_size = 1.0
mesh = fenics.RectangleMesh(fenics.Point(0, 0), fenics.Point(domain_size, domain_size), nx, ny) 






domain = mshr.Circle(fenics.Point(0, 0), 1)


domain=mshr.Ellipse(fenics.Point(0,0),0.5,1)

mesh = mshr.generate_mesh(domain, 25)



periodic_boundary_condition = PeriodicBoundary()
    

 

vector_element = fenics.VectorElement('P',fenics.triangle,2,dim = 2)
single_element = fenics.FiniteElement('P',fenics.triangle,2)
mixed_element = fenics.MixedElement(vector_element,single_element)
V = fenics.FunctionSpace(mesh, mixed_element)#,constrained_domain=periodic_boundary_condition)
   





k=0.5
v,r = fenics.TestFunctions(V)

full_trial_function = fenics.Function(V)

u, rho = fenics.split(full_trial_function)

full_trial_function_n = fenics.Function(V)

u_n, rho_n = fenics.split(full_trial_function_n)

u_initial = fenics.Constant((0.0,0.0))



rho_initial = fenics.Expression('1/k0', degree=2,k0 = k)

rho_initial = fenics.Expression('1-0.1*x[1]', degree=2)

u_n = fenics.interpolate(u_initial, V.sub(0).collapse())



rho_n = fenics.interpolate(rho_initial, V.sub(1).collapse())

#np.random.seed(0)

rho_n.vector().set_local(np.array(rho_n.vector())+1.0*(0.5-np.random.random(rho_n.vector().size())))

fenics.assign(full_trial_function_n, [u_n,rho_n])

u_n, rho_n = fenics.split(full_trial_function_n)




def epsilon(u): 
       return 0.5*(fenics.grad (u) + fenics.grad (u).T )



fenics.method=2
Z1=fenics.Constant(-7)
z2=fenics.Constant(0.1)

b=fenics.Constant(6)

ku0=fenics.Constant(0.5)
alpha=fenics.Constant(1)

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


     

import ufl

n = fenics.FacetNormal(mesh)


F = (-fenics.inner(u_n,v)*fenics.dx + fenics.inner(u,v)*fenics.dx
     
    -dt*K(rho,rho_n)*chi(0)*ufl.nabla_div(v)*fenics.dx
 
    +dt*(b+(K(rho,rho_n))*c*chi(1))*fenics.inner(epsilon(u),fenics.grad(v))*fenics.dx
 
    +eta*fenics.inner(epsilon(u),fenics.grad(v))*fenics.dx-eta*fenics.inner(epsilon(u_n),fenics.grad(v))*fenics.dx
      
    -rho_n*r*fenics.dx+rho*r*fenics.dx
 
    +fenics.inner(rho*u_n,fenics.grad(r))*fenics.dx-fenics.inner(rho*u,fenics.grad(r))*fenics.dx 
  
    +fenics.inner(dt*d*fenics.grad(rho),fenics.grad(r))*fenics.dx
    
    

    
    +dt*k*fenics.exp(alpha*ufl.nabla_div(u))*rho*r*fenics.dx
    
    
    -dt*r*fenics.dx)
    
  #  +dt*c*epsilon(u)*r*fenics.dx)
  
 #   +dt*c*ufl.nabla_div(v)*r*fenics.dx)
    


iters=0
t = 0






actin,myosin=[],[]


plt.figure()

# problem = fenics.NonlinearVariationalProblem(F,full_trial_function,J=fenics.derivative(F,full_trial_function))

    
# solver = fenics.NonlinearVariationalSolver(problem)
# stype = 'newton'
# solver.parameters['nonlinear_solver']=stype
# sprms = solver.parameters[stype+'_solver']

# # Set maximum iterations:
# sprms['maximum_iterations'] = 100




base="./pltactin/"


for entry in os.listdir(base):
    os.remove(base+entry)#Read all photos



base="./pltmyosin/"

for entry in os.listdir(base):
    os.remove(base+entry)#Read all photos






times=[]

increment=0

for n in range(num_steps):
    increment+=1
    start_time=time.time()
  
    # Update current time
    t += dt
    

    
    
    J = fenics.derivative(F, full_trial_function)

    
    fenics.solve(F==0,full_trial_function,J=J)
    
   # solver.solve()
    
    vis_u, vis_rho = full_trial_function.split()
    
    if increment//1==increment/1:
          
            
        plt.close()
        
        fig=plt.figure()
        ax=plt.axes()
         
        ax.set_xlim([-1,1])
        
        plt.ylabel("y")
        plt.xlabel("x")   
        
    
        
        c = fenics.plot(fenics.interpolate(vis_rho, V.sub(1).collapse()), mode='color',vmin=0,vmax=25)
        
        plt.colorbar(c)
        
        # myosin.append(c.get_data()[1])
      
        fig.savefig('./pltmyosin/'+str(iters)+'.png')
       #  
     #   plt.show()
        # 
        plt.close()
        
        
        
        
        fig=plt.figure()
        
        ax=plt.axes()
        
      #  ax.set_aspect('equal')
    
        ax.set_xlim([-1,1])
        
        plt.ylabel("y")
        plt.xlabel("x")   
        
        c=fenics.plot(1-ufl.nabla_div(vis_u), mode='color',vmin=0,vmax=20)
        
        plt.colorbar(c)
        
       # myosin.append(c.get_data()[1])
       
      #  plt.show()
        
        plt.close()
       
        fig.savefig('./pltactin/'+str(iters)+'.png')
    
        
    
    
  
    
    
    
  #  fenics.plot(1-ufl.nabla_div(vis_u))
  
   # plt.show()
    


#    plt.close()
      
   # a=fenics.plot(1-c*vis_u.dx(0)) 
   # actin.append(a[0].get_data()[1])
  
  
 #   plt.show()
    
    iters+=1
    
    
    end_time=time.time()
    
    dtime=-start_time+end_time
    
    time_left=(T-t)/dt*dtime
    
    times.append(time_left)
    
    if len(times)>=10:
        time_left=np.mean(times)
        print(",Estimated time remaining:", round(time_left/60,3) ,"min")
        times=[]
    
    
    
    print('time is:',t)
    
    

    full_trial_function_n.assign(full_trial_function)
     
    
    
a=1/0 #Break





def movie():
    global num_steps
    
 #   basepaths=["./pltdu/","./pltu/"]#,"./pltmyosin/","./pltactin/"]
 #   movie_names=["du.mp4","u.mp4",]#"myosin.mp4","actin.mp4"]
    
    basepaths=["./pltactin/"]
    movie_names=["actin2.mp4"]
   
    

    import moviepy.video.io.ImageSequenceClip
    
    
    for movie in movie_names:
        try:
            os.remove(movie)
            
        except:
            pass

    j=0
    
    for basepath in basepaths:
    

         
        photos=[]
        for entry in os.listdir(basepath): #Read all photos
            if os.path.isfile(os.path.join(basepath, entry)):
                photos.append(entry)
    
        
        
        
        
    
        fps=int(len(photos)/13)
        
        image_files=[]
        for i in range(1,len(photos)):
            image_files.append(basepath+str(i)+".png")
                 
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(movie_names[j])
        
        j+=1
            
          





    

vertex_values_u = vis_rho.compute_vertex_values() 
    
   
    
   
mesh1 = fenics.UnitSquareMesh (10,10)
V = fenics.VectorFunctionSpace(mesh1, "Lagrange", 1)      
n = V.dim()                                #  is 11*11*2 = 242                                                           
d = mesh1.geometry().dim()            #  is  2                                             
dof_coordinates = V.tabulate_dof_coordinates()     
   
    
a=np.zeros((50,50))
    
for n,i in enumerate(dof_coordinates):
    
    b=i*50
    
    a[int(b[0]),int(b[1])]=1
    
plt.imshow(a)
   

n = V.sub(1).dim()                                                                      
d = mesh.geometry().dim()                                                        

dof_coordinates = V.sub(0).collapse().tabulate_dof_coordinates()  

                    
dof_coordinates.resize((n, d))                                                   
dof_x = dof_coordinates[:, 0]                                                    
dof_y = dof_coordinates[:, 1]                                                    

   
fig = plt.figure()                                                               
ax = fig.add_subplot()                                       
b=ax.scatter(dof_x, dof_y)# vis_rho.vector().get_local(), c='b', marker='.')                  
plt.show()      
    

    
   
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
xi = yi = np.arange(-1,1,0.02)

xi,yi = np.meshgrid(xi,yi)



z = vis_rho.vector().get_local()
# interpolate



zi = griddata((dof_x,dof_y),z[2::3],(xi,yi),method='cubic')


plt.contourf(zi)
   
    
   
    
   
xi,yi = np.meshgrid(dof_x,dof_y)
   
    

    
plt.imshow(xi)
    
    
    
    
    
    
    
    
    
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

x = np.linspace(-1,1,100)
y =  np.linspace(-1,1,100)
X, Y = np.meshgrid(x,y)



T = z
# Choose npts random point from the discrete domain of our model function
npts = 400
px, py = np.random.choice(x, npts), np.random.choice(y, npts)

fig, ax = plt.subplots(nrows=2, ncols=2)
# Plot the model function and the randomly selected sample points
ax[0,0].contourf(X, Y, T)
ax[0,0].scatter(px, py, c='k', alpha=0.2, marker='.')
ax[0,0].set_title('Sample points on f(X,Y)')

# Interpolate using three different methods and plot
for i, method in enumerate(('nearest', 'linear', 'cubic')):
    Ti = griddata((px, py), z, (X, Y), method=method)
    r, c = (i+1) // 2, (i+1) % 2
    ax[r,c].contourf(X, Y, Ti)
    ax[r,c].set_title("method = '{}'".format(method))

plt.tight_layout()
plt.show()
    
    
    
    
    
    
vertex_values = np.zeros(mesh.num_vertices())
for vertex in fenics.vertices(mesh):
  x = vertex.x(0)
  y = vertex.x(1)
  vertex_values[vertex.index()] = abs(x + y)

full_trial_function.vector()[:] = vertex_values[fenics.dof_to_vertex_map(full_trial_function)]
    
    
    
    
u_v=full_trial_function.vector()
x_dofs = V.sub(0).dofmap().dofs()
y_dofs = V.sub(1).dofmap().dofs()


for x_dof, y_dof in zip(x_dofs, y_dofs):
  print(dof_coordinates[x_dof], dof_coordinates[y_dof], u_v[x_dof], u_v[y_dof] )
   
    
   



import moviepy.video.io.ImageSequenceClip


import cv2 as cv
   
def movie1():
    global num_steps
    
 #   basepaths=["./pltdu/","./pltu/"]#,"./pltmyosin/","./pltactin/"]
 #   movie_names=["du.mp4","u.mp4",]#"myosin.mp4","actin.mp4"]
    
    basepaths=["./pltactin/","./pltmyosin/"]
    movie_names=["actin22.mp4","./myosin22/"]
   
    png_photos1=[]
    for i in range(0,2000):
        png_photos1.append(cv.imread(basepaths[0]+str(i)+".png"))
        
           
    png_photos2=[]
    for i in range(0,2000):
        png_photos2.append(cv.imread(basepaths[1]+str(i)+".png"))
        
        
    time=0
    
    stack1=[]
    for png in range(len(png_photos1)):
        
      a=cv.putText(png_photos1[png],'Time: '+str(round(time,4)),(10,20), cv.FONT_HERSHEY_SIMPLEX, .3,(0,0,0),1,cv.LINE_AA)
        
      actin=cv.putText(a,'Actin density',(160,20), cv.FONT_HERSHEY_SIMPLEX, .3,(0,0,0),1,cv.LINE_AA)
      
      time+=0.004
        

      myosin=cv.putText(png_photos2[png],'Myosin density',(160,20), cv.FONT_HERSHEY_SIMPLEX, .3,(0,0,0),1,cv.LINE_AA)
      
     
        
     
      stack=np.hstack((actin,myosin))
      
      
      stack1.append(stack)
      
      
    for iters,stack in enumerate(stack1):
          
        cv.imwrite("./stacks/"+str(iters)+'.png',stack)
        





    basepaths=["./stacks/"]
    

    j=0
    
    for basepath in basepaths:
    

         
        photos=[]
        for entry in os.listdir(basepath): #Read all photos
            if os.path.isfile(os.path.join(basepath, entry)):
                photos.append(entry)
    
        
        
        
        
    
        fps=int(len(photos)/13)
        
        image_files=[]
        for i in range(0,len(photos)-1):
            image_files.append(basepath+str(i)+".png")
                 
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile('myosin_actin_stack.mp4')
        
        j+=1
            
          



    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    