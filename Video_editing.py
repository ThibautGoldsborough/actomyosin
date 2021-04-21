#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 18:50:18 2021

@author: thibautgold
"""

#%matplotlib auto
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import draw


basepath ="./MB301110_i_4_rot"




photos=[]
for entry in os.listdir(basepath): #Read all photos
    if os.path.isfile(os.path.join(basepath, entry)):
        photos.append(entry)
        
        
photos.sort()

tiff_photos=[]

#Only select the brightest stacks
for tiff_index in range(len(photos)): 
    if photos[tiff_index]!='.DS_Store':
        tiff_photos.append(cv.imread(basepath+"/"+photos[tiff_index],cv.IMREAD_GRAYSCALE))
        

init=False
        
def GUI(event,x,y,flags,param):
    global start
    global tiff_photos
    global iter_photo
    global init
    global X,Y
    
    global rem

    if start==True:
        if init==False:
            X,Y=x,y
            init=True
            
        
        
        cv.line(tiff_photos[iter_photo],(x,y),(X,Y),(255),2)
        
        X,Y=x,y
        

    

iter_photo=61


#tiff_photos=saved.copy()


saved=[]

start=False
while True:
    
    rem=False    
    
    
    Image_str='Image'+str(iter_photo+1)
    
    
    cv.namedWindow(Image_str,flags=cv.WINDOW_NORMAL)
    
    
    cv.setMouseCallback(Image_str,GUI)
    
    while(1):
        

        k = cv.waitKey(1) & 0xFF
        
        if k ==ord('s'):
            start=True
        if k ==ord('e'):
            start=False
        
        
        
        if k == ord('n'):
            
            saved.append(tiff_photos[iter_photo])
    
          
            if iter_photo<len(tiff_photos)-1:
                
                
                iter_photo+=1
            else:
                iter_photo=0
    cv.imshow(Image_str,tiff_photos[iter_photo])
    
    
    
line1=tiff_photos[0][400:401,0:600]
y=[]  
x=[] 
j=0
for i in line1[0]:
    
    y.append(i)
    x.append(j)
    j+=1
    

plt.plot(x,y)
    

    
    
 
    
basepath ="./Cropped" 


tiff_photos1=[]

for i in range(1,85):
    tiff_photos1.append(cv.imread(basepath+"/img"+str(i)+".tiff",cv.IMREAD_GRAYSCALE))
    
    
#Only select the brightest stacks

iter_photo=0


#tiff_photos=saved.copy()



skel=[]

count=0
for photo in tiff_photos1:
    if count>0:
    
        ret,img = cv.threshold(photo,250,255,cv.THRESH_BINARY)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(img),\
        connectivity=4)
            
    
        blank=np.zeros((1024,1024))
        
        blank[output==0]=1
        
        overlay=prev+blank
        
        prev=blank
        
        ret,img5 = cv.threshold(overlay.astype("uint8"),0,2,cv.THRESH_BINARY)
        
        img5=img5//2
        
        img5*=255
        

        
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(img5),\
        connectivity=4)
            

        
        img=np.ones((1024,1024))

        sizes = stats[1:, -1]; nb_components = nb_components - 1   
        for i in range(0, nb_components):
            if sizes[i] >=10000 :
                img[output==i+1]=0 
  
        
        Skeletonized_Image = (skeletonize(img).astype(np.uint8))*255
        
        skel.append(Skeletonized_Image)
        
       # dist = cv.distanceTransform(img.astype('uint8'), cv.DIST_L2, 3) #Distance of a pixel to closest 0 value !
    


    else:
        ret,img = cv.threshold(photo,250,255,cv.THRESH_BINARY)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(img),\
        connectivity=4)
            
        prev=np.zeros((1024,1024))
        
        prev[output==0]=1
        
        skel.append(prev*255)
        
    
    count+=1
    

    


saved=[]

counter=0

for photo in tiff_photos:
              
    ret,img = cv.threshold(skel[counter].astype("uint8"),250,255,cv.THRESH_BINARY)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(img),\
    connectivity=4)
        
    blank=np.zeros((1024,1024))
    
    blank[output==2]=1
    
   # blank=cv.dilate(blank.astype('uint8'),None,iterations=10)
    
    img1=blank*photo
    
    img2=img1.astype('uint8')

    saved.append(img2)
    
    counter+=1
 
    




    

while True:
    
 
       
   
    
    k = cv.waitKey(1) & 0xFF

        
    if k == ord('n'):
            

          
        if iter_photo<len(tiff_photos)-1:
                
                
            iter_photo+=1
        else:
            iter_photo=0
            

    cv.imshow('Image_str',saved[iter_photo])
    
     

    
    
   
iteration=0
    
for photo in saved:
    iteration+=1
    
    photo=photo[60:900,60:600]
    cv.imwrite("./Cropped3/img"+str(iteration)+".tiff",photo)
    
    
    
    
    
    
    
basepath ="./Cropped3" 


tiff_photos2=[]


for i in range(1,85):
    tiff_photos2.append(cv.imread(basepath+"/img"+str(i)+".tiff",cv.IMREAD_GRAYSCALE))

    
iter_photo=0  





while True:
    
 
       
   
    
    k = cv.waitKey(1) & 0xFF

        
    if k == ord('n'):
            
     
        if iter_photo<len(tiff_photos2)-1:
            

                         
            iter_photo+=1
            
            
            
            
        else:
            iter_photo=0
            
    img=tiff_photos2[iter_photo]
    
    img1 = cv.medianBlur(img,9)
    
    
    ret,img = cv.threshold(img1,2,255,cv.THRESH_BINARY)
    
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(img),connectivity=4)


    sizes = stats[1:, -1]; nb_components = nb_components - 1  
    
    contours,hierarchy = cv.findContours(img.astype(np.uint8), 1,method= cv.CHAIN_APPROX_NONE)
    cnt = contours[0]
    rect = cv.minAreaRect(cnt)
    rows,cols = img.shape[:2]
    [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)


    #x0+t*vx=x
    
    
    
    #y0+t*vx=cols or 0
    
    x1=int(x+int(rows-y/vy)*vx)
    y1=rows
    
    x0=int(x+int(0-y/vy)*vx)
    y0=0
    

    cv.line(img1,(x0,y0),(x1,y1),(255),2)
    
    
    line = np.transpose(np.array(draw.line(x0,y0,x1,y1)))

    
    data = img1[line[:-1, 1], line[:-1, 0]]
    

    cv.imshow('Image_str',img1)






area=[]

actin=[]

cross_section=[]

cnts=[]



for photo in tiff_photos2:
    
    img1 = cv.medianBlur(photo,9)
    
    
    ret,img = cv.threshold(img1,2,255,cv.THRESH_BINARY)
    
    

    
    contours,hierarchy = cv.findContours(img.astype(np.uint8), 1,method= cv.CHAIN_APPROX_NONE)
    cnt = contours[0]
    rect = cv.minAreaRect(cnt)
    rows,cols = img.shape[:2]
    [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
    
    
    cnts.append(len(cnt))


    #x0+t*vx=x
 
    #y0+t*vx=cols or 0
    
    x1=int(x+int(rows-y/vy)*vx)
    y1=rows
    
    x0=int(x+int(0-y/vy)*vx)
    y0=0
    

  #  cv.line(img1,(x0,y0),(x1,y1),(255),2)
    
    
    line = np.transpose(np.array(draw.line(x0,y0,x1,y1)))

    

    all_data=[]
    
    for i in range(-1,1):

        data = img1[line[:-40, 1]+i, line[:-40, 0]]
        
        all_data.append(data)
    
        
    
    cross_section.append(np.mean(all_data, axis=0))
    

    area.append(len(np.where(img==255)[0]))
    

    sums=np.sum((img1>=60)*img1)
        
    
    actin.append(sums)



plt.figure()
plt.ylabel("Intensity")
plt.xlabel("Length (pixels)")
plt.plot(cross_section[7])


plt.show()


lengths=[]
for section in cross_section:
     indices = np.where(section==0.0)
     arr = np.delete(section, indices)
     lengths.append(len(arr))
     
     
     
    


plt.figure()

plt.plot(actin)


plt.xlabel("Frame number")

plt.ylabel("Total brightness (AU)")

plt.show()




plt.figure()

plt.plot(area)


plt.xlabel("Frame number")

plt.ylabel("Cell area (pixels)")

plt.show()



plt.figure()
x=area


y=actin
plt.scatter(x,y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.xlabel("Area (pixels)")
plt.ylabel("Actin (light intensity)")


plt.show()


#Kymograph
plt.figure()
experiment_arr=np.array(cross_section)  
plt.imshow(experiment_arr,interpolation='nearest', aspect='auto')
plt.ylabel("Time (minutes)")
plt.xlabel("Distance (x/L)")   
cbar = plt.colorbar()
cbar.set_label('Pixel intensity', rotation=270)
plt.yticks([0, 12, 24, 36, 48, 60, 72, 84],[0,1,2,3,4,5,6,7])
plt.xticks(  plt.xticks([0, 150, 300, 450, 600, 750],[0,0.2,0.4,0.6,0.8,1]))
plt.show()   





from scipy.stats import pearsonr


corr, _ = pearsonr(area,actin)


#0.38239341767805335
   
a=[]
    
for j in range(len(cross_section)):
    fig=plt.figure()
    
    plt.xlim(0,840)
    plt.ylim(0,150)
    
   # plt.plot(cross_section[j])
    
    numbers = cross_section[j]
    window_size = 20

    i = 0
    moving_averages = []
    
    while i < len(numbers) - window_size + 1:
        this_window = numbers[i : i + window_size]
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    
    a.append(moving_averages)
    
    
  #  plt.plot(moving_averages)
        
   # fig.savefig(str(j)+'.png') 
    
    


    
basepath ="./CROSS_SECTION/" 


cross=[]


for i in range(0,84):
    cross.append(cv.imread(basepath+str(i)+".png",cv.IMREAD_GRAYSCALE))

 
    
iter_photo=0

while True:
    

    k = cv.waitKey(1) & 0xFF

        
    if k == ord('n'):
            

          
        if iter_photo<len(cross)-1:
                
                
            iter_photo+=1
        else:
            iter_photo=0
            

    cv.imshow('Image_str',cross[iter_photo])
    
     

    
import os
import moviepy.video.io.ImageSequenceClip
basepath="./CROSS_SECTION/" 
fps=3



image_files=[]
for i in range(0,84):
    image_files.append(basepath+str(i)+".png")
    
    

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('profile.mp4')
    
    
    
    
        