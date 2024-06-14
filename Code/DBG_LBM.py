#!/usr/bin/env python
# coding: utf-8

#Lattice Boltzmann Method
#Disease Biophysics Group
#Written by John F. Zimmerman
#Updated 11/08/2021

import tensorflow as tf
import os
import numpy as np
import math 
import cv2
from tqdm import *

from tensorflow.python.framework.ops import disable_eager_execution

from matplotlib.pyplot import *
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# In[16]:


class Grid():
    def __init__(self, vis, shape,numsteps, Fvecs=15,dx=1.0, dt=1.0,cycletime=30,bkg=0.0001):
        self.shape  = shape
        self.Nneigh = 15
        self.Dim    = 3
        self.tau    = tf.constant(0.5 +3*dt*vis/(dx**2),dtype=tf.float16)
        
        self.FrameVelx = tf.Variable(np.zeros([1]),dtype=tf.float16)
        self.Vel    = tf.Variable(np.zeros([1,shape[0],shape[1],shape[2],3], dtype=np.float16)) #Store Velocities
        self.Rho    = tf.Variable(np.zeros([1,shape[0],shape[1],shape[2],1], dtype=np.float16)) #Store densities
        self.MIndex = tf.Variable(np.zeros([1,shape[0],shape[1],shape[2],1], dtype=np.float16)) #Store materials index (walls)
        self.ObjVels = tf.Variable(np.zeros([cycletime,1,shape[0],shape[1],shape[2],4], dtype=np.float16)) #A split of all input velocities, plus an object mask

        self.F      = tf.Variable(np.zeros([1,shape[0],shape[1],shape[2],self.Nneigh], dtype=np.float16)) #Density of state info
        self.OutFlux= tf.Variable(np.zeros([numsteps], dtype=np.float16))
        
        self.W      = tf.reshape(Grid.weights(), (self.Dim + 1)*[1] + [self.Nneigh])
        self.C      = tf.reshape(Grid.vecConstants(Fvecs), self.Dim*[1] + [self.Nneigh,3]) 
        self.Op     = tf.reshape(Grid.Bounce_Mat(), self.Dim*[1] + [self.Nneigh,self.Nneigh])
        self.St     = Grid.Stream_Mat()
        self.Cs     = dx/dt
        self.dx     = dx
        self.dt     = dt
        self.Inlet  = Grid.InletMask(shape) #Used in discreet flow problems
        self.Outlet = Grid.OutletMask(shape) #Used in discreet flow problems
        self.Continue = tf.Variable(True)
        self.swimmass = 1.0 #Density of swimmer relative to fluid

    def weights():
        #D3Q15 - vector weights normalized by distance
        nom1 = 9.0
        nom2 = 72.0

        return tf.constant([2.0/nom1, 1.0/nom1, 1.0/nom1, 1.0/nom1, 1.0/nom1,  1./nom1,  1.0/nom1, 1.0/nom2, 1.0/nom2 , 1.0/nom2, 1.0/nom2, 1.0/nom2, 1.0/nom2, 1.0/nom2, 1.0/nom2],dtype=tf.float16)
        
    
    def vecConstants(Fvecs):
        # Lattice weights, how info is transfered
        C = tf.constant(
        [ [ 0, 0, 0], [ 1, 0, 0], [-1, 0, 0], [ 0, 1, 0], [ 0,-1, 0], 
          [ 0, 0, 1], [ 0, 0,-1], [ 1, 1, 1], [-1,-1,-1], [ 1, 1,-1], 
          [-1,-1, 1], [ 1,-1, 1], [-1, 1,-1], [ 1,-1,-1], [-1, 1, 1] ], dtype=tf.float16)
        C = tf.reshape(C,[1,1,1,Fvecs,3])
        return C
    
    def Stream_Mat():
        # Stream Matrix - How Data is streamed to neighboring areas
        Stream_Mat = np.zeros((3,3,3,15,15))
        #identity
        Stream_Mat[1,1,1, 0, 0] = 1.0
        #1-6
        Stream_Mat[2,1,1, 1, 1] = 1.0
        Stream_Mat[0,1,1, 2, 2] = 1.0
        Stream_Mat[1,2,1, 3, 3] = 1.0
        Stream_Mat[1,0,1, 4, 4] = 1.0
        Stream_Mat[1,1,2, 5, 5] = 1.0
        Stream_Mat[1,1,0, 6, 6] = 1.0
        #7-14
        Stream_Mat[2,2,2, 7, 7] = 1.0
        Stream_Mat[0,0,0, 8, 8] = 1.0
        Stream_Mat[2,2,0, 9, 9] = 1.0
        Stream_Mat[0,0,2,10,10] = 1.0
        Stream_Mat[2,0,2,11,11] = 1.0
        Stream_Mat[0,2,0,12,12] = 1.0
        Stream_Mat[2,0,0,13,13] = 1.0
        Stream_Mat[0,2,2,14,14] = 1.0
        Stream_Mat = tf.constant(Stream_Mat, dtype=tf.float16)
        return Stream_Mat
    
    def Bounce_Mat():
        # Bounce Matrix - How information bounces off of the walls
        Bounce_Mat = np.zeros([15,15])
        Bounce_Mat[0,0]=1
        for i in range(0,7):
                Bounce_Mat[i*2+1,i*2+2]=1.0
                Bounce_Mat[i*2+2,i*2+1]=1.0
        Bounce_Mat = tf.constant(tf.convert_to_tensor(Bounce_Mat.astype(np.float16)), dtype=tf.float16)
        return Bounce_Mat
    
    def FlowThrough(self):
        #Setup of wall conditions for a flow through box
        shape = self.shape
        matgrid = np.zeros([1,shape[0],shape[1],shape[2],1])
        
        #Default Box Shape Around Grid
        matgrid[:,:,:,0:2,:]=1
        matgrid[:,:,:,-2:,:]=1
        matgrid[:,:,0:2,:,:]=1
        matgrid[:,:,-2:,:,:]=1
        
        return tf.convert_to_tensor(matgrid.astype(np.float16))
    

    def ClosedBox(self):
        #Setup of wall conditions for a closed container
        shape = self.shape
        matgrid = np.zeros([1,shape[0],shape[1],shape[2],1])
        
        #Default Box Shape Around Grid
        matgrid[:,:,:,0:2,:]=1
        matgrid[:,:,:,-2:,:]=1
        matgrid[:,:,0:2,:,:]=1
        matgrid[:,:,-2:,:,:]=1
        matgrid[:,0:2,:,:,:]=1
        matgrid[:,-2:,:,:,:]=1
        

        return tf.convert_to_tensor(matgrid.astype(np.float16))


    def CavityBox(self):
        shape = self.shape
        matgrid = np.zeros([1,shape[0],shape[1],shape[2],1])
        
        #Default Box Shape Around Grid
        matgrid[:,:,:,-1:,:]=1
        matgrid[:,:,0:1,:,:]=1
        matgrid[:,:,-1:,:,:]=1
        matgrid[:,0:1,:,:,:]=1
        matgrid[:,-1:,:,:,:]=1

        return tf.convert_to_tensor(matgrid.astype(np.float16))


    def InitializeGrid(self):
        #Initialize grid at the start of the simulation, setting density of states to equilibrium conditions
        startRho = 1
        BounceMask = tf.cast(tf.math.equal(self.MIndex,1),tf.float16)
        vel = self.Vel
        vel_dot_vel = tf.expand_dims(tf.reduce_sum(vel * vel, axis=4), axis=4)
        vel_dot_c = tf.reduce_sum(tf.expand_dims(vel, axis=4) * self.C, axis=5)
        feq = self.W * startRho*(1.0 + (3.0/self.Cs)*vel_dot_c + (4.5/(self.Cs*self.Cs))*(vel_dot_c*vel_dot_c) - (1.5/(self.Cs*self.Cs))*vel_dot_vel)

        vel = vel * (1.0 - BounceMask)
        rho = startRho*(1.0 - BounceMask) + startRho*(BounceMask)
        
        self.F.assign(feq)
        self.Rho.assign(rho)
        self.Vel.assign(vel)
        return
    
    @tf.function
    def DefaultBound(self,Matnum):
        Fis = self.Fi
        
        #Boundary Conditions - Where The Walls are located return as a false-true grid
        BounceMask = tf.cast(tf.math.equal(grid.MIndex,Matnum),tf.float16)
        
        #Flipping Fi values at the wall interface - No slip boundary condition
        f = tf.multiply(Fis, tf.expand_dims(BounceMask,axis=-1))
        f = tf.nn.conv3d( tf.expand_dims(f,axis=0), self.Rebound,[1, 1, 1, 1, 1], padding='VALID')
        f = tf.squeeze(f)
        return f, BounceMask

    def InletMask(shape):
        #Make a one hot encoded mask for source of Velocities
        ExtraVel = np.zeros([1,shape[0],shape[1],shape[2],3])
        #ExtraVel[:25,75:78,:,1] = 1.0
        ExtraVel[:,:1,:,:,0] = 1.0
        

        return tf.Variable(tf.convert_to_tensor(ExtraVel.astype(np.float16)))
    
    
    def OutletMask(shape):
        #Make a one hot encoded mask for the outlet
        Outvel = np.zeros([1,shape[0],shape[1],shape[2],3])
        Outvel[:,-5:-4,2:-2,2:-2,0] = 1.0

        return tf.constant(tf.convert_to_tensor(Outvel.astype(np.float16)))
        
    @tf.function
    def Stream(self,ObjVel):
        # stream f
        f_pad = pad_mobius(self.F) #make data wrap around the edges of the box
        f_pad = simple_conv(f_pad, self.St) #define how data will be convolved
        
        # calc new velocity and density
        Rho = tf.expand_dims(tf.reduce_sum(f_pad, self.Dim+1), self.Dim+1)
        
        
        #Simulation Stability Criterion
        Rho = tf.where(Rho<np.float16(0.15), np.float16(0.15), Rho) #Set minimum density of states as backup     
        Rho = tf.where(tf.math.is_inf(Rho), self.Rho, Rho) #Check for infinite and none existant values
        Rho = tf.where(tf.math.is_nan(Rho), self.Rho, Rho)
        
        Vel = simple_conv(f_pad, self.C)
        Vel = Vel*self.Cs/Rho

        #Introducing Fin Velocity
        Vel = tf.where(ObjVel != 0, ObjVel, Vel)
        
        #Filter for stability
        Vel = self.filtVels(Vel)
        Vel = tf.where(tf.math.is_inf(Vel), self.Vel, Vel)
        Vel = tf.where(tf.math.is_nan(Vel), self.Vel, Vel)
        
        tf.debugging.check_numerics(f_pad, message='Error-Fi-Stream Step')
        tf.debugging.check_numerics(Rho, message='Error-Rho-Stream Step')
        tf.debugging.check_numerics(Vel, message='Error-Vel-Stream Step')

        #Assigning local values to global values
        self.F.assign(f_pad)
        self.Rho.assign(Rho)
        self.Vel.assign(Vel)
        
        return
    
    @tf.function
    def ObjectStream(self,ObjVels,ObjMask):
        Vels = self.Vel
        OutletMask = self.Outlet
        Vels = Vels*(1.0-OutletMask) #introduce velocities from outlet

        self.Vel.assign(Vels)
        return 
    
    @tf.function
    def Ehrenfests(self,f,W,Feq):
        Sfeq = self.calcEntropy(Feq,W)
        Sf = self.calcEntropy(f,W)
        return Sfeq-Sf
    
    
    def calcEntropy(self,f,W):
        return -1*tf.expand_dims(tf.reduce_sum(f*tf.math.log(f/W), self.Dim+1), self.Dim+1)
    
    def filtVels(self,Vel,critical=5):
        std = tf.math.reduce_std(Vel)
        mean = tf.math.reduce_mean(Vel)
        Vel = tf.where(Vel> mean+critical*std, self.Vel, Vel)
        Vel = tf.where(Vel< mean-critical*std, self.Vel, Vel)
        
        return Vel
    
    def filtF(self,f,Feq,critical=5):
        std = tf.math.reduce_std(f)
        mean = tf.math.reduce_mean(f)
        f = tf.where(f> mean+critical*std, Feq, f)
        f = tf.where(f< mean-critical*std, Feq, f)
        
        return f
    
    @tf.function
    def Collide(self,ObjMask):
        # boundary bounce piece
        #BounceMask = tf.cast(tf.math.equal(self.MIndex,1),tf.float16)
        BounceMask = tf.cast(tf.math.equal(self.MIndex,1),tf.float16)
        BounceMask = BounceMask + (1-ObjMask)
        f_boundary = tf.multiply(self.F[0], BounceMask)
        f_boundary = simple_conv(f_boundary, self.Op)

        # make vel bforce and rho
        f   = self.F
        vel = self.Vel
        rho = self.Rho # to stop dividing by zero
        tau = self.tau
        W = self.W

        # calc v dots
        vel_dot_vel = tf.expand_dims(tf.reduce_sum(vel * vel, axis=self.Dim+1), axis=self.Dim+1)
        vel_dot_c = simple_conv(vel, tf.transpose(self.C, [0,1,2,4,3]))
        
        #print('veldotc shape: ' +str(vel_dot_c.shape))
        
        # calc Feq
        Feq = self.W * rho * (1.0 + 3.0*vel_dot_c/self.Cs + 4.5*vel_dot_c*vel_dot_c/(self.Cs*self.Cs) - 1.5*vel_dot_vel/(self.Cs*self.Cs))
        #Feq = tf.where(Feq<0.001, 0.001, Feq)
        
        
        # collision calc
        NonEq = f - Feq
        f = f - NonEq/tau
        
        #Stability
        S = self.Ehrenfests(f,W,Feq)
        Ehrenfestcrit = np.float16(0.001)
        f = tf.where(S> Ehrenfestcrit, Feq, f)
        
        #f = tf.where(f<np.float16(0.001), np.float16(0.001), f)
        f = self.filtF(f,Feq)
        f = tf.where(tf.math.is_nan(f), self.F, f)
        f = tf.where(tf.math.is_inf(f), self.F, f)
        
        
        # combine boundary and no boundary values
        f_no_boundary = tf.multiply(f, (1.0-BounceMask))
        f = f_no_boundary + f_boundary

        tf.debugging.check_numerics(f, message='Error-f-Collide Step')
        self.F.assign(f)
                   
        
        return
    
    @tf.function
    def ObjCollide(self,ObjVels):
        #Introducing momentum from our fin into the LBM simulation
        f   = self.F
        tau = self.tau
        rho = self.Rho
        #ObjVels = tf.expand_dims(self.ObjVels[cycle,:,:,:,:],axis=0)
        #ObjVels = self.ObjVels[cycle]
        
        #Interaction with Moving Objects
        #Assume Equilibrium For Moving Surface
        Obj_vel_dot_vel = tf.expand_dims(tf.reduce_sum(ObjVels * ObjVels, axis=self.Dim+1), axis=self.Dim+1)
        Objvel_dot_c = simple_conv(ObjVels, tf.transpose(self.C, [0,1,2,4,3]))
        ObjEq = self.W * self.Rho * (1.0 + (3.0/self.Cs)*Objvel_dot_c + (4.5/(self.Cs*self.Cs))*(Objvel_dot_c*Objvel_dot_c)- (1.5/(self.Cs*self.Cs))*Obj_vel_dot_vel)
        #ObjEq = self.W  * (1.0 + (3.0/self.Cs)*Objvel_dot_c + (4.5/(self.Cs*self.Cs))*(Objvel_dot_c*Objvel_dot_c)- (1.5/(self.Cs*self.Cs))*Obj_vel_dot_vel)
        
        #f = tf.where(Obj_vel_dot_vel != 0, f-(f-ObjEq)/tau, f)
        f = tf.where(Obj_vel_dot_vel != 0, f+self.W*6*(Objvel_dot_c/self.Cs)*rho, f)
        #f = tf.where(Obj_vel_dot_vel != 0, ObjEq, f)
        
        f = self.filtF(f,ObjEq)
        f = tf.where(tf.math.is_nan(f), self.F, f)
        f = tf.where(tf.math.is_inf(f), self.F, f)
        
        self.F.assign(f)
        return
    
    def evalOutFlux(self,frame):
        dx = self.dx
        rho = self.Rho 
        vels = self.Vel
        outletMaskPlane = self.Outlet
        
        massflux =tf.reduce_sum(rho*vels*outletMaskPlane*dx*dx) #Assumes uniform size boxes such that dx=dy=dz
        #self.OutFlux[frame].assign(massflux) #Units should be g/mm^2
        self.OutFlux[frame].assign(tf.reduce_sum(self.FrameVelx))
        return
        
    def expand_pad(self):
        f_mobius = self.F
        rho = self.Rho
        #mRho = tf.reduce_mean(rho)
        #mRho = tf.math.reduce_min(rho)
        
        feq = self.W *(1.0)
        #feq = self.W*(0.3)
        pad = tf.zeros_like(f_mobius[:,0:1])+feq
        f_mobius = tf.concat(axis=1, values=[pad,   f_mobius, pad]) 
        
        pad = tf.zeros_like(f_mobius[:,:,0:1])+feq
        f_mobius = tf.concat(axis=2, values=[pad, f_mobius, pad])
        pad = tf.zeros_like(f_mobius[:,:,:,0:1])+feq
        f_mobius = tf.concat(axis=3, values=[pad, f_mobius, pad])

        return f_mobius
    
    @tf.function
    def Surfaceforces(self,ObjMask,cycletime):
        F = self.Cs*simple_conv(self.F*(1-ObjMask), self.C)
        return F
    
    def ApplyForces(self,Force,cycletime):
        #F = m*a
        #u = u +a*t
        #Fi = self.F
        u = self.FrameVelx
        ma = 978.0*self.swimmass #assumes dimensionless mass of 1
        
        #Ftot = F2+F1
        Fsum = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(Force,axis=1),axis=1),axis=1)
        u = u+(Fsum[:,0]/ma)*(self.dt*cycletime)
        #u = u*0+1.5 #use this to set constant velocity
        self.FrameVelx.assign(u)
        
        self.RollFrame(u)
        
        return

    #@tf.function
    def RollFrame(self, u):
        #Allows for a continuose roll of the grid, breaking it int
        F = self.F
        Vels = self.Vel
        Rho = self.Rho
        initialshift = int(tf.math.floor(u))
        
        F = tf.roll(F,shift=initialshift,axis=1)
        Vels = tf.roll(Vels,shift=initialshift,axis=1)
        Rho = tf.roll(Rho,shift=initialshift,axis=1)
        
        remainder = u-initialshift
        rolldirect = int(tf.where(remainder > 0,1,-1))
        F2 = tf.roll(F,shift=rolldirect,axis=1)
        Vel2 = tf.roll(Vels,shift=rolldirect,axis=1)
        Rho2 = tf.roll(Rho,shift=rolldirect,axis=1)
        
        F = F2*(tf.cast(remainder,F2.dtype))+F*tf.cast((1-tf.abs(remainder)),F.dtype)
        Vels = Vel2*(tf.cast(remainder,Vel2.dtype))+Vels*tf.cast((1-tf.abs(remainder)),F.dtype)
        Rho = Rho2*(tf.cast(remainder,Rho2.dtype))+Rho*tf.cast((1-tf.abs(remainder)),F.dtype)
        
        self.F.assign(F)
        self.Vel.assign(Vels)
        self.Rho.assign(Rho)
        

        return 
    
def pad_mobius(f):
    f_mobius = f
    f_mobius = tf.concat(axis=1, values=[f_mobius[:,-1:],   f_mobius, f_mobius[:,0:1]]) 
    f_mobius = tf.concat(axis=2, values=[f_mobius[:,:,-1:], f_mobius, f_mobius[:,:,0:1]])
    if len(f.get_shape()) == 5:
        f_mobius = tf.concat(axis=3, values=[f_mobius[:,:,:,-1:], f_mobius, f_mobius[:,:,:,0:1]])
    
    return f_mobius
    
    
def simple_conv(x, k):
    #A simplified 2D or 3D convolution operation
    if   len(x.get_shape()) == 4:
        y = tf.nn.conv2d(x, k, [1, 1, 1, 1],    padding='VALID')
    elif len(x.get_shape()) == 5:
        y = tf.nn.conv3d(x, k, [1, 1, 1, 1, 1], padding='VALID')
    return y
    
def SaveVideo(grid,i,cycle,video,scalemax,U,V,zheight=5,debug=False):
    shape = grid.shape
    frame = grid.Vel.numpy()
    obj = grid.ObjVels[cycle].numpy()
    mask = tf.squeeze(grid.MIndex).numpy()
    
    u = frame[0,:,:,zheight,0]
    v = frame[0,:,:,zheight,1]
    U = np.append(U,np.sum(u))
    V = np.append(V,np.sum(v))
    
    if debug:
        Fis = grid.F.numpy()
        Rho = grid.Rho.numpy()
        print('Rho Max' +str(np.max(Rho)))
        print('Rho Min' +str(np.min(Rho)))
        print('Fi Max: '+str(np.max(np.abs(Fis))))
        print('Frame Shape: '+str(frame.shape))

    frame = np.sum(np.abs(frame[0,:,:,:,:]),axis=-1)
    obj = np.sum(np.abs(obj[0,:,:,:,:]),axis=-1) #Show the MTF Location
    
    frame[mask!=0]=0
    obj[obj!=0]=scalemax
    #print('mtf size' + str(obj[obj!=0].size))
    
    frame = frame+obj #Combine MTF Mask with Velocity Frame

    #Where in the slice we are looking 
    frame = np.average(frame,axis=-1)
    
    if (scalemax != 0):
        frame= scalebar(frame,shape,size=10,smax=scalemax)
        frame[np.isnan(frame)]=0.0
        frame[frame>=scalemax]=scalemax
        frame = 255 * (frame/scalemax)
        frame[frame>=255]=255
        frame = np.uint8(frame)
        #frame = cv2.applyColorMap(frame, 2)
        frame = apply_custom_colormap(frame)
        frame = cv2.resize(frame, (shape[1]*3, shape[0]*3+20))
        
    else:    
        frame[np.isnan(frame)]=0.0
        frame = 255 * (frame/np.max(frame))
        frame[frame>=255]=255
        frame = np.uint8(frame)
        #frame = cv2.applyColorMap(frame, 2)
        frame = apply_custom_colormap(frame)
        frame = cv2.resize(frame, (shape[1]*3, shape[0]*3))
    
    video.write(frame)
    return frame, U,V

def colorpallette():
    #New Gray-> Blue Colormap
    cbits = 256
    vals = np.ones((cbits, 4))
    vals[:, 0] = np.linspace(245./256,20./256, cbits)
    vals[:, 1] = np.linspace(245./256,20./256, cbits)
    vals[:, 2] = np.linspace(245./256,120./256,  cbits)
    cpalette = ListedColormap(vals)
    return cpalette

def apply_custom_colormap(image_gray):
    assert image_gray.dtype == np.uint8, 'must be np.uint8 image'
    if image_gray.ndim == 3: image_gray = image_gray.squeeze(-1)

    # Initialize the matplotlib color map
    cms = colorpallette()
    sm = matplotlib.cm.ScalarMappable(cmap=cms)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256))[:,0:3]    # color range RGBA => RGB
    color_range = (color_range*255.0).astype(np.uint8)         # [0,1] => [0,255]
    color_range = np.squeeze(np.dstack([color_range[:,2], color_range[:,1], color_range[:,0]]), 0)  # RGB => BGR

    # Apply colormap for each channel individually
    channels = [cv2.LUT(image_gray, color_range[:,i]) for i in range(3)]
    return np.dstack(channels)

def scalebar(frame,shape,size=20,smax=10):
    a = np.zeros((shape[0],size))
    for i in range(0,a.shape[1]):
        a[:,i] = np.linspace(0,smax,shape[0])
    frame = np.concatenate((frame,a),axis=1)
    return frame



    #@tf.function
def InitialVel(shape):
    #velInit = np.ones((shape[0],shape[1],shape[2],3))
    print("initializing Velocities")
    #velInit = np.random.randint(-100,100,size=(1,shape[0],shape[1],shape[2],3))*0.00002
    velInit = np.zeros([1,shape[0],shape[1],shape[2],3])

    return tf.convert_to_tensor(velInit.astype(np.float16))
    
def run(grid,num_steps,video,cycletime=30,savenum=10,debug=False,scalemax=0.0,streamtime=2):
    
    U = np.array([])
    V = np.array([])
    #Gives initial Fi values
    grid.InitializeGrid()
    Fi = grid.F.numpy()
    if debug:
        print('Max Start Fi:' + str(np.max(Fi)))
        print('Max Min Fi:' + str(np.min(Fi)))
        print('cycletime:' + str(cycletime))
    
    timestep = 0
    
    #Main Loop for program
    for i in tqdm(range(0,num_steps),position=0, leave=True):

        cycle = timestep%cycletime      
        ObjVel = grid.ObjVels[cycle,:,:,:,:,:3]
        ObjMask = 1-tf.cast(tf.expand_dims(grid.ObjVels[cycle,:,:,:,:,3],axis=-1),tf.float16)

        for loop in range(0,streamtime):
                grid.Collide(ObjMask)
                grid.ObjCollide(ObjVel)
                grid.Stream(ObjVel)
        #grid.ObjectStream(ObjVel,ObjMask) # introduce if you have an inlet/ outlet mask in the simulation
        Force = grid.Surfaceforces(ObjMask,cycletime)
        grid.ApplyForces(Force,cycletime)
        
        grid.evalOutFlux(i)
        
        if i%savenum==0:
            #Saves a video of Grid
            frame,U,V = SaveVideo(grid,i,cycle,video,scalemax,U,V,debug=debug)
        timestep +=1


        #Save Last wave
        if i >= num_steps-cycletime:
            grid.ObjVels[cycle,:,:,:,:,:3].assign(grid.Vel*(ObjMask))
            
    return U,V


# In[17]:


def main():
    print(tf.__version__)
    tf.compat.v2.keras.backend.clear_session
    
    #Initial Variables
    #Area constraints (mm)
    total_grid_size_x = 30.0 #mm
    total_grid_size_y = 30.0 #mm
    total_grid_size_z = 20.0 #mm

    #Kinematic Viscosity - water at 37 deg = 6.969e-7 m^2/s
    vis = 6.969e-7*(1000**2) #mm^2/s
    #vis = 0.0000001
    #vis = 0.15

    #Steptime
    dt = 1.0/30.0 #Second per step
    dx = 0.2 #Size of each grid space

    #Stimulation
    Hz = 1.5
    cycletime = int(np.round(Hz/dt))
    print(cycletime)

    #Cleanup From Constants
    xsize = np.round(total_grid_size_x/(float(dx)*2))*2
    ysize = np.round(total_grid_size_y/(float(dx)*2))*2
    zsize = np.round(total_grid_size_z/(float(dx)*2))*2
    shape = np.array([int(xsize), int(ysize), int(zsize)])
    #shape = [128,128,256]
    #tau = tf.constant(0.5 +3*dt*vis/(dx**2))
    print('grid points: ' + str(shape))
    
    
    #Initialize Video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter()
    Vsize = int(3)
    success = video.open('DBG_Flow.avi', fourcc, 30, (shape[1]*Vsize, shape[0]*Vsize+20), True)

    #Initialize Grid - Kinematic Viscocity, Grid Shape, Grid Spacing, Time per frame
    grid = Grid(vis,shape,dx=dx,dt=dt,cycletime=cycletime)

    #Material Grid
    grid.MIndex.assign(grid.ClosedBox(cycletime))

    grid.Vel.assign(InitialVel(grid.shape))
    print(grid.Vel.shape)

    #Run Program - grid, number of iterations, How often to Save
    frame = run(grid,1000,video,savenum=10,cycletime=cycletime,debug=False)
    video.release()
    cv2.destroyAllWindows()


# In[18]:


if __name__ == '__main__':
  main()




