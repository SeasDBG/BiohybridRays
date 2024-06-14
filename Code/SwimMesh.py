#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Generate Swimmer Meshes and Kinematics
#Disease Biophysics Group
#Written by John Zimmerman
#Updated 4/14/20

#Triangle package - https://rufat.be/triangle/installing.html

get_ipython().run_line_magic('matplotlib', 'inline')

import random
from matplotlib.font_manager import FontProperties
from scipy.spatial import Delaunay

from random import randint
from time import time

from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import matplotlib
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import math
from scipy.stats import boltzmann
import scipy.interpolate
from matplotlib.font_manager import FontProperties
from GeoSwimmer import GeoSwimmer
import triangle
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial import ConvexHull


def lerp(a,b,t):
    return (1.0 - t)*a + t*b

def pointdist(x,y,angle):
    #Returns the distance from a point to a line defined by y=tan(alpha)*x, where alpha is in radians
    return np.abs(np.tan(angle)*x-y)/np.sqrt(np.tan(angle)**2+1)

def sigmoid(x,slope=10,intercept=0.45):
    return 1.0/(1.0+np.exp(-slope*(x-intercept)))
    
def gaussian(x, mu, sigma):
    return np.exp(-1.0*np.power(x - mu, 2.0) / (2.0 * np.power(sigma, 2.0)))

def generateMeshHalf(SwimName,sample=50,gscale=5,arg='pqa0.5'):
    
    #Generate the mesh of a single fin
    t = np.linspace(0,1,100)
    swimX,swimY = GeoSwimmer.readDNAXY(SwimName)
    swimX,swimY = swimX*gscale,swimY*gscale
    swimX = swimX-np.average(swimX)
    
    swimshapeY = UnivariateSpline(t,swimY,k=3,s=.00001)
    swimshapeX = UnivariateSpline(t,swimX,k=3,s=.00001)
    e = np.linspace(0,1,sample)
    
    #Evenly Space vertices along boarder
    dx = swimshapeX.derivative()
    dy = swimshapeY.derivative()
    z = np.cumsum(np.sqrt(dx(t)**2+dy(t)**2))
    z = z/z.max()
    z = np.append(np.array([0]),z)
    t = np.linspace(0,1,z.size)
    arcspline = np.interp(e,z,t)

    vertsize = sample
    #Generate Vertices and segment list - vertices are anchor points, and segmentsa are bounding boxes listing vertice numbers
    verts = np.zeros((vertsize+1,2))
    segmap = np.zeros((vertsize+1,2),dtype=int)

    for i in range(0,vertsize):
        verts[i,0]=swimshapeX(arcspline[i])
        verts[i,1]=swimshapeY(arcspline[i])
        segmap[i,0]=int(i)
        segmap[i,1]=int(i+1)
    verts[-1,0] = 0
    verts[-1,1] = 0
    segmap[-1,0] = int(vertsize)

    vertdict = dict(vertices=verts,segments=segmap)
    tri = triangle.triangulate(vertdict,arg)
    return tri, vertdict

def generateMesh(SwimName,sample=30,gscale=5,arg='pqa0.5'):
    #Generate the mesh for the full swimmer shape (both fins)
    t = np.linspace(0,1,100)
    swimX,swimY = GeoSwimmer.readDNAXY(SwimName)
    swimX,swimY = swimX*gscale,swimY*gscale
    swimX = swimX-np.average(swimX)

    swimshapeY = UnivariateSpline(t,swimY,k=3,s=.000001)
    swimshapeX = UnivariateSpline(t,swimX,k=3,s=.000001)
    
    
    #Evenly Space vertices along boarder
    e = np.linspace(0,1,sample)
    dx = swimshapeX.derivative()
    dy = swimshapeY.derivative()
    #Arclength Function
    z = np.cumsum(np.sqrt(dx(t)**2+dy(t)**2))
    z = z/z.max()
    z = np.append(np.array([0]),z)
    t = np.linspace(0,1,z.size)
    arcspline = np.interp(e,z,t)

    vertsize = sample
    
    #Generate Vertices and segment list - vertices are anchor points, and segmentsa are bounding boxes listing vertice numbers
    verts = np.zeros((vertsize*2-2,2))
    segmap = np.zeros((vertsize*2-2,2),dtype=int)
    
    for i in range(0,vertsize*2-2):
        if i < vertsize:
            #verts[i,0]=swimshapeX(arcspline(e[i]))

            #verts[i,1]=swimshapeY(arcspline(e[i]))
            verts[i,0]=swimshapeX(arcspline[i])

            verts[i,1]=swimshapeY(arcspline[i])
        segmap[i,0]=int(i)
        segmap[i,1]=int(i+1)
    verts[vertsize:,0] = verts[vertsize-2:0:-1,0]
    verts[vertsize:,1] = -verts[vertsize-2:0:-1,1]
    
    segmap[-1,1] = int(0)
    vertdict = dict(vertices=verts,segments=segmap)
    tri = triangle.triangulate(vertdict,arg)
    return tri, vertdict
    
def circleProject(x,r):
    #Project onto a circle coordinate for thin film contraction
    xnew = np.zeros(x.size)
    z = np.zeros(x.size)
    for i in range(0,x.size):
        #circ = 2*math.pi*r
        theta = x[i]/r
        #theta = (x[i]/circ)*2
        if theta < np.pi/2:
            xnew[i] = math.sin(theta)*r
            z[i] = r-math.cos(theta)*r
        else:
            xnew[i] = math.cos(theta-np.pi/2)*r
            z[i] = r+math.sin(theta-np.pi/2)*r
    return xnew,z
    
def contractTime(x,waves):
    #converts linear x into seesaw pattern, given the number of waves over 2-Pi
    return 0.499*np.sin((x*waves*np.pi/4-(1/2)*(np.sin(x*waves*np.pi/4)))+.25*np.pi)+0.50


#Find Which Grid Points are in Mesh
def find_grid(xnew, ynew, znew, triangles, grid, VertNorms,thic=0.04):
    zlist = np.zeros(len(grid[:,0]))
    #thic = .04
    
    xmax = 1.1*(np.max(xnew)+thic)
    ymax = 1.1*(np.max(ynew)+thic)
    zmax = 1.1*(np.max(znew)+thic)
    indexlist = np.arange(grid[:,0].shape[0])

    prunelist =  indexlist[grid[:,0]<=xmax]
    prunelist = prunelist[grid[prunelist,1]<=ymax]
    prunelist = prunelist[grid[prunelist,2]<=zmax]

    #Check Whole Swimmer

    swimverts = np.zeros((len(xnew)*2,3))
    swimverts[:,0] = np.append(xnew+thic*VertNorms[:,0],xnew-thic*VertNorms[:,0])
    swimverts[:,1] = np.append(ynew+thic*VertNorms[:,1],ynew-thic*VertNorms[:,1])
    swimverts[:,2] = np.append(znew+thic*VertNorms[:,2],znew-thic*VertNorms[:,2])
    hull = Delaunay(swimverts)
    #print(grid[:,0].shape[0])
    #print('Narrowing Down Points...')

    indexlist = indexlist[prunelist]
    print(indexlist.size)

    subindex = np.array([],dtype=int)
    for i in indexlist:
        pnt = grid[i,:]
        if IsInHull3D(pnt[0],pnt[1],pnt[2], hull):
            subindex = np.append(subindex,int(i))

    #Check Individual Triangles
    for j in range(0,len(triangles)):
        tripoly = np.zeros((6,3))
        k=0
        tripoly[:,k] = np.append(xnew[triangles[j]]+thic*VertNorms[triangles[j],k],xnew[triangles[j]]-thic*VertNorms[triangles[j],k])
        k=1
        tripoly[:,k] = np.append(ynew[triangles[j]]+thic*VertNorms[triangles[j],k],ynew[triangles[j]]-thic*VertNorms[triangles[j],k])
        k=2
        tripoly[:,k] = np.append(znew[triangles[j]]+thic*VertNorms[triangles[j],k],znew[triangles[j]]-thic*VertNorms[triangles[j],k])
        hull = Delaunay(tripoly)
        #print(j)
        zmax = np.max(tripoly[:,2])
        zmin = np.min(tripoly[:,2])
        xmax = np.max(tripoly[:,0])
        xmin = np.min(tripoly[:,0])
        ymax = np.max(tripoly[:,1])
        ymin = np.min(tripoly[:,1])
        i=0
        for i in subindex:
            pnt = grid[i,:]
            if zmin<pnt[2] and pnt[2]<zmax and pnt[0]<xmax and pnt[0]>xmin and pnt[1]<ymax and pnt[1]>ymin:
                if IsInHull3D(pnt[0],pnt[1],pnt[2], hull):
                    zlist[i] = j
                    #print('found')
                    subindex = subindex[subindex!=i]
    zlist[zlist==-10] = 0
    print(str(zlist[zlist!=0].shape[0])+' points total')
    return zlist


#Returns the Normal Vectors of a set of Input Triangles, given x,y,z and triangle list (nodes)
#Inverse returns the inverted normal vectors at the node
#Trinorm returns the norm of the triangle faces, given as the x,y,z coordinate of those nodes, and then the vectors
def ContstructVertNorms(xnew,ynew,znew,triangles,inverse=False,TriNorms=False):
    TriNorm = np.zeros((len(triangles[:,0]),3))
    Vec1 = np.zeros((len(triangles[:,0]),3))
    Vec2 = np.zeros((len(triangles[:,0]),3))

    #Construct Vector1
    #X Vector
    Vec1[:,0] = xnew[triangles[:,1]]-xnew[triangles[:,0]]
    #Y Vecotr
    Vec1[:,1] = ynew[triangles[:,1]]-ynew[triangles[:,0]]
    #Z Vector
    Vec1[:,2] = znew[triangles[:,1]]-znew[triangles[:,0]]

    #Construct Vector2
    #X Vector
    Vec2[:,0] = xnew[triangles[:,2]]-xnew[triangles[:,0]]
    #Y Vector
    Vec2[:,1] = ynew[triangles[:,2]]-ynew[triangles[:,0]]
    #Z Vector
    Vec2[:,2] = znew[triangles[:,2]]-znew[triangles[:,0]]


    #Construct a Normal Vecotr or Each Triangle
    TriNorm = np.cross(Vec2,Vec1) if inverse else np.cross(Vec1,Vec2)
    TriNorm = np.array([v/np.linalg.norm(v) for v in TriNorm])


    #Position of Those Vecotors
    Trix = np.average((xnew[triangles[:,0]],xnew[triangles[:,1]],xnew[triangles[:,2]]),axis=0)
    Triy = np.average((ynew[triangles[:,0]],ynew[triangles[:,1]],ynew[triangles[:,2]]),axis=0)
    Triz = np.average((znew[triangles[:,0]],znew[triangles[:,1]],znew[triangles[:,2]]),axis=0)

    #Triangles Touching a Given Vertex
    #list of Triangles
    TrisTouchingVert = np.array([],dtype=object)
    
    #Index for that List
    TrisTouchingVertIndex = np.zeros(len(xnew))
    for i in range(0,len(xnew)):
        k=0
        for j in range(0,len(triangles)):
            if np.isin(i,triangles[j,:]):
                TrisTouchingVert = np.append(TrisTouchingVert,j)
                k+=1
        TrisTouchingVertIndex[i] = k
        #print(TriList)
        #TrisTouchingVert = np.append(TrisTouchingVert,TriList)


    VertNorms = np.zeros((len(xnew),3))
    for i in range(0,len(xnew)):
        TrisTouchingVert = np.array([],dtype=int)
        for j in range(0,len(triangles)):
            if np.isin(i,triangles[j,:]):
                TrisTouchingVert = np.append(TrisTouchingVert,j)
        VertNorms[i,0]= np.average(TriNorm[TrisTouchingVert,0])
        VertNorms[i,1]= np.average(TriNorm[TrisTouchingVert,1])
        VertNorms[i,2]= np.average(TriNorm[TrisTouchingVert,2])
    return VertNorms, TriNorm, Trix,Triy,Triz

def angcordtoZ(x,y,r,ang):
    #Returns Z locations of an X,Y array with angle ang on a cricle projection of radius r
    #Angle given in degrees
    shift = np.min(x)
    x = x-shift
    
    xmax = np.max(x)
    xnew = np.zeros(x.size)
    ynew = np.zeros(x.size)
    z = np.zeros(x.size)
    y1 =  np.zeros(x.size)
    x1 =  np.zeros(x.size)
    L0 = np.zeros(x.size)
    rad = math.radians(ang)
    
    if (ang > 90):
        xnew = x+shift
        ynew = y
        print('Out of bounds')
        return xnew,ynew,z
    
    if (ang < -90):
        xnew = x+shift
        ynew = y
        print('Out of bounds')
        return xnew,ynew,z
    
    if (ang == 0):
        l1, z = circleProject(y,r)
        xnew = x+shift
        ynew = l1
        return xnew,ynew,z
    
    if (ang > 0):
        for i in range(0,len(x)):

            if (y[i]> (math.tan(rad)*(xmax-x[i]))):
                L0[i] = pointdist(x[i],y[i],rad)
                x1[i] = math.sin(rad)*L0[i]
                y1[i] = math.cos(rad)*L0[i]

    
    if (ang < 0):
        rad = math.radians(np.abs(ang))
        for i in range(0,len(x)):
            if (y[i]> (math.tan(rad)*x[i])):
                L0[i] = pointdist(x[i],y[i],rad)
                x1[i] = math.sin(rad)*L0[i]
                y1[i] = math.cos(rad)*L0[i]


    
    L1 , z = circleProject(L0,r)

    x2 = L1*math.sin(rad)
    y2 = L1*math.cos(rad)
    if ang>0:
        xnew = x-(x1-x2)+shift
        ynew = y-(y1-y2)
    else:
        xnew = x+(x1-x2)+shift
        ynew = y-(y1-y2)
    #print(xnew.size)
    return xnew,ynew,z

def IsInHull3D(x,y,z, hull):
    '''
    Checks if `pnt` is inside the convex hull.
    `hull` -- a QHull ConvexHull object
    `pnt` -- point array of shape (3,)
    '''
    pnt = np.array([[x,y,z]])
    if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

    return hull.find_simplex(pnt)>=0


def genGrid(xsize,ysize,zsize,shape):
    #Converts lattice points to grid units
    x = np.linspace(-xsize/2, xsize/2, shape[0])
    #y = np.linspace(-ysize/3, ysize*2/3, shape[1])
    y = np.linspace(0, ysize, shape[1])
    z = np.linspace(0, zsize, shape[2])
    xv, yv,zv = np.meshgrid(x, y,z)
    grid = np.zeros((xv.size,3))
    grid[:,0] = xv.reshape(-1)
    grid[:,1] = yv.reshape(-1)
    grid[:,2] = zv.reshape(-1)
    return grid


def reduce_slant(verts,angle):
    #Reduction in relaxation based on area contracted
    free = 0.0
    xmin = np.min(verts[:,0])
    ymin = np.min(verts[:,1])
    
    for vert in verts:
        if (vert[1]-ymin)>(vert[0]-xmin)*np.tan(np.radians(np.abs(angle))):
            free +=1.0
    #Prevent divide by zero errors, although origin should fall within coordspace
    if free <= 1.0:
        free = 1.0
    
   # return ((float(verts.shape[0])/free)-1.0)*3.14
    return (float(verts.shape[0])/free)
    #return 1.0

def genVideo(swimname,savepath,slant = 0, basename='swim',frames=60,circleamp=2.0,circlebase=2.15):
    tri,vertdict = generateMeshHalf(swimname,40)
    if circleamp>circlebase:
        print('WARNING: Circle base should be greater than amplitude')
        return

    #Generate Flapping Video
    a = np.array(tri['vertices'])
    triangles = tri['triangles']
    swimheight = np.zeros(int(np.size(tri['vertices'])/2))
    frames = 60
    slantangle = slant
    for i in range(0,frames):
        ynew = np.zeros(a[:,0].size)
        radiuscircle = 2*np.sin(math.radians(6*360*(float(i)/frames)))+2.15


        xnew,ynew, swimheight = angcordtoZ(a[:,0],a[:,1],radiuscircle,slantangle)

        #print(np.max(ynew))
        angle = 360*(float(i)/frames)
        xmin= -1.2
        xmax = 1.2
        ymin = -1.2
        ymax = 1.2
        zmax = 1.2
        zmin = -1.2
        clf()
        fig = figure(figsize=(8,8))
        ax3 = fig.add_subplot(1, 1, 1, projection='3d')
        #ax3 = fig.gca(projection='3d')
        ax3.plot([xmin,xmax],[ymax,ymax],[zmin,zmin],'k-')
        ax3.plot([xmin,xmin],[ymin,ymax],[zmin,zmin],'k-')
        ax3.plot([xmin,xmin],[ymax,ymax],[zmin,zmax],'k-')
        ax3.set_xlim(xmin,xmax)
        ax3.set_ylim(ymin,ymax)
        #ax3.plot(movAvg(prtX-np.average(prtX),n), movAvg(prtY-np.average(prtY),n),movAvg(t/60,n),linewidth=2)
        #ax3.plot(a[:,0], a[:,1],swimheight,'k.')
        ax3.plot(xnew, -ynew,swimheight,'k.')
        ax3.plot(xnew, ynew,swimheight,'k.')
        for j in range(0,int(triangles.size/3)):
            ax3.plot(xnew[triangles[j]],ynew[triangles[j]],swimheight[triangles[j]],'r',linewidth=1)
            ax3.plot(xnew[triangles[j]],-ynew[triangles[j]],swimheight[triangles[j]],'r',linewidth=1)
        ax3.view_init(30, angle)
        ax3.set_zticklabels([])
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        fig.subplots_adjust(left=0, right=1.0, bottom=0, top=1)
        #bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)

        savefig(savepath +basename +str(i)+'.png',dpi=250)
    return


def solveContraction(verts,minr,maxr,slantangle,slantr,frame,totalframes,profilespline,difspline):
    #verts = [n,3] array of vertice coordinates given in x,y,z
    #minr,maxr = minimum and maximum radius in mm
    #slantr = slantreduction for angled cantielvers, accounts for lost area (should be >1.0)
    #Frame = current frame number
    #totalframe = total number of frames for the curve
    #profilespline = spline of the contraction profile (from 0-1)
    #slantangle = angle of the slope, given in degree, between 90 to -90
    
    #Moves between the maximum and minimum radius dependeing on where it is in the contraction profile 
    radiuscircle = lerp(minr,maxr,profilespline(frame/float(totalframes)))
    
    slantx,slanty, slantz = angcordtoZ(verts[:,0],verts[:,1],radiuscircle,slantangle)
    straighty , straightz = circleProject(verts[:,1],radiuscircle*slantr)
       
    
    t = difspline(float(frame/float(totalframes)))
    
    
    cutoff = 0.95
    if t>cutoff:
        t = cutoff
    elif t<0:
        t = 0
    
    slanty = lerp(slanty,straighty,t)
    slantz = lerp(slantz,straightz,t)
    slantx = lerp(slantx,verts[:,0],t)
    
    return slantx,slanty,slantz

import is_point

def SearchUpperVerts(j,indexlist,zlist,xnew, ynew, znew, triangles, grid, VertNorms,thic):
    tripoly = np.zeros((6,3))
    k=0
    tripoly[:,k] = np.append(xnew[triangles[j]]+thic*VertNorms[triangles[j],k],xnew[triangles[j]])
    k=1
    tripoly[:,k] = np.append(ynew[triangles[j]]+thic*VertNorms[triangles[j],k],ynew[triangles[j]])
    k=2
    tripoly[:,k] = np.append(znew[triangles[j]]+thic*VertNorms[triangles[j],k],znew[triangles[j]])
    #hull = Delaunay(tripoly)

    #Get Boundaries of Triangle
    zmax = np.max(tripoly[:,2])
    zmin = np.min(tripoly[:,2])
    xmax = np.max(tripoly[:,0])
    xmin = np.min(tripoly[:,0])
    ymax = np.max(tripoly[:,1])
    ymin = np.min(tripoly[:,1])

    #Bounding Box of Grid in area of Triangle
    prunelist =  indexlist[grid[:,0]<=xmax]
    prunelist = prunelist[grid[prunelist,0]>=xmin]
    prunelist = prunelist[grid[prunelist,1]<=ymax]
    prunelist = prunelist[grid[prunelist,1]>=ymin]
    prunelist = prunelist[grid[prunelist,2]<=zmax]
    prunelist = prunelist[grid[prunelist,2]>=zmin]

    #Remove points previosely found
    prunelist = prunelist[zlist[prunelist]==0]

    #Find Points in Triangle
    pnt = grid[prunelist,:]

    boolmask = is_point.is_point(tripoly,pnt)
    
    return boolmask,prunelist

def SearchLowerVerts(j,indexlist,zlist,xnew, ynew, znew, triangles, grid, VertNorms,thic):
    tripoly = np.zeros((6,3))
    k=0
    tripoly[:,k] = np.append(xnew[triangles[j]],xnew[triangles[j]]-thic*VertNorms[triangles[j],k])
    k=1
    tripoly[:,k] = np.append(ynew[triangles[j]],ynew[triangles[j]]-thic*VertNorms[triangles[j],k])
    k=2
    tripoly[:,k] = np.append(znew[triangles[j]],znew[triangles[j]]-thic*VertNorms[triangles[j],k])
    #hull = Delaunay(tripoly)

    #Get Boundaries of Triangle
    zmax = np.max(tripoly[:,2])
    zmin = np.min(tripoly[:,2])
    xmax = np.max(tripoly[:,0])
    xmin = np.min(tripoly[:,0])
    ymax = np.max(tripoly[:,1])
    ymin = np.min(tripoly[:,1])

    #Bounding Box of Grid in area of Triangle
    prunelist =  indexlist[grid[:,0]<=xmax]
    prunelist = prunelist[grid[prunelist,0]>=xmin]
    prunelist = prunelist[grid[prunelist,1]<=ymax]
    prunelist = prunelist[grid[prunelist,1]>=ymin]
    prunelist = prunelist[grid[prunelist,2]<=zmax]
    prunelist = prunelist[grid[prunelist,2]>=zmin]

    #Remove points previosely found
    prunelist = prunelist[zlist[prunelist]==0]

    #Find Points in Triangle
    pnt = grid[prunelist,:]

    boolmask = is_point.is_point(tripoly,pnt)
    
    return boolmask, prunelist

def fast_find_grid(xnew, ynew, znew, triangles, grid, VertNorms,thic=0.04):
    zlist = np.zeros(len(grid[:,0]))
    indexlist = np.arange(grid[:,0].shape[0])
    
    for j in range(0,len(triangles)):

        
        boolmask,prunelist = SearchUpperVerts(j,indexlist,zlist,xnew, ynew, znew, triangles, grid, VertNorms,thic)
        zlist[prunelist[boolmask==True]]=j
        
        boolmask,prunelist = SearchLowerVerts(j,indexlist,zlist,xnew, ynew, znew, triangles, grid, VertNorms,thic)
        zlist[prunelist[boolmask==True]]=-1*j

                
    return zlist


def genMTFVideo(width,height,savepath, minr,maxr,contractionspline,difspline,slant = 0, basename='swim',frames=60,norms=False, transparent=False):
    tri,vertdict = generateMTFMesh(width,height,sample=30,arg='pqa0.5')

    #Generate Flapping Video
    relaxframe = int(0.3*frames)
    verts = np.array(tri['vertices'])
    triangles = tri['triangles']
    diastole = False
    slantr = reduce_slant(verts,slant)
    print(slantr)

    #Record Kinematics of Stroke
    dx = np.array([])
    dy = np.array([])
    dz = np.array([])
    for i in range(0,frames):
        if i == relaxframe:
                diastole = True
        
        
        #Returns coordinate kinematics
        xnew,ynew,znew = solveContraction(verts,minr,maxr,slant,slantr,i,frames,contractionspline,difspline,relaxframe,diastole)
        
        
        #Track Kinematics
        if i ==0:
            xold =xnew 
            yold =ynew 
            zold =znew
        
        dx = np.append(dx,np.sum(xnew-xold))
        dy = np.append(dy,np.sum(ynew-yold))
        dz = np.append(dz,np.sum(znew-zold))
        

        
        if norms:
            VertNorms, TriNorm, Trix,Triy,Triz = ContstructVertNorms(xnew,ynew, znew,triangles,inverse=False)
            if i ==0:
                trixold,triyold,trizold = Trix,Triy,Triz

            U = np.sqrt((Trix-trixold)**2+(Triy-triyold)**2+(Triz-trizold)**2)         
            #U = np.concatenate(([0],U))
            
            if i ==0:
                Xsign = np.ones_like(U)
                Ysign = -1*np.ones_like(U)
                Zsign = np.ones_like(U)
                
            else:
                #Xsign = binary_sign(np.concatenate([1],Trix-xold))
                Xsign = binary_sign(Trix-trixold)
                #Ysign = -1*binary_sign(np.concatenate([1],Triy-yold))
                Ysign = -1*binary_sign(Triy-triyold)
                #Zsign = binary_sign(np.concatenate([1],Triz-zold))
                Zsign = binary_sign(Triz-trizold)
            
            trixold,triyold,trizold = Trix,Triy,Triz
            
            
        #Plot Changes
        angle = 360*(float(i)/frames)
        xmin= -3
        xmax = 3
        ymin = -0
        ymax = 6
        zmax = 3
        zmin = -3
        clf()
        fig = figure(figsize=(8,8))
        ax3 = fig.add_subplot(1, 1, 1, projection='3d')
        #ax3 = fig.gca(projection='3d')
        ax3.plot([xmin,xmax],[ymax,ymax],[zmin,zmin],'k-')
        ax3.plot([xmin,xmin],[ymin,ymax],[zmin,zmin],'k-')
        ax3.plot([xmin,xmin],[ymax,ymax],[zmin,zmax],'k-')
        ax3.set_xlim(xmin,xmax)
        ax3.set_ylim(ymin,ymax)
        ax3.set_zlim(zmin,zmax)
        #ax3.plot(movAvg(prtX-np.average(prtX),n), movAvg(prtY-np.average(prtY),n),movAvg(t/60,n),linewidth=2)
        #ax3.plot(a[:,0], a[:,1],swimheight,'k.')
        #ax3.plot(xnew, -ynew,znew,'k.')
        ax3.plot(xnew, ynew,znew,'k.')
        #ax3.plot(hullx,hully,hullz,'r.')
        for j in range(0,int(triangles.size/3)):
            ax3.plot(xnew[triangles[j]],ynew[triangles[j]],znew[triangles[j]],'r',linewidth=1)
            #ax3.plot(xnew[triangles[j]],-ynew[triangles[j]],znew[triangles[j]],'r',linewidth=1)
        if norms:
            #M = U
            print(U.max())
            print(U.min())
            scalemin = 0
            scalemax = .18
            cms = matplotlib.cm.jet
            C = (U-scalemin)/scalemax
            C = cms(C)
            #C = sm(U)
             
            ax3.quiver(Trix,Triy,Triz, TriNorm[:,0]*U*Xsign, TriNorm[:,1]*U*Ysign, TriNorm[:,2]*U*Zsign, colors=C ,arrow_length_ratio =0.15,linewidth=1.5,normalize=True)
            #ax3.quiver(Trix,Triy,Triz, TriNorm[:,0]*U*Xsign, TriNorm[:,1]*U*Ysign, TriNorm[:,2]*U*Zsign, normalize=True,arrow_length_ratio =1)

        
        #ax3.view_init(30, angle)
        ax3.view_init(90, 0)
        #ax3.view_init(30, 260)
        ax3.set_zticklabels([])
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        if transparent:
            ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax3.set_axis_off()
        fig.subplots_adjust(left=0, right=1.0, bottom=0, top=1)
        #bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)

        xold =xnew 
        yold =ynew 
        zold =znew
        
        savefig(savepath +basename +str(i)+'.png',dpi=250)
        close()
    return dx,dy,dz


