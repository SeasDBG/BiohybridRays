#!/usr/bin/env python
# coding: utf-8

# In[108]:


#Geometric Evolution Program
#Disease Biophysics Group
#Written by John Zimmerman
#Updated 11/04/21
#Version 0.3.2

import random
from matplotlib.font_manager import FontProperties

from random import randint
import time

from scipy.interpolate import UnivariateSpline
import matplotlib
import numpy as np
from matplotlib.pyplot import *
import math
from scipy.stats import boltzmann
import scipy.interpolate
from matplotlib.font_manager import FontProperties

class GeoSwimmer:
    global numBaseFunc
    numBaseFunc = 14

    def lerp(a,b,t):
        return (1.0 - t)*a + t*b
    
    def scalesigmoid(x,slope=10):
        #Set maximum aspect ratios
        return 0.5/(1+np.exp(-slope*(x-0.5)))+.5
    
    def seedtoInt(seedArray):
        intSeed = int(0)
        for i in range(0,len(seedArray)):
            intSeed = (intSeed + ord(seedArray[i])*(126**i))%sys.maxsize
        return intSeed

    def Bezier(t,p0,p1,p2,p3):
        return p0*(1-t)**3+3*t*p1*(1-t)**2+3*p2*(1-t)*t**2+p3*t**3

    def BezierP(t,p):
        return p[0]*(1-t)**3+3*t*p[1]*(1-t)**2+3*p[2]*(1-t)*t**2+p[3]*t**3

    def BezierArray(tArray,p0,p1,p2,p3):
        outputArray = np.ones(tArray.size)
        for i in range(0,tArray.size):
            outputArray[i] = GeoSwimmer.Bezier(tArray[i],p0,p1,p2,p3)
        return outputArray

    def BezierArrayP(tArray,p):
        outputArray = np.ones(tArray.size)
        for i in range(0,tArray.size):
            outputArray[i] = GeoSwimmer.Bezier(tArray[i],p[0],p[1],p[2],p[3])
        return outputArray
    
    #Turns DNA into a shape
    def readDNAXY(DNA,BasisFuncs=np.array([]),TargetArea=1):
        if BasisFuncs.size==0:
            BasisFuncs,xBasisFuncs,yBasisFuncs = GeoSwimmer.constructBaseFuncs()
        weights = GeoSwimmer.getWeights(DNA)
        readSplines, AR = GeoSwimmer.translate(DNA,BasisFuncs)
        print(AR)

        t = np.linspace(0,1,100)
        yComb = np.zeros(t.size)
        for j in range(0,len(DNA)):
            for i in range(0,t.size):
                yComb[i] =  yComb[i] + readSplines[j](t[i])*weights[j]
        
        yComb = (yComb-yComb.min())/(yComb.max()-yComb.min())
        yComb = yComb*(1/(2*AR))
        
        #Apply netown's method to find the correct area scaling
        g = 1
        evalnum = 1
        passes = 1
        lastarea = 1
        lastg = 0
        
        while (evalnum>0.0001 and passes<5000):
            x = t*AR*g
            y = yComb*g
            dx = np.average(np.diff(x))
            area = np.trapz(y)*dx
            print('g: '+str(g))
            print('area: '+str(area))
            
            evalnum = (area-TargetArea)**2
            delta = ((area-TargetArea) - (lastarea-TargetArea))/(g-lastg)

            newg = np.abs(g - ((area-TargetArea)/delta))
            lastg = g
            g = newg
            lastarea = area

            passes += 1

        if (passes >=5000):
            print("Shape Converge Failure")
        
        #print(scalingX)
        return x ,y

    #Returns the weights for shape of swimmer
    def getWeights(DNA):
        weights = np.ones(len(DNA))
        m = -1/float(len(DNA))
        b = 1
        for i in range(0,len(DNA)):
            weights[i] = m*i+b
        return weights

    #Returns the weights for scaling
    def getWeightsRev(DNA):
        weights = np.ones(len(DNA))
        if (len(DNA) == 1):
            return weights
        m = 1/(float(len(DNA)))
        b = 0
        for i in range(0,len(DNA)-1):
            weights[i] = (m*(i+1)+b)
        return weights/float(weights.sum())


    def translate(DNA,BasisFuncs):
        #Turns DNA basis functions into splines. turns the DNA into geometry -A/N basis scales Only-
        OutputSplines = np.array([])
        xscaling = 0.5
        weights = GeoSwimmer.getWeightsRev(DNA)
        for i in range(0,len(DNA)):
            if (DNA[i] == "A"):
                OutputSplines = np.append(OutputSplines,BasisFuncs[0])
                xscaling = xscaling - 0.5*weights[i]

            elif (DNA[i] == "B"):
                OutputSplines = np.append(OutputSplines,BasisFuncs[1])
            elif (DNA[i] == "C"):
                OutputSplines = np.append(OutputSplines,BasisFuncs[2])
            elif (DNA[i] == "D"):
                OutputSplines = np.append(OutputSplines,BasisFuncs[3])
            elif (DNA[i] == "E"):
                OutputSplines = np.append(OutputSplines,BasisFuncs[4])
            elif (DNA[i] == "F"):
                OutputSplines = np.append(OutputSplines,BasisFuncs[5])
            elif (DNA[i] == "G"):
                OutputSplines = np.append(OutputSplines,BasisFuncs[6])
            elif (DNA[i] == "H"):
                OutputSplines = np.append(OutputSplines,BasisFuncs[7])
            elif (DNA[i] == "I"):
                OutputSplines = np.append(OutputSplines,BasisFuncs[8])
            elif (DNA[i] == "J"):
                OutputSplines = np.append(OutputSplines,BasisFuncs[9])
            elif (DNA[i] == "K"):
                OutputSplines = np.append(OutputSplines,BasisFuncs[10])
            elif (DNA[i] == "L"):
                OutputSplines = np.append(OutputSplines,BasisFuncs[11])
            elif (DNA[i] == "M"):
                OutputSplines = np.append(OutputSplines,BasisFuncs[12])
            elif (DNA[i] == "N"):
                OutputSplines = np.append(OutputSplines,BasisFuncs[13])
                xscaling = xscaling + 0.5*weights[i]
            else:
                print("Encountered an alien basepair, inserting G")
                print(DNA[i])
                OutputSplines = np.append(OutputSplines,BasisFuncs[6])

        #Final scaling based on sigmoidal fit
        AR = GeoSwimmer.scalesigmoid(xscaling)
        return OutputSplines , AR
    
    def genRandDNA(numBaseP,seedVal=int((time.process_time()*1000))):
        if isinstance(seedVal, (str)):
            seedVal = GeoSwimmer.seedtoInt(seedVal)
        random.seed(seedVal)
        DNA = ''
        #13 is here as the number of basis functions zero indexed
        for i in range(0,numBaseP):
            DNA = DNA + str(GeoSwimmer.intToBaseP(randint(1,numBaseFunc)))
        return DNA

    def intToBaseP(intval):
        return chr(intval+64)
    
    def gScale(y,scalingX):
        area = np.trapz(y)*scalingX
        yz = y/(2*scalingX) 
        return(10*np.sqrt(1/(2*np.trapz(yz))))
    
    def DirectEvolve (DNA,seedVal=int((time.process_time()*1000)),insert_del=True,double=True):
        #Mutate DNA to another strand of DNA
        if isinstance(seedVal, (str)):
            seedVal = GeoSwimmer.seedtoInt(seedVal)
        random.seed(seedVal)
        
        newDNA = ''
        insertcheck = True
        if insert_del:
            insertcheck = False
            #Insertion, Deletion - threshold
            threshold = 5
            maxint = 100
            ins_del_num = randint(0,maxint)

            
            #Insertion Check
            if ins_del_num>(maxint-threshold):
                loc = randint(-1,len(DNA))
                if (loc == len(DNA)):
                    newDNA = DNA + str(GeoSwimmer.intToBaseP(randint(1,numBaseFunc)))
                    #print('Insert1')
                elif (loc == -1):
                    newDNA = str(GeoSwimmer.intToBaseP(randint(1,numBaseFunc)))+DNA
                else:
                    for k in range (0,len(DNA)):
                        newDNA = newDNA + DNA[k]
                        if (k == loc):
                            newDNA = newDNA + str(GeoSwimmer.intToBaseP(randint(1,numBaseFunc)))
            
            #Deletion Check
            elif ins_del_num<threshold:
                if (len(DNA)>1):
                    loc = randint(0,len(DNA)-1)
                    for k in range(0,len(DNA)):
                        if (k != loc):
                            newDNA = newDNA + DNA[k]
                else:
                    newDNA = GeoSwimmer.DirectEvolve(DNA,seedVal+1)
            
            else:
                insertcheck = True
         
        #Replacement
        #Checks if you already had an insertion/deletion event
        if insertcheck:
            loc = randint(0,len(DNA)-1)
            for k in range (0,len(DNA)):
                if (k == loc):
                    newDNA = newDNA + str(GeoSwimmer.intToBaseP(randint(1,numBaseFunc)))
                else:
                    newDNA = newDNA + DNA[k]
                        
        if double:
            #Double mutation weighting - 95
            threshold = 5
            maxint = 100
            if randint(0,maxint)<threshold:
                newDNA = GeoSwimmer.DirectEvolve(newDNA,seedVal+1,insert_del=insert_del)
        
        #Same check
        if newDNA == DNA:
            #print('same')
            newDNA = GeoSwimmer.DirectEvolve(DNA,seedVal+1,insert_del=insert_del,double=double)
        
        #Check that new DNA is not all flat Basis
        checksum = 0
        for char in newDNA:
            if char=='A':
                checksum += 0
            elif char=='N':
                checksum += 0
            else:
                checksum += 1
        if checksum == 0:
            print('**Warning** Flat gDNA detected- Generating New gDNA')
            newDNA = GeoSwimmer.DirectEvolve(DNA,seedVal+1,insert_del=insert_del,double=double)
                
        return newDNA
    
    
    def constructBaseFuncs():
        #Provide points for basic bezier cruves
        numBaseFunc = 14

        t = np.linspace(0,1,100)


        #Define Bezier Basis Functions

        x1 = np.array([0,0,0,1])
        y1 = np.array([0,0,0,0])

        x2 = np.array([0,0,1,1])
        y2 = np.array([0,0,1,0])

        x7 = np.array([0,0,1,1])
        y7 = np.array([0,1,1,0])

        x3 = np.array([0,1,1,1])
        y3 = np.array([0,0,1,0])

        #x5 = np.array([0,1,1,1])
        #y5 = np.array([0,1,0,0])

        x11 = np.array([0,0.5,0,1])
        y11 = np.array([0,0,1,0])

        x8 = np.array([0,1,0,1])
        y8 = np.array([0,1,1,0])

        x13 = np.array([0,0,1,1])
        y13 = np.array([0,1,0,0])

        x10 = np.array([0,.5,0,1])
        y10 = np.array([0,1,0,0])

        x9 = np.array([0,1,0,1])
        y9 = np.array([0,1,0,0])

        x6 = np.array([0,1,0,1])
        y6 = np.array([0,0,1,0])

        x5 = np.array([0,1,.5,1])
        y5 = np.array([0,0,1,0])

        x12 = np.array([0,0,0,1])
        y12 = np.array([0,1,0,0])

        #x13 = np.array([0,0,0,1])
        #y13 = np.array([0,0,1,0])

        x4 = np.array([0,1,0.5,1])
        y4 = np.array([0,1,0,0])

        x14 = np.array([0,0,0,1])
        y14 = np.array([0,0,0,0])


        #Turn them into one array
        x=np.array([])
        x = np.append(x,x1)
        x = np.append(x,x2)
        x = np.append(x,x3)
        x = np.append(x,x4)
        x = np.append(x,x5)
        x = np.append(x,x6)
        x = np.append(x,x7)
        x = np.append(x,x8)
        x = np.append(x,x9)
        x = np.append(x,x10)
        x = np.append(x,x11)
        x = np.append(x,x12)
        x = np.append(x,x13)
        x = np.append(x,x14)

        #Make the Array Readable
        x.shape = (numBaseFunc,4)

        #Repeat for the y values
        y=np.array([])
        y = np.append(y,y1)
        y = np.append(y,y2)
        y = np.append(y,y3)
        y = np.append(y,y4)
        y = np.append(y,y5)
        y = np.append(y,y6)
        y = np.append(y,y7)
        y = np.append(y,y8)
        y = np.append(y,y9)
        y = np.append(y,y10)
        y = np.append(y,y11)
        y = np.append(y,y12)
        y = np.append(y,y13)
        y = np.append(y,y14)

        y.shape = (numBaseFunc,4)


        #Construct a spline to paramaterize values
        splineArray = np.array([])
        for i in range(0,numBaseFunc):
            spline1x = UnivariateSpline(t,GeoSwimmer.BezierP(t,x[i]))
            spline1y = UnivariateSpline(spline1x(t),GeoSwimmer.BezierP(t,y[i]),k=3,s=.0001)
            splineArray = np.append(splineArray,spline1y)
        return splineArray,x,y
    
    def GenerateSwimArray(SeedDNA,SwimNum,GenNum,DNAlength,EvolveDNA=False,printDATA=False,insert_del=True,double=True,saveDATA = ''):
        
        if (printDATA == True):
            print('SwimNum: '+str(SwimNum))
            print('GenNum: '+str(GenNum))
            print('DNALength: '+str(DNAlength))
            print('SeedDNA: '+str(SeedDNA))
            print('EvolveDNA: '+str(EvolveDNA))
            print('')
        
        if (len(saveDATA) != 0):
            SaveTxt = np.array([])
            SaveTxt = np.append(SaveTxt,'SwimNum: '+str(SwimNum))
            SaveTxt = np.append(SaveTxt,'GenNum: '+str(GenNum))
            SaveTxt = np.append(SaveTxt,'DNALength: '+str(DNAlength))
            SaveTxt = np.append(SaveTxt,'SeedDNA: '+str(SeedDNA))
            SaveTxt = np.append(SaveTxt,'EvolveDNA: '+str(EvolveDNA))
            SaveTxt = np.append(SaveTxt,'')
        
        randDNAarray = np.array([])
        
        for i in range(0,SwimNum):
            if (EvolveDNA ==True):
                randDNAarray= np.append( randDNAarray, GeoSwimmer.DirectEvolve(SeedDNA,str(SeedDNA)+str(i)+str(GenNum),insert_del=insert_del,double=double))
            else:
                randDNAarray= np.append( randDNAarray, GeoSwimmer.genRandDNA(DNAlength,GeoSwimmer.seedtoInt(SeedDNA)+i+GenNum))
            j = SwimNum+1
            while randDNAarray[-1] in randDNAarray[:-1]:
                randDNAarray[-1] = GeoSwimmer.DirectEvolve(randDNAarray[-1],str(SeedDNA)+str(i)+str(GenNum)+str('ALT')+str(j),insert_del=insert_del,double=double)
                j = j+1
            
            if (len(saveDATA) != 0):
                SaveTxt = np.append(SaveTxt,'Num: ' + str(i) + ' DNA: ' + str(randDNAarray[-1])) 
                
            if (printDATA == True):
                print('Num: ' + str(i) + ' DNA: ' + str(randDNAarray[-1]))
                
                
            if (len(saveDATA) != 0):
                SaveTxt = np.append(SaveTxt,'')
                np.savetxt(saveDATA+'.txt',SaveTxt,delimiter=" ",fmt="%s")
        
        return randDNAarray
                
    def GenerateSwimmers(SeedDNA,SwimNum,GenNum,DNAlength,EvolveDNA=False,printDATA = True,saveDATA = '',plotSwim=True,plotTxt = False,sorting='Center',SurfaceArea = 50,FontSize = 1.5,WellSizeX=2.9,WellSizeY = 4.2,DPI=600,Conv = 1/25.4,Padding = 0.1):
        #Generate Swimmers v4
        #Sort Method options = Center (only works for 24), LeftUp or Growth 
        #Conv inch to mm
        worldscale = DPI*Conv*(15.4/2)

        if (printDATA == True):

            print('SwimNum: '+str(SwimNum))
            print('GenNum: '+str(GenNum))
            print('DNALength: '+str(DNAlength))
            print('SeedDNA: '+str(SeedDNA))
            print('EvolveDNA: '+str(EvolveDNA))
            print('Sorting: '+str(sorting))
            print('WellsizeX: '+str(WellSizeX))
            print('WellsizeY: '+str(WellSizeY))
            print('DPI: '+str(DPI))
            print('Conv (inch to mm): '+str(Conv))
            print('WorldScale: '+str(worldscale))
            print('Padding: '+str(Padding))
            print('SurfaceArea: '+str(SurfaceArea))
            print('')

        if (len(saveDATA) != 0):
            SaveTxt = np.array([])
            SaveTxt = np.append(SaveTxt,'SwimNum: '+str(SwimNum))
            SaveTxt = np.append(SaveTxt,'GenNum: '+str(GenNum))
            SaveTxt = np.append(SaveTxt,'DNALength: '+str(DNAlength))
            SaveTxt = np.append(SaveTxt,'SeedDNA: '+str(SeedDNA))
            SaveTxt = np.append(SaveTxt,'EvolveDNA: '+str(EvolveDNA))
            SaveTxt = np.append(SaveTxt,'Sorting: '+str(sorting))
            SaveTxt = np.append(SaveTxt,'WellsizeX: '+str(WellSizeX))
            SaveTxt = np.append(SaveTxt,'WellsizeY: '+str(WellSizeY))
            SaveTxt = np.append(SaveTxt,'DPI: '+str(DPI))
            SaveTxt = np.append(SaveTxt,'Conv (inch to mm): '+str(Conv))
            SaveTxt = np.append(SaveTxt,'WorldScale: '+str(worldscale))
            SaveTxt = np.append(SaveTxt,'Padding: '+str(Padding))
            SaveTxt = np.append(SaveTxt,'SurfaceArea: '+str(SurfaceArea))
            SaveTxt = np.append(SaveTxt,'')


        #Setup Storage
        x = np.linspace(0,1,100)
        randDNAarray = np.array([])
        randDNAarray = np.array([])
        scaleArray = np.array([])
        #gscaleArray = np.array([])
        yArraay = np.zeros((SwimNum,x.size))
        yMaxArray = np.array([])
        xMaxArray = np.array([])
        
        if plotTxt:
            font = FontProperties()
            #font.set_style('italic')
            font.set_weight('bold')
            font.set_size(FontSize)

        #Get Swimmers and Attributes
        BasisFuncs = GeoSwimmer.constructBaseFuncs()[0]
        for i in range(0,SwimNum):
            if (EvolveDNA ==True):
                randDNAarray= np.append( randDNAarray, GeoSwimmer.DirectEvolve(SeedDNA,str(SeedDNA)+str(i)+str(GenNum)))
            else:
                randDNAarray= np.append( randDNAarray, GeoSwimmer.genRandDNA(DNAlength,GeoSwimmer.seedtoInt(SeedDNA)+i+GenNum))
            j = SwimNum+1
            while randDNAarray[-1] in randDNAarray[:-1]:
                randDNAarray[-1] = GeoSwimmer.DirectEvolve(randDNAarray[-1],str(SeedDNA)+str(i)+str(GenNum)+str('ALT')+str(j))
                j = j+1

            y,scaling = GeoSwimmer.readDNA(randDNAarray[i],BasisFuncs,SurfaceArea)

            yArraay[i,:]= y
            scaleArray= np.append(scaleArray,scaling)
            yMaxArray = np.append(yMaxArray,np.max(y*scaling*worldscale))
            xMaxArray = np.append(xMaxArray,np.max(x*scaling*worldscale))
            if (len(saveDATA) != 0):
                SaveTxt = np.append(SaveTxt,'Num: ' + str(i) + ' DNA: ' + str(randDNAarray[-1])) 
                
            if (printDATA == True):
                print('Num: ' + str(i) + ' DNA: ' + str(randDNAarray[-1]))
                #print(f'Gscale:{gscaleArray[i]}')
                print('xmax: ' + str(xMaxArray[-1]) + ' ymax: ' + str(yMaxArray[-1]) + ' AR: ' + str(scaling))


        #Determine how to sort the output
        #print(SwimNum)
        if (sorting == "Center"):
            xadjust,yadjust = GeoSwimmer.xysort_centers(SwimNum,xMaxArray,WellSizeX,WellSizeY,DPI)
        else:
            if (sorting == "Growth"):
                xadjust,yadjust = GeoSwimmer.xysort(SwimNum, xMaxArray,yMaxArray,Padding,worldscale,WellSizeX,WellSizeY,DPI,False)
            else:
                xadjust,yadjust = GeoSwimmer.xysort_leftup(SwimNum, xMaxArray,yMaxArray,Padding,worldscale,WellSizeX,WellSizeY,DPI)

        if plotSwim==True:

            fig = figure(figsize=(WellSizeX,WellSizeY))
            ax=fig.add_axes([0,0,1,1])
            ax.set_axis_off()
            for i in range(0,SwimNum):

                x = np.linspace(0,1,100)*scaleArray[i]*worldscale
                y = yArraay[i]*scaleArray[i]*worldscale

                plot(x+xadjust[i],y+yadjust[i],'k',linewidth=0.25)
                plot(x+xadjust[i],-y+yadjust[i],'k',linewidth=0.25)
                if plotTxt:
                    ax.text(xadjust[i]+0.5*scaleArray[i]*worldscale, yadjust[i]-(2.5*FontSize), str(randDNAarray[i]), fontproperties=font,horizontalalignment='center')
            xlim(0,WellSizeX*DPI)
            ylim(0,WellSizeY*DPI)
            if len(saveDATA) != 0:
                savefig(saveDATA+'.png',dpi=DPI)


        if (len(saveDATA) != 0):
            SaveTxt = np.append(SaveTxt,'')
            #SaveTxt = np.append(SaveTxt,'Bezier Curves')
            BasisFuncs,xBasisFuncs,yBasisFuncs = GeoSwimmer.constructBaseFuncs()
            SaveTxt = np.append(SaveTxt,'xBezier Curves')
            for i in xBasisFuncs:
                #print(i)
                SaveTxt = np.append(SaveTxt,str(i))    
            SaveTxt = np.append(SaveTxt,'yBezier Curves')
            for i in yBasisFuncs:
                SaveTxt = np.append(SaveTxt,str(i)) 
            np.savetxt(saveDATA+'.txt',SaveTxt,delimiter=" ",fmt="%s")
        return randDNAarray,scaleArray,xadjust,yadjust
    
    def CheckDNADistro(DNAArray,printData=True):
        a = np.arange((ord('N')-64))
        basis = np.array([])
        for char in a:
            basis = np.append(basis,chr(char+65)) 

        countArray = np.zeros(ord('N')-64)
        for DNA in DNAArray:
            for char in DNA:
                #print(char)
                countArray[ord(char)-65] +=1
        if printData:
            for i in range(0,len(basis)):
                print(f'{basis[i]}: {countArray[i]} - '  + "{0:.2f}".format((countArray[i]/np.sum(countArray))*100)+'%')
    
        return countArray
    
    def PlotSwimmer(y,Scale,t=np.array([]),linecolor = 'k'):
        g = GeoSwimmer.gScale(DNA,Scale)
        if t.size == 0:
            t = np.linspace(0,1,yArraay[0].size)
        yz = g*y*(1/(2*Scale))
        plotp = plot(t*g,yz,linecolor,linewidth=3)
        plotp = plot(t*g,-yz,linecolor,linewidth=3)
        
    def PlotSwimmerArray(yArraay,scaleArray,xadjust=np.array([]),yadjust=np.array([]),x=np.array([]),plotTxt=False,FontSize = 1.5,WellSizeX=2.9,WellSizeY = 4.2,DPI=600,Conv = 1/25.4,Padding = 0.1):
        worldscale = DPI*Conv*(15.4/2)
        if x.size == 0:
            x = np.linspace(0,1,yArraay[0].size)
        if xadjust.size == 0:
            xadjust,yadjust = GeoSwimmer.xysort_centers(scaleArray.size,1,WellSizeX,WellSizeY,DPI)
        for i in range(0,SwimNum):
            plot(x*scaleArray[i]*worldscale+xadjust[i],yArraay[i]*worldscale/scaleArray[i]+yadjust[i],'k',linewidth=0.25)
            plot(x*scaleArray[i]*worldscale+xadjust[i],yArraay[i]*worldscale/scaleArray[i]+yadjust[i],'k',linewidth=0.25)
            if plotTxt:
                    ax.text(xadjust[i]+0.5*scaleArray[i]*worldscale, yadjust[i]-(2.5*FontSize), str(randDNAarray[i]), fontproperties=font,horizontalalignment='center')
    
    def xysort (swimCount, xMaxArray,yMaxarray,padding,worldscale,WellSizeX,WellSizeY,DPI,debug=False):

        #Sorts and organizes the swimmers, returning the x and y postions of the swimmers to print
        #Setup Adjustments

         #Arrange the Fish in packing Order
        #Grows from the bottom left corner
        #Packs from left to right, until the column is taller than it is wide, then it expands to the right, 
        #packing up until it would go over the top, then it adds to the left again. Gives Psuedo dense packing
        xadjust = np.array([0+Padding*worldscale])
        yadjust = np.array([yMaxArray[0]+Padding*worldscale*2])
        xbreak = xMaxArray[0]+Padding*worldscale
        ybreak = yMaxArray[0]*2+Padding*worldscale*2


        xsave = xbreak
        ysave = ybreak
        if ybreak<xbreak:
            if (debug == True):
                print('ybreak<xbreak')

            xcur = Padding*worldscale
            ycur = ysave+Padding*worldscale
        else:
            if (debug == True):
                print('ybreak>xbreak')
            xcur = xbreak
            ycur = Padding*worldscale


        for i in range(1,swimCount):
            if (debug == True):
                print(i)
                print('xbreak: '+str(xbreak)+ ' ybreak: '+str(ybreak))
            if xbreak>ybreak:
                if xcur + xMaxArray[i] + Padding*worldscale < WellSizeX*DPI:
                    if xcur + xMaxArray[i] + Padding*worldscale < xsave:
                        if (debug == True):
                            print('Path 1')
                        xadjust = np.append(xadjust,xcur+Padding*worldscale)
                        yadjust = np.append(yadjust,ycur+yMaxArray[i]+Padding*worldscale)
                        xcur = xcur + xMaxArray[i] + Padding*worldscale
                        if ycur+yMaxArray[i]*2+Padding*worldscale>ysave:
                            ysave = ycur+yMaxArray[i]*2+Padding*worldscale

                    else:
                        if ysave>ybreak:
                            ybreak = ysave

                        if xbreak>ybreak:
                            if (debug == True):
                                print('Path 2')
                            xcur = Padding*worldscale
                            ycur = ysave
                            xadjust = np.append(xadjust,xcur+Padding*worldscale)
                            yadjust = np.append(yadjust,ycur+yMaxArray[i]+Padding*worldscale)
                            xcur = xcur + xMaxArray[i] + Padding*worldscale
                            if ycur+yMaxArray[i]*2+Padding*worldscale>ysave:
                                ysave = ycur+yMaxArray[i]*2+Padding*worldscale
                        else:
                            if xbreak + xMaxArray[i] + Padding*worldscale < WellSizeX*DPI:
                                if (debug == True):
                                    print('Path 3')

                                ybreak = ysave
                                xcur = Padding*worldscale+xbreak
                                ycur = Padding*worldscale

                                xadjust = np.append(xadjust,xcur+Padding*worldscale)
                                yadjust = np.append(yadjust,ycur+yMaxArray[i]+Padding*worldscale)
                                xcur = xcur + xMaxArray[i] + Padding*worldscale
                                if xcur > xsave:
                                    xsave = xcur
                                ysave = ycur+yMaxArray[i]*2+Padding*worldscale
                            else:
                                if (debug == True):
                                    print('Path 4')
                                xcur = Padding*worldscale
                                ycur = ybreak
                                xadjust = np.append(xadjust,xcur+Padding*worldscale)
                                yadjust = np.append(yadjust,ycur+yMaxArray[i]+Padding*worldscale)
                                xcur = xcur + xMaxArray[i] + Padding*worldscale
                                if ycur+yMaxArray[i]*2+Padding*worldscale>ysave:
                                    ysave = ycur+yMaxArray[i]*2+Padding*worldscale 
                else:
                    if (debug == True):
                        print('Path 5')
                    xcur = Padding*worldscale
                    ycur = ysave
                    xadjust = np.append(xadjust,xcur+Padding*worldscale)
                    yadjust = np.append(yadjust,ycur+yMaxArray[i]+Padding*worldscale)
                    xcur = xcur + xMaxArray[i] + Padding*worldscale
                    if ycur+yMaxArray[i]*2+Padding*worldscale>ysave:
                        ysave = ycur+yMaxArray[i]*2+Padding*worldscale

            else:
                if xcur + xMaxArray[i] + Padding*worldscale < WellSizeX*DPI:
                    if xcur + xMaxArray[i] + Padding*worldscale < ybreak:
                        if (debug == True):
                            print('Path 6')

                        xadjust = np.append(xadjust,xcur+Padding*worldscale)
                        yadjust = np.append(yadjust,ycur+yMaxArray[i]+Padding*worldscale)
                        xcur = xcur + xMaxArray[i] + Padding*worldscale
                        if ycur+yMaxArray[i]*2+Padding*worldscale>ysave:
                            ysave = ycur+yMaxArray[i]*2+Padding*worldscale
                        if xcur > xsave:

                            xsave = xcur
                            xbreak = xcur
                            ycur = Padding*worldscale
                            #ysave = ycur+yMaxArray[i]*2+Padding*worldscale

                        if ysave>ybreak:
                            ybreak = ysave
                    else:
                        ycur = ysave
                        if ycur + yMaxArray[i]*2 + Padding*worldscale > ybreak:
                            if (debug == True):
                                print('Path 7')

                            #Transition
                            xbreak = xsave
                            xcur = Padding*worldscale
                            ycur = Padding*worldscale+ybreak
                            xadjust = np.append(xadjust,xcur+Padding*worldscale)
                            yadjust = np.append(yadjust,ycur+yMaxArray[i]+Padding*worldscale)
                            xcur = xcur + xMaxArray[i] + Padding*worldscale
                            ysave = ycur+yMaxArray[i]*2+Padding*worldscale
                            if ysave>ybreak:
                                ybreak = ysave
                        else:
                            if (debug == True):
                                print('Path 8')
                            xcur = xbreak
                            xadjust = np.append(xadjust,xcur+Padding*worldscale)
                            yadjust = np.append(yadjust,ycur+yMaxArray[i]+Padding*worldscale)
                            xcur = xcur + xMaxArray[i] + Padding*worldscale
                            if xcur > xsave:
                                xsave = xcur
                            if ycur+yMaxArray[i]*2+Padding*worldscale>ysave:
                                ysave = ycur+yMaxArray[i]*2+Padding*worldscale
                else:
                    if (debug == True):
                        print('Path 9')
                    if ysave>ybreak:
                        ybreak = ysave

                    xcur = Padding*worldscale
                    ycur = ybreak
                    xadjust = np.append(xadjust,xcur+Padding*worldscale)
                    yadjust = np.append(yadjust,ycur+yMaxArray[i]+Padding*worldscale)
                    xcur = xcur + xMaxArray[i] + Padding*worldscale
                    if ycur+yMaxArray[i]*2+Padding*worldscale>ysave:
                        ysave = ycur+yMaxArray[i]*2+Padding*worldscale
            #print(i)
            #print('x:' + str(xcur) + ' y: ' + str(ycur))
            if xcur> WellSizeX*DPI:
                print('WARNING, OFF SCREEN!')
            if ycur> WellSizeY*DPI:
                print('WARNING, OFF SCREEN!')
            #print(xadjust)

        return xadjust,yadjust

    def xysort_leftup(swimCount, xMaxArray,yMaxarray,padding,worldscale,WellSizeX,WellSizeY,DPI,debug=False):
        #Setup Adjustments
        xadjust = np.array([0+Padding*worldscale])
        yadjust = np.array([yMaxArray[0]+Padding*worldscale*2])
        j = 0;
        xcur = 0;
        xbreak = xMaxArray[0]+Padding*worldscale
        ybreak = yMaxArray[0]*2+Padding*worldscale*2
        ymax = 0;
        xSave = 0;

        #Find Swimmer packing location
        for i in range(1,swimCount):

            if yadjust[i-1]+yMaxArray[i-1]+Padding*worldscale+2*yMaxArray[i] < WellSizeY*DPI:
                if xcur + xMaxArray[i] +Padding*worldscale < xbreak:
                    xadjust = np.append(xadjust,xcur+Padding*worldscale)
                    xcur = xcur + xMaxArray[i] +Padding*worldscale
                    yadjust = np.append(yadjust,ybreak+yMaxArray[i]+Padding*worldscale)
                    if yMaxArray[i] > ymax:
                        ymax = yMaxArray[i]
                else:
                    ybreak = 2*ymax+Padding*worldscale+ybreak
                    ymax = yMaxArray[i]
                    if xcur > xbreak:
                        xbreak = xcur
                    xcur = xMaxArray[i]+Padding*worldscale+xSave
                    yadjust = np.append(yadjust,ybreak+yMaxArray[i]+Padding*worldscale)
                    xadjust = np.append(xadjust,xSave+Padding*worldscale)
            else:
                xSave = xbreak+Padding*worldscale
                xcur = xMaxArray[i]+Padding*worldscale+xSave
                xbreak = xbreak + xMaxArray[i]+Padding*worldscale
                ybreak = yMaxArray[i]*2+Padding*worldscale
                ymax = 0;
                #for k in range(j,i):
                    #if xMaxArray[k]>xmax:
                        ##print(k)
                        #xmax = xMaxArray[k]
                yadjust = np.append(yadjust,yMaxArray[i]+Padding*worldscale)
                xadjust = np.append(xadjust,xSave+Padding*worldscale)
                j= i
        return xadjust,yadjust

    def xysort_centers(SwimNum,xMaxArray,WellSizeX,WellSizeY,DPI):
        xadjust = np.zeros(SwimNum)
        yadjust = np.zeros(SwimNum)
        j = 0;
        xref = np.linspace(0,WellSizeX*DPI,4+2)
        yref = np.linspace(0,WellSizeY*DPI,6+2)

        for y in range(0,6):
            for x in range(0,4):
                xadjust[j] = xref[x+1]-xMaxArray[j]/2
                yadjust[j] = yref[y+1]
                if (x%2 ==1):
                    yadjust[j] = (yref[y+1]-yref[y])/4 + yadjust[j]
                else:
                    yadjust[j] =  yadjust[j] - (yref[y+1]-yref[y])/4
                j = j+1;
        return xadjust,yadjust

