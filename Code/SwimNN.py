#!/usr/bin/env python
# coding: utf-8
#Machine Learning of Swimmer Velocities
#Disease Biophysics Group
#Written by John Zimmerman
#Updated 4/22/21

import tensorflow as tf
import numpy as np
from itertools import product
import pandas as pd
#from scipy import interpolate
import scipy.interpolate
import scipy.stats
import math
from matplotlib.pyplot import *
get_ipython().run_line_magic('matplotlib', 'inline')
import h5py
import GeoSwimmer
import statsmodels.api as sm

print(tf.__version__)
tf.compat.v2.keras.backend.clear_session()

class SwimVel:
    def gen_approx_swim_vels(DNA,syn1mag=0.5,syn2mag=0.5,syn3mag=0.5,syn4mag=0.5,syn5mag=0.25):
        swimVels = SwimVel.approx_dna_to_vel_base(DNA)+SwimVel.approx_dna_to_vel_syn5(DNA,syn5mag)+SwimVel.approx_dna_to_vel_syn1(DNA,syn1mag)+SwimVel.approx_dna_to_vel_syn2(DNA,syn2mag)+SwimVel.approx_dna_to_vel_syn3(DNA,syn3mag)+SwimVel.approx_dna_to_vel_syn4(DNA,syn4mag)
        return swimVels
    
    def approx_dna_to_vel_base(DNA):
        val = 0
        weights = SwimDNA.getWeights(DNA)
        for a,wgt in zip(DNA,weights):
            val = val + np.sin((float(ord(a)-65)/13.)*4*np.pi+np.pi*5/4)*wgt
        return val
    
    def approx_dna_to_vel_syn1(DNA, mag=0.5):
        #adds value for streaks (JKLMN)
        val = 0
        if mag !=0:
            weights = SwimDNA.getWeights(DNA)
            i=0
            for a in DNA:
                if i>0:
                    if (ord(a)-ord(DNA[i-1])!=0):
                        val = val + ((13-np.abs(ord(a)-ord(DNA[i-1])))/13)*weights[i]
                        #val = val + mag*(1-weights[i])
                i+=1
        return val*mag

    def approx_dna_to_vel_syn2(DNA, mag=0.5):
        #Subtracts value for negative streaks (NMLKJ)
        val = 0
        if mag !=0:
            weights = SwimDNA.getWeights(DNA)
            i=0
            for a in DNA:
                if i>0:
                    if (ord(a)-ord(DNA[i-1])==-1):
                        val = val - mag*weights[i]
                i+=1
        return val
    
    def approx_dna_to_vel_syn3(DNA, mag=0.5):
        #Multiples Weights from first half with second half - rev order
        weights = SwimDNA.getWeights(DNA)
        chr_values = np.zeros(len(DNA))
        if mag !=0:
            i = 0
            for a,wgt in zip(DNA,weights):
                chr_values[i] =  ((13-float(ord(a)-65))/13.)*wgt
                i+=1
        
        return np.sin(chr_values.reshape(2,-1).prod(axis=0)*10).sum()*mag    
    
    def approx_dna_to_vel_syn4(DNA, mag=0.5):
        #Adds value for middle charcter interactions
        weights = SwimDNA.getWeights(DNA)
        characters = np.zeros(len(DNA))
        if mag !=0:
            i=0
            for a, w in zip(DNA, weights):
                characters[i] = float(ord(a) - 65)
                i += 1
            weights[:2] = 0
            weights[-2:] = 0
        return np.dot(characters, weights).sum()*mag

    def approx_dna_to_vel_syn5(DNA, mag=0.25):
        #Random depending on repeat charcters
        counts = {a:0 for a in SwimDNA.DNABasis()}
        val = 1
        if mag !=0:
            for a in DNA:
                counts[a]+=1
            for k,v in counts.items():
                if (v == 1):
                    val +=1
                elif (v > 1):
                    val -= v/5
        return val*mag

class SwimDNA:
    def getWeights(DNA):
        weights = np.ones(len(DNA))
        m = -1/float(len(DNA))
        b = 1
        for i in range(0,len(DNA)):
            weights[i] = m*i+b
        weights = weights/np.sum(weights)
        return weights
    
    def getRevWeights(DNA):
        weights = np.zeros(len(DNA))
        m = 1/float(len(DNA))
        b = 1
        for i in range(0,len(DNA)):
            weights[i] = m*i+b
        weights = weights/np.sum(weights)
        return weights

    def string_to_matrix(string):
        matrix = np.zeros((14,len(string)), dtype=np.int8)
        for position, letter in enumerate(string):
            num = (ord(letter.upper()) - 65) % 14
            matrix[num, position] = 1
        return matrix
    
    def DNABasis():
        return ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
    
    def avgDnaListDistance(dnalist):
        #Feed in dataframe.DNA list
        avglist = np.zeros(dnalist.size) #Declare List for storing Averages

        for DNA1,i in zip(dnalist,np.arange(dnalist.size)): #Cycled through each DNA
            dist = 0
            for DNA2 in dnalist: #Compare to Each Other DNA in database
                if DNA1 != DNA2:
                    dist += SwimDNA.dnaDistance(DNA1,DNA2)
            dist = dist/float(dnalist.size-1) #Average Out Values
            avglist[i] = dist
        return avglist

    def dnaDistance(DNA1,DNA2):
        vector1 = SwimDNA.DNAVector(DNA1)
        vector2 = SwimDNA.DNAVector(DNA2)
        dist = 0
        for a in SwimDNA.DNABasis():
            dist += (vector1[a]-vector2[a])**2

        return np.sqrt(dist)

    def DNAVector(DNA,revweights=False):
        if revweights:
            weights = SwimDNA.getRevWeights(DNA)
        else:
            weights = SwimDNA.getWeights(DNA)
        
        vector = {a:0 for a in SwimDNA.DNABasis()}
        num = np.arange(len(DNA))

        for a,i in zip(DNA,num):
                    vector[a]+=1*weights[i]
        return vector
    
class RadarPlot:
    #Radar Plot
    def constructRadarBasis(dnaAlphabet):
        ang = np.linspace(0,2*np.pi,len(dnaAlphabet)+1)
        alphabetVectors = np.zeros((len(dnaAlphabet),2))

        i = 0
        for theta in ang[:-1]:
            alphabetVectors[i,0],alphabetVectors[i,1]=np.sin(theta),np.cos(theta)
            i+=1
        return alphabetVectors

    def DNARadarPoints(DNA,alphabetVectors):
        weights = SwimDNA.getWeights(DNA)
        mDNA = SwimDNA.string_to_matrix(DNA)
        x = np.sum(weights*mDNA.T.dot(alphabetVectors[:,0]))
        y = np.sum(weights*mDNA.T.dot(alphabetVectors[:,1]))
        return x,y
    
    def DNARadarPointsList(DNA,alphabetVectors):
        weights = SwimDNA.getWeights(DNA)
        mDNA = SwimDNA.string_to_matrix(DNA)
        x = np.sum(weights*mDNA.T.dot(alphabetVectors[:,0]))
        y = np.sum(weights*mDNA.T.dot(alphabetVectors[:,1]))
        return np.array([x,y])
    
    
    def SwimListRadarPoints(SwimList):
        alphabetVectors = RadarPlot.constructRadarBasis(SwimDNA.DNABasis())
        xlist = np.zeros(len(SwimList.DNA))
        ylist = np.zeros(len(SwimList.DNA))
        i=0
        for DNA in SwimList.DNA:
            xlist[i],ylist[i] = RadarPlot.DNARadarPoints(DNA,alphabetVectors)
            i+=1
        return xlist,ylist 
    
    def PlotRadarMesh(xdata,ydata,zdata,res=100,figheight=5,figwidth=5,vmin=0,vmax=1):
        alphabetVectors = RadarPlot.constructRadarBasis(SwimDNA.DNABasis())
        
        x = np.linspace(-1,1,res)
        y = np.linspace(-1,1,res)
        xs,ys = np.meshgrid(x,y)

        gridinterp = scipy.interpolate.griddata((xdata,ydata),zdata,(xs,ys))


        #Plot Velocity Map
        fig, ax = subplots(figsize=(figwidth,figheight))
        cms = matplotlib.cm.jet
        normv = matplotlib.colors.Normalize()
        normv.autoscale(np.array([vmin,vmax]))
        vsm = matplotlib.cm.ScalarMappable(cmap=cms,norm=normv)
        vsm.set_array([])
        #cms = matplotlib.cm.jet


        ax.pcolormesh(xs,ys, gridinterp,cmap=cms,vmin=vmin,vmax=vmax)
        
        
        circle = Circle((0, 0), 1, fill=False,lw=2,ls='--')
        ax.add_patch(circle)
        
        #Radar Labels
        i=0
        for letter in SwimDNA.DNABasis():
            text(alphabetVectors[i,0]*1.1,alphabetVectors[i,1]*1.1,letter,fontweight='bold')
            i+=1
        
        cbar = fig.colorbar(vsm)
        cbar.set_label(('Velocity'), rotation=270,fontsize=30,labelpad=30)
        
        axis('off')
        axis('scaled')
        #fig.tight_layout(rect=[-1.5, 1.5, -1.5, 1.5])
        return fig
    
    def EvenGrid(circres,radres):
        if radres<1:
            radres = 1
        if circres<1:
            circres = 1
        
        ang = np.linspace(0,2*np.pi,len(dnaAlphabet)*int(np.round(circres)))
        line = np.linspace(0.1,1,int(np.round(radres))+1)
        #line = 1-(1-line)**2
        line = line/line.max()
        
        x = np.zeros(ang.size*line.size+1)
        y = np.zeros(ang.size*line.size+1)
        #print(f'xsize: {x.size}')
        #print(f'ysize: {y.size}')
        
        i=0
        for theta in ang:
            for point in line:
                x[i],y[i]=np.sin(theta)*point,np.cos(theta)*point
                #print(point)
                i+=1
            #print(i)
        return x,y    

class NN:
    def OneHotEncodeList (SwimList):
        trainMatrix  =np.zeros((SwimList.DNA.size,14,len(SwimList.DNA.iloc[0])))

        i=0
        for DNA in SwimList.DNA:
            trainMatrix[i,:,:] = SwimDNA.string_to_matrix(DNA)
            i+=1

        trainMatrix= tf.convert_to_tensor(trainMatrix,dtype=tf.float16)    
        return trainMatrix

    def LabelTensor (SwimList):
        labelMatrix  =np.zeros(SwimList.Label.size)

        i=0
        for DNA in SwimList.DNA:
            trainMatrix[i,:,:] = SwimDNA.string_to_matrix(DNA)
            i+=1

        trainMatrix= tf.convert_to_tensor(trainMatrix,dtype=tf.float32)    
        return trainMatrix
    
    def basicModel(InputLength):
        #Inputs
        inputA = tf.keras.Input(shape=(14,InputLength))

        x = tf.keras.layers.Flatten()(inputA)
        x = tf.keras.layers.Dense(6, activation="relu")(x) #Primary Importance of Each Basis Function
        x = tf.keras.layers.Dense(1,activation='sigmoid')(x)

        model = tf.keras.models.Model(inputs=[inputA], outputs=x)
        return model
    
    def Hessian_DualLoss_refine(InputLength):
        #Inputs
        inputA = tf.keras.Input(shape=(14,InputLength))
        
        #Baseline Importance of Each Function
        add_branch = tf.keras.layers.Flatten()(inputA)
        add_branch = tf.keras.layers.LeakyReLU()(add_branch)
        add_branch = tf.keras.layers.Dense(6, activation="relu")(add_branch)
        add_branch = tf.keras.Model(inputs=inputA, outputs=add_branch)
              
        #Branch 2 - Interactions
        hessian = tf.keras.layers.Flatten()(inputA)
        hessian = tf.keras.layers.Dense(14*14, activation="relu",use_bias=False)(hessian)
        y = tf.keras.layers.Reshape((-1, 14, 14))(hessian)
        y = tf.keras.layers.LeakyReLU()(y)
        y = tf.keras.layers.Dropout(0.2)(y)
        y = tf.keras.layers.Conv2DTranspose(2, (5, 5), strides=(2, 2), padding='same', use_bias=True)(y)
        y = tf.keras.layers.Dropout(0.15)(y)
        y = tf.keras.layers.LeakyReLU()(y)
        y = tf.keras.layers.Conv2DTranspose(2, (5, 5), strides=(2, 2), padding='same', use_bias=True)(y)
        y = tf.keras.layers.Dropout(0.15)(y)
        y = tf.keras.layers.LeakyReLU()(y)
        y = tf.keras.layers.Conv2D(2, (5, 5), strides=(2, 2), padding='same')(y)
        y = tf.keras.layers.Dropout(0.15)(y)
        y = tf.keras.layers.LeakyReLU()(y)
        y = tf.keras.layers.Conv2D(6, (4, 4), strides=(2, 2), padding='same')(y)
        y = tf.keras.layers.Dropout(0.15)(y)
        y = tf.keras.layers.LeakyReLU()(y)        
        y = tf.keras.layers.GlobalMaxPooling2D()(y)
        y = tf.keras.layers.Dense(6, activation="relu")(y)
        y = tf.keras.Model(inputs=inputA, outputs=y)

        #combine the output of the two branches - add branch gives main weight, multiply gives interaction
        combined = tf.keras.layers.concatenate([add_branch.output,y.output])

        # Combined Outputs
        z = tf.keras.layers.Flatten()(combined)
        z = tf.keras.layers.Dense(1, activation="sigmoid")(z)
        
        base = tf.keras.layers.Flatten()(add_branch.output)
        base = tf.keras.layers.LeakyReLU()(base) 
        base = tf.keras.layers.Dense(6, activation="relu")(base)
        base = tf.keras.layers.Dense(1, activation="sigmoid")(base)

        #Combined model outputs join of branches
        model = tf.keras.models.Model(inputs=[inputA], outputs=[z,base])
        return model
    
    def genNNModel(InputLength):
        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(14, InputLength)),
          tf.keras.layers.Dense(14, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(14*14, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(14, activation='relu'),
          tf.keras.layers.Dense(1,activation='sigmoid')
        ])
        return model
   
    def genNN_MatMul_model(InputLength):# define two sets of inputs
        inputA = tf.keras.Input(shape=(14,InputLength))

        # Dense Input Branch - Make a 14 dim vector
        x = tf.keras.layers.Flatten()(inputA)
        x = tf.keras.layers.Dense(14, activation="relu")(x)
        x = tf.keras.layers.Reshape((14, 1))(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.Model(inputs=inputA, outputs=x)
        
        #Build cross-correlation Matrix
        y = tf.keras.layers.Flatten()(inputA)
        y = tf.keras.layers.Dense(14*14, activation="relu")(y)
        y = tf.keras.layers.Reshape((14, 14))(y)
        #x = tf.keras.layers.BatchNormalization()(x)
        y = tf.keras.Model(inputs=inputA, outputs=y)

        
        #combine the output of the two branches
        combined = tf.linalg.matmul(y.output, x.output)

        # Combined Outputs
        z = tf.keras.layers.Flatten()(combined)
        z = tf.keras.layers.Dense(14, activation="relu")(z)
        z = tf.keras.layers.Dropout(0.2)(z)
        z = tf.keras.layers.Dense(6, activation="relu")(z)
        z = tf.keras.layers.Dense(1, activation="sigmoid")(z)

        #Combined model outputs join of branches
        model = tf.keras.models.Model(inputs=[inputA], outputs=z)
        return model


    def genNN_mixed_model(InputLength):# define two sets of inputs
        inputA = tf.keras.Input(shape=(14,InputLength))

        # Dense Input Branch
        x = tf.keras.layers.Flatten()(inputA)
        x = tf.keras.layers.Dense(14, activation="relu")(x)
        x = tf.keras.Model(inputs=inputA, outputs=x)
        
        # the second branch opreates on the second input
        y = tf.keras.layers.Reshape((-1, -1, 1))(inputA)
        y = tf.keras.layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same')(y)
        y = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(y)
        y = tf.keras.layers.Dropout(0.2)(y)
        y = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)(y)
        y = tf.keras.layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False)(y)
        y = tf.keras.layers.Dropout(0.2)(y)
        y = tf.keras.layers.Reshape((14, InputLength))(inputA)
        y = tf.keras.layers.Flatten()(y)
        y = tf.keras.layers.Dense(14, activation="relu")(y)
        y = tf.keras.Model(inputs=inputA, outputs=y)
        
        #combine the output of the two branches
        combined = tf.keras.layers.add([x.output, y.output])

        # Combined Outputs
        z = tf.keras.layers.Flatten()(combined)
        z = tf.keras.layers.Dense(14, activation="relu")(z)
        z = tf.keras.layers.Dropout(0.2)(z)
        z = tf.keras.layers.Dense(6, activation="relu")(z)
        z = tf.keras.layers.Dense(1, activation="sigmoid")(z)

        #Combined model outputs join of branches
        model = tf.keras.models.Model(inputs=[inputA], outputs=z)
        return model
    
    def genNNModelLim(InputLength):
        tf.keras.backend.set_floatx('float64')
        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(14, InputLength)),
          tf.keras.layers.Dense(14, activation='relu'),
          tf.keras.layers.Dropout(0.1),
          tf.keras.layers.Dense(14, activation='relu'),
          tf.keras.layers.Dropout(0.1),
          tf.keras.layers.Dense(14, activation='relu'),
          tf.keras.layers.Dense(1,activation='sigmoid')
        ])
        return model
    
    def NNOptimizer(lr=0.001,epsilon=1e-07,amsgrad=False):
        return tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon, amsgrad=amsgrad, name='Adam')
    
    def cum_KL_divergence(ge_c,ge_mean,ge_sigma):
        def KL_div(cdf_pred, cdf_true):
            P = NN.TF_pdf(NN.TF_ppf(cdf_pred,ge_c,ge_mean,ge_sigma),ge_c,ge_mean,ge_sigma) #CDF -> PDF
            Q = NN.TF_pdf(NN.TF_ppf(cdf_true,ge_c,ge_mean,ge_sigma),ge_c,ge_mean,ge_sigma)  #CDF -> PDF
            P = tf.where(tf.math.is_nan(P),tf.convert_to_tensor(0.000001,dtype=tf.float64),P)
            Q = tf.where(tf.math.is_nan(Q),tf.convert_to_tensor(0.000001,dtype=tf.float64),Q)

            return tf.math.square(cdf_pred-cdf_true)#+P*tf.math.log(P/Q)
        return KL_div
   
    def TF_ppf(q,c,mean,sigma):
        x=-tf.math.log(-tf.math.log(q))
        
        return tf.where((c == 0) & (x == x),x,-tf.math.expm1(-c *x )/c)*sigma+mean
    
    def TF_pdf(x,c,mean,sigma):
        z = (x-mean)/sigma
        return tf.where((c==0),tf.math.exp(z)*tf.math.exp(-1*tf.math.exp(z)),tf.math.exp(-(1-c*z)**(1/c))*(1-c*z)**(1/c-1))*(1/sigma)

class SwimSearch:
    def seededList(df):
        alphabet = SwimDNA.DNABasis()
        seedlist=  np.array([])
        for a in alphabet[1:-1]:
            seedlist = np.append(seedlist,(a*6))
            seedlist = np.append(seedlist,a+alphabet[0]*5)
            seedlist = np.append(seedlist,a+alphabet[-1]*5)
        
        seed = pd.DataFrame({'DNA':seedlist})
        seed = df[df.DNA.isin(seed.DNA)]
        
        return seed
    
    def normalizeArray(array,minV,maxV):
        return (array-minV)/float(maxV-minV)

    def rescaleArray(array,minV,maxV):
        return (array)*float(maxV-minV)+minV

    def top_predicted(dataframe,num=2000):
        topLabel = dataframe.nlargest(num,"Label")
        topModel = dataframe.nlargest(num,"ModelLabel")
        correct = topModel[topModel.DNA.isin(topLabel.DNA)==True].DNA.size
        print(f'correct: {correct}')
        return float(correct)/float(len(topLabel.DNA.to_list()))
    
    def SearchMethodHandler(searchmethod):
        #Handles the swimmer selection algorithim for the next generation, defaults to random
        #Terms include 'Method1', 'Random'
        if searchmethod == 'Method1':
            print('Search Method: Method1 - 1/2 Top Predicted, 1/2 Random')
            searchfunc = SwimSearch.Next_Gen_Train_Method1
            
        elif searchmethod == 'Method2':
            print('Search Method: Clustered Vector Distance Directed Evolution')
            searchfunc = SwimSearch.Next_Gen_Train_Method2
        
        elif searchmethod == 'MinAndMax':
            print('Search Method: Top 1/2. Bottom 1/2')
            searchfunc = SwimSearch.Next_Gen_Train_MinAndMax
        
        elif searchmethod == 'MinMaxDE':
            print('Search Method: Top 1/2. Bottom 1/2, wih directed evolution')
            searchfunc = SwimSearch.Next_Gen_Train_MinMaxDE
        
        elif searchmethod == 'Random':
            print('Search Method: Random')
            searchfunc = SwimSearch.Next_Gen_Train_Random
        
        
        elif searchmethod == 'Max':
            print('Search Method: Max')
            searchfunc = SwimSearch.Next_Gen_Train_Max
        
        else:
            print('Search Method: no method picked. Defaulting to random.')
            searchfunc = SwimSearch.Next_Gen_Train_Random

        return searchfunc

    
    def Next_Gen_Train_Method1(df,train,simsperloop,loopnum,seed=1337):
        #Training swimmers are seeded with 1/2 random, and 1/2 top predicted swimmers by the model
        i = 0
        check = True
        while check:
            top = df.nlargest(int(np.round(simsperloop/2.)+i),'ModelLabel')
            includeNum = top[~top.DNA.isin(train.DNA)].DNA.size
            if (includeNum>=int(np.round(simsperloop/2.))):
                check = False
            elif (i>1000):
                check = False
            i+=1
        print(top.DNA.tolist())


        #Append Training Set with top Untrained Values and some random states
        remain = simsperloop-top[~top.DNA.isin(train.DNA)].DNA.size #Take random samples for the remaining simulaitons
        print(f'remaining: {remain}')

        train = train.append(top[~top.DNA.isin(train.DNA)])
        train = train.append(df[~df.DNA.isin(train.DNA)].sample(remain,random_state=seed+simsperloop*loopnum+seed*simsperloop)) 

        return train
    
    def Next_Gen_Train_Method2(df,train,simsperloop,loopnum,seed=1337):
        top = df[~df.DNA.isin(train.DNA)].nlargest(int(np.round(simsperloop)),'ModelLabel')
        top['AvgDistance'] =  SwimDNA.avgDnaListDistance(top.DNA)
        
        mid = top.nlargest(1,'AvgDistance').DNA.to_list()[0]
        far = top.nsmallest(1,'AvgDistance').DNA.to_list()[0]
        
        DNAlength = len(train.DNA.to_list()[0])
        
        GenNum = loopnum
        SwimNum = int(np.round(simsperloop/2))-1
        middnalist = GeoSwimmer.GeoSwimmer.GenerateSwimArray(mid,SwimNum,GenNum,DNAlength,EvolveDNA=True,printDATA=False,insert_del=False,double=False,saveDATA = '')
        fardnalist = GeoSwimmer.GeoSwimmer.GenerateSwimArray(far,SwimNum,GenNum,DNAlength,EvolveDNA=True,printDATA=False,insert_del=False,double=False,saveDATA = '')
        
        #Combine into a DNA list
        dnalist = np.append(middnalist,fardnalist)
        dnalist = np.append(dnalist,far)
        dnalist = np.append(dnalist,mid)
        
        #Lookup values from full dataframe
        top = pd.DataFrame({'DNA':dnalist})
        top = df[df.DNA.isin(top.DNA)]

        #Append Training Set with top Untrained Values and some random states
        remain = simsperloop-top[~top.DNA.isin(train.DNA)].DNA.size #Take random samples for the remaining simulaitons
        print(f'remaining: {remain}')

        train = train.append(top[~top.DNA.isin(train.DNA)])
        #Append Training Set with random states if there is overlap
        train = train.append(df[~df.DNA.isin(train.DNA)].sample(remain,random_state=seed+simsperloop*loopnum+seed*simsperloop)) 

        return train
    
    def Next_Gen_Train_MinMaxDE(df,train,simsperloop,loopnum,seed=1337):
        top = df[~df.DNA.isin(train.DNA)].nlargest(int(np.round(simsperloop)),'ModelLabel')
        top['AvgDistance'] =  SwimDNA.avgDnaListDistance(top.DNA)
        
        bottom = df[~df.DNA.isin(train.DNA)].nsmallest(int(np.round(simsperloop)),'ModelLabel')
        bottom['AvgDistance'] =  SwimDNA.avgDnaListDistance(bottom.DNA)
        
        top_mid = top.nlargest(1,'AvgDistance').DNA.to_list()[0]
        top_far = top.nsmallest(1,'AvgDistance').DNA.to_list()[0]
        
        bot_mid = bottom.nlargest(1,'AvgDistance').DNA.to_list()[0]
        bot_far = bottom.nsmallest(1,'AvgDistance').DNA.to_list()[0]
        
        DNAlength = len(train.DNA.to_list()[0])
        
        GenNum = loopnum
        SwimNum = int(np.round(simsperloop/4))-1
        top_mid_DNA = GeoSwimmer.GeoSwimmer.GenerateSwimArray(top_mid,SwimNum,GenNum,DNAlength,EvolveDNA=True,printDATA=False,insert_del=False,double=False,saveDATA = '')
        top_far_DNA = GeoSwimmer.GeoSwimmer.GenerateSwimArray(top_far,SwimNum,GenNum,DNAlength,EvolveDNA=True,printDATA=False,insert_del=False,double=False,saveDATA = '')
        
        bottom_mid_DNA = GeoSwimmer.GeoSwimmer.GenerateSwimArray(bot_mid,SwimNum,GenNum,DNAlength,EvolveDNA=True,printDATA=False,insert_del=False,double=False,saveDATA = '')
        bottom_far_DNA = GeoSwimmer.GeoSwimmer.GenerateSwimArray(bot_far,SwimNum,GenNum,DNAlength,EvolveDNA=True,printDATA=False,insert_del=False,double=False,saveDATA = '')
        
        
        #Combine into a DNA list
        dnalist = np.append(top_mid_DNA,top_far_DNA)
        dnalist = np.append(dnalist,top_far)
        dnalist = np.append(dnalist,top_mid)
        
        dnalist = np.append(dnalist,bottom_mid_DNA)
        dnalist = np.append(dnalist,bottom_far_DNA)
        dnalist = np.append(dnalist,bot_far)
        dnalist = np.append(dnalist,bot_mid)
        
        #Lookup values from full dataframe
        top = pd.DataFrame({'DNA':dnalist})
        top = df[df.DNA.isin(top.DNA)]

        #Append Training Set with top Untrained Values and some random states
        remain = simsperloop-top[~top.DNA.isin(train.DNA)].DNA.size #Take random samples for the remaining simulaitons
        print(f'remaining: {remain}')

        train = train.append(top[~top.DNA.isin(train.DNA)])
        #Append Training Set with random states if there is overlap/ already been studied
        train = train.append(df[~df.DNA.isin(train.DNA)].sample(remain,random_state=seed+simsperloop*loopnum+seed*simsperloop)) 

        return train
    
    def Next_Gen_Train_MinAndMax(df,train,simsperloop,loopnum,seed=1337):
        top = df[~df.DNA.isin(train.DNA)].nlargest(int(np.round(simsperloop/2.)),'ModelLabel')
        bottom = df[~df.DNA.isin(train.DNA)].nsmallest(int(np.round(simsperloop/2.)),'ModelLabel')

        train = train.append(top[~top.DNA.isin(train.DNA)])
        train = train.append(bottom[~bottom.DNA.isin(train.DNA)])
        
        return train
    
    def Next_Gen_Train_Max(df,train,simsperloop,loopnum,seed=1337):
        top = df[~df.DNA.isin(train.DNA)].nlargest(int(np.round(simsperloop)),'ModelLabel')
        train = train.append(top[~top.DNA.isin(train.DNA)])
        
        return train
    
    def Next_Gen_Train_Random(df,train,simsperloop,loopnum,seed=1337):
        #Training swimmers are seeded with totally random swimmers
        top = df[~df.DNA.isin(train.DNA)].sample(simsperloop,random_state=seed+simsperloop*loopnum+seed*simsperloop)
        train = train.append(top[~top.DNA.isin(train.DNA)])
        print(f'NewSwimmers: {top.DNA.tolist()}')
        
        return train


    
    def Train_Loop(savepath,loadpath, seed=12345, InitialNumbers=1000,testNum=1000,epochsperloop=100,loops=10,simsperloop=100, save=False,method='Random',lr=0.001,epsilon=1e-07):

        df = pd.read_pickle(loadpath+'results2.pkl')
        dnaLength = len(df.DNA[0])
        #print(f'DNAL: {dnaLength}')

        #Initial Training Data/ Test Data Sampling
        train = SwimSearch.seededList(df) #Seed with some selected basis functions (BAAAAA,BNNNNN,BAAAAA)
        remain = InitialNumbers-train.DNA.size
        train = train.append(df[~df.DNA.isin(train.DNA)].sample(remain,random_state=seed)) 

        #Initiate Neural Network Model
        tf.keras.backend.clear_session()
        model = NN.genNN_mixed_model(dnaLength)
        mse = tf.keras.losses.MeanAbsoluteError()
        model.compile(optimizer=NN.NNOptimizer(lr=lr,epsilon=epsilon),
                      loss=mse,
                      metrics=['accuracy'])

        #Save Weights to Reset
        model.save_weights(savepath+'model.h5')
        
        searchmethod = SwimSearch.SearchMethodHandler(method)
        
        print('One Hot Encoding Dataframe...')
        OH_df = NN.OneHotEncodeList(df)
        print('Done.')
        
        if save:
            TopPercentile = np.zeros(loops)

        for loopnum in range(0,loops):
            print(f'Train min Value: {train.Label.min()}')
            print(f'Train Max Value: {train.Label.max()}')

            #Reset Model - Reloading Weights
            model.load_weights(savepath+'model.h5')

            #Prepare Samples for Model
            test = df[~df.DNA.isin(train.DNA)].sample(testNum,random_state=seed+1+loopnum*simsperloop) #Generate Test Set of Swimmers
            rescale_max =  train.Label.max()*1.5
            rescale_min =  train.Label.min()*1.5

            OH_train = NN.OneHotEncodeList(train) #One Hot encoded tensor
            OH_test = NN.OneHotEncodeList(test) #One Hot encoded tensor

            train_label = tf.convert_to_tensor(SwimSearch.normalizeArray(train.Label.to_numpy(),rescale_min,rescale_max),dtype=tf.float32)
            test_label = tf.convert_to_tensor(SwimSearch.normalizeArray(test.Label.to_numpy(),rescale_min,rescale_max),dtype=tf.float32)

            #Fit Model to data
            model.fit(OH_train, train_label, epochs=epochsperloop)
            print(f'Loop {loopnum} MSE: {np.average((test.Label.to_numpy()-SwimSearch.rescaleArray(model(OH_test,training=False).numpy(),rescale_min,rescale_max))**2)}')

            #Update Model Label
            print('Updating Model Label...')
            df['ModelLabel'] = SwimSearch.rescaleArray(model(OH_df,training=False).numpy(),rescale_min,rescale_max)            
            
            train = searchmethod(df,train,simsperloop,loopnum,seed)
            
            if save:
                sample = df.sample(50000,random_state=seed)
                OH_sample = NN.OneHotEncodeList(sample)
                fig = RadarPlot.PlotRadarMesh(sample.RadX,sample.RadY,sample.ModelLabel,figwidth=9,figheight=6.5,vmin=df.Label.min(),vmax=df.Label.max(),res=100)
                savefig(savepath+'Model_'+method+'_seed'+str(seed)+'_LoopNum'+str(loopnum)+'.png')
                clf()
                close()
                
                TopPercentile[loopnum] = SwimSearch.top_predicted(df,2000)
                   
        if save:
            np.savetxt(savepath+'TopPerc2000_'+method+'_seed'+str(seed)+'.txt',TopPercentile)
        #return train
        return model, df, rescale_min, rescale_max

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    def top_predicted_vel(dataframe,num=2000):
        topLabel = dataframe.nlargest(num,"Vel")
        topModel = dataframe.nlargest(num,"ModelLabel")
        correct = topModel[topModel.DNA.isin(topLabel.DNA)==True].DNA.size
        print(f'correct: {correct}')
        return float(correct)/float(len(topLabel.DNA.to_list()))
    
    def Train_Loop_cdf(savepath,loadpath, seed=12345, InitialNumbers=1000,testNum=1000,epochsperloop=100,loops=10,simsperloop=100, save=False,method='Random',lr=0.001,epsilon=1e-07):

        df = pd.read_pickle(loadpath+'results3.pkl')
        df['ModelLabel'] = np.zeros(df.DNA.size)
        dnaLength = len(df.DNA[0])
        #print(f'DNAL: {dnaLength}')

        #Initial Training Data/ Test Data Sampling
        train = SwimSearch.seededList(df) #Seed with some selected basis functions (BAAAAA,BNNNNN,CAAAAA, etc)
        remain = InitialNumbers-train.DNA.size
        train = train.append(df[~df.DNA.isin(train.DNA)].sample(remain,random_state=seed)) 
        #train = df.sample(InitialNumbers,random_state=seed)
        
        

        #Initiate Neural Network Model
        tf.keras.backend.clear_session()
        model = NN.genNN_mixed_model(dnaLength)
        #model = NN.genNN_MatMul_model(dnaLength)

        #Save Weights to Reset
        model.save_weights(savepath+'model.h5')
        
        searchmethod = SwimSearch.SearchMethodHandler(method)
        
        print('One Hot Encoding Dataframe...')
        OH_df = NN.OneHotEncodeList(df)
        print('Done.')
        
        if save:
            TopPercentile = np.zeros(loops)

        for loopnum in range(0,loops):
            print(f'Train min Value: {train.Vel.min()}')
            print(f'Train Max Value: {train.Vel.max()}')

            #Reset Model - Reloading Weights
            #model.load_weights(savepath+'model.h5')

            #Estimate CDF of Training Samples
            train_vels = np.array(train.Vel.to_list()) #Convert of Array
            print('Fitting CDF to array')
            ge_c,ge_mean,ge_sigma = scipy.stats.genextreme.fit(train_vels) #Fit using extreme value thereom
            print(f'ge_c: {ge_c}, ge_mean: {ge_mean}, ge_sigma: {ge_sigma}')
            
            
            #Prepare Samples for Model
            test = df[~df.DNA.isin(train.DNA)].sample(testNum,random_state=seed+1+loopnum*simsperloop) #Generate Test Set of Swimmers
            train['CDF'] = scipy.stats.genextreme.cdf(train.Vel, ge_c,loc=ge_mean,scale=ge_sigma)
            test['CDF'] = scipy.stats.genextreme.cdf(test.Vel, ge_c,loc=ge_mean,scale=ge_sigma)

            OH_train = NN.OneHotEncodeList(train) #One Hot encoded tensor
            OH_test = NN.OneHotEncodeList(test) #One Hot encoded tensor

            train_label = tf.convert_to_tensor(train.CDF.to_numpy(),dtype=tf.float64)
            test_label = tf.convert_to_tensor(test.CDF.to_numpy(),dtype=tf.float64)
            
            #return train,test
            #Update Model
            mse = tf.keras.losses.MeanSquaredError()
            model.compile(optimizer=NN.NNOptimizer(lr=lr,epsilon=epsilon),
                      loss=mse,
                      metrics=['accuracy'])
            
            #Fit Model to data
            model.fit(OH_train, train_label, epochs=epochsperloop)
            print(f'Loop {loopnum} MSE: {np.average((test.CDF.to_numpy()-model(OH_test,training=False).numpy())**2)}')

            #Update Model Label
            print('Updating Model Label...')
            splits = int(np.round(df.DNA.size/4))
            breakList = list(SwimSearch.chunks(np.arange(df.DNA.size),splits))
            modelLabel = np.zeros(df.DNA.size)
            for breaks in breakList:
                modelLabel[breaks[0]:breaks[-1]+1] = model(OH_df[breaks[0]:breaks[-1]+1],training=False).numpy().reshape(-1)
            
            df['ModelLabel']=modelLabel
            
            #Update training data to include next results
            train = searchmethod(df,train,simsperloop,loopnum,seed)

            if save:
                sample = df.sample(50000,random_state=seed)
                OH_sample = NN.OneHotEncodeList(sample)
                fig = RadarPlot.PlotRadarMesh(sample.RadX,sample.RadY,sample.ModelLabel,figwidth=9,figheight=6.5,vmin=0,vmax=1,res=100)
                savefig(savepath+'Model_'+method+'_seed'+str(seed)+'_LoopNum'+str(loopnum)+'.png')
                clf()
                close()

                TopPercentile[loopnum] = SwimSearch.top_predicted_vel(df,2000)
        
        if save:
            np.savetxt(savepath+'TopPerc2000_'+method+'_seed'+str(seed)+'.txt',TopPercentile)

        return model, df, train



