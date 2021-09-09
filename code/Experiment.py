from psychopy import visual, gui,parallel,event,misc,core
from psychopy.misc import pix2deg, deg2pix, cm2deg
from psychopy.event import getKeys
from matustools.CameraThread import Camera
import time, sys,os,datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from matustools.SMI import  *
from matustools.psycho import *
from scipy.stats import scoreatpercentile as sap
#from matustools.SMI import *
N=128
CIRCLE=np.ones((N,N))*-1
for i in range(N):
    for j in range(N):
        if np.sqrt((i-N/2+0.5)**2+(j-N/2+0.5)**2)<N/2:
            CIRCLE[i,j]= 1

class Qpursuit01():
    monitor = asus 
    refreshRate= 120 # hz
    scale=1.53 #multiplier
    itemSize= 1 # in degrees of visial angle
    boxSize=[32,18] # rectangle width and height
    boxLW= 0.1# box line width
    stimOffset=[0,0]# stimulus offset in cm
    bckgCLR= [-0.22,-0.26,-0.24]# [1 -1]
    boxCLR= [1,1,1]
    itemCLR= [1,1,1]# [1 -1]                  
    trialDur=10 # in seconds
    winPos=(0,0)# in pixels
    fullscr=True
    collisionHandling=[0,0,0,0]
    agInitDir= [[1,0],[0,1],[1,1],[3,13]] # vectors with initial movment direction
    agInitSpeed= 14# initial agent speed 20 for 10m, 8 for 4m
    agInitPos= [[0,0]]*len(agInitDir)#initial agent position
    #agInitPos[-2]=[10,3]
    agCount=1 # more agents not implemented
    # settings for adaptive-velocity policy
    fw=9 # width of ET data filter in frames
    maxDistance=10 # maximum target to gaze distance
    pursuitMinVel= 5 # minimum velocity to accept sample as pursuit
    minOnTarget=0.2
    minPursuit=0.7
    minSpeed=5
    maxSpeed=40
    stepSize=[8,4,2,1,1]
    ###############################################
    itemCLR=np.array(itemCLR)
    boxSize=np.array(boxSize)*scale
    boxLW=boxLW*scale
    refreshRate=float(refreshRate)
    itemSize=itemSize*scale
    from os import getcwd as __getcwd
    __path = __getcwd()
    __path = __path.rstrip('code')
    from os.path import sep as __sep
    #path=path.replace('Users','Benutzer')
    inputPath=__path+"input"+__sep
    outputPath=__path+"output"+__sep
    #nrTrials=len(agInitVel)
    stimOffset=[cm2deg(stimOffset[0],monitor),
        cm2deg(stimOffset[1],monitor)]
    #agInitPos=np.array(agInitPos,ndmin=2)
    for i in range(len(agInitDir)):
        agInitDir[i]= np.array(agInitDir[i],ndmin=2)
        agInitPos[i]= np.array(agInitPos[i],ndmin=2)
    
    bs=boxSize/2-boxLW/2.-itemSize/2.
    __verts=[[bs[X],bs[Y]],[bs[X],-bs[Y]],[-bs[X],-bs[Y]],[-bs[X],bs[Y]]]
    wallsXYs=[]
    for i in range(len(__verts)):
        wallsXYs.append([__verts[i-1],__verts[i]])
    
Qpursuit02= type('Qpursuit02',Qpursuit01.__bases__,dict(Qpursuit01.__dict__))
Qpursuit02.collisionHandling=[1,1,0,0,1]
Qpursuit02.stepSize=[-12,18,6,-18,0]
Qpursuit02.agInitSpeed= 18
agInitDir= [[3,13],[1,1],[1,1],[3,13],[3,13]]
for i in range(len(agInitDir)):
    agInitDir[i]= np.array(agInitDir[i],ndmin=2)
Qpursuit02.agInitDir=agInitDir

Qpursuit03= type('Qpursuit03',Qpursuit02.__bases__,dict(Qpursuit02.__dict__))
Qpursuit03.collisionHandling=[0,0,1,1,0]
        
def checkCollision(pos,vel,walls,collisionType):
    vel1=np.copy(vel);pos1=pos+vel
    for w in walls:
        for c in [X,Y]:
            if (np.abs(pos[0,c])>np.abs(w[0][c]) and 
                w[0][c]==w[1][c] and np.sign(pos[0,c])==np.sign(w[0][c])):
                vel1[0,c]*=-1
                if collisionType==1:vel1[0,1-c]*=-1
                pos1[0,c]=2*w[0][c]-pos[0,c]
                return pos1,vel1
    return pos1, vel1 


def nextSpeed(oldSpeed,step,fracOnTarget,fracPursuit,expID,Q):
    if expID==0:
        if fracOnTarget<Q.minOnTarget: return oldSpeed
        inc=(np.sqrt(Q.maxSpeed)-np.sqrt(Q.minSpeed))/20.*step
        if fracPursuit>Q.minPursuit:
            prop=np.square(np.sqrt(oldSpeed)+inc)
            return min(Q.maxSpeed,prop)
        else:
            prop=np.square(np.sqrt(oldSpeed)-inc)
            return max(Q.minSpeed,prop)
    elif expID==1 or expID==2:
        return oldSpeed+step
    
class Experiment():
    def __init__(self,calibVp=None):
        self.vpinfo,fn=infoboxBaby((200,400))
        self.expID=self.vpinfo[1]-1
        self.fn='pursuit%02d'%(self.expID+1)+fn
        self.Q=[Qpursuit01,Qpursuit02,Qpursuit03][self.expID]
        if calibVp is None: calibVp=self.vpinfo[0]
        self.ET=Eyetracker(self.Q.outputPath+self.fn,self.Q.monitor.getDistance(),
            calibVp=calibVp,fcallback=self.getFrameIndex)
        self.win=QinitDisplay(self.Q)
        self.AC=AttentionCatcher(self.win,[0,0],
            self.Q.inputPath+'attentionCatcher.ogg',self.Q.refreshRate)
        self.elem=visual.ElementArrayStim(self.win,fieldShape='circle', 
            nElements=self.Q.agCount, sizes=[self.Q.itemSize],interpolate=True,
            elementMask=CIRCLE,elementTex=None,colors=self.Q.itemCLR)
        lw=deg2pix(self.Q.boxLW,self.Q.monitor)
        self.box=visual.Rect(self.win,width=self.Q.boxSize[X],lineColor=self.Q.boxCLR,
            height=self.Q.boxSize[Y],lineWidth=lw)
        Qsave(self.Q,self.fn)
        self.out=open(self.Q.outputPath+self.fn+'.res','w')
        self.f=-1
        self.jumpToCalibration=False
        self.cam=Camera(self.getFrameIndex,self.Q.outputPath+self.fn)
        self.cam.start()
    def showTrial(self):
        while True:#todo
            sd=self.ET.getSample()
            if not sd is None:
                #if sd[LDIAM]>0: print sd
                if (sd[LDIAM]>0 and sd[LX]**2+sd[LY]**2<100 or 
                    sd[RDIAM]>0 and sd[RX]**2+sd[RY]**2<100): break
            self.AC.draw()
            self.box.draw()
            self.win.flip()
            for key in getKeys():
                if key in ['return','c']:
                    self.skip=True
                    self.AC.stop()
                    if key in ['c']:self.jumpToCalibration=True
                    return
            
        self.AC.stop()
        t0=time.time()
        self.ET.sendMsg('trialon')
        self.f=0
        self.k=0
        self.ts=np.zeros(int(self.Q.refreshRate)*self.Q.trialDur)*np.nan
        self.dat=np.zeros((self.ET.refreshRate*self.Q.trialDur+100,4))*np.nan
        self.cam.resume()
        while time.time()-t0<self.Q.trialDur and not self.skip:
            pos,self.vel=checkCollision(self.elem.xys,self.vel,self.Q.wallsXYs,
                self.Q.collisionHandling[self.block])
            self.elem.setXYs(pos)
            self.elem.draw()
            self.box.draw()
            pre='%f '%core.getTime()
            dat=[self.trial,self.f,pos[0,X],pos[0,Y],self.vel[0,X],self.vel[0,Y]]
            self.out.write(pre+('{} '*(len(dat)-1)+'{}\n').format(*dat))
            self.win.flip()
            D=self.ET.getSample()
            if not D is None:
                D=np.array(D)
                self.dat[self.k,X]=np.nanmean(D[[LX,RX]])
                self.dat[self.k,Y]=np.nanmean(D[[LY,RY]])
                self.dat[self.k,2+X]=pos[0,X]
                self.dat[self.k,2+Y]=pos[0,Y]
                self.k+=1
            for key in getKeys():
                if key in ['return']:
                    self.skip=True
                    self.ET.sendMsg('skiptrial')
                elif key in ['c']:
                    self.skip=True
                    self.jumpToCalibration=True
                    self.ET.sendMsg('skiptrial')
            self.ts[self.f]=core.getTime()
            self.f+=1
        self.ts=np.diff(self.ts[:self.f])*1000
        self.cam.pause()
        self.out.flush()
        self.f=-1
        self.ET.sendMsg('trialoff')
    def getFrameIndex(self):
        return self.f
    def terminate(self):
        self.ET.terminate()
        self.win.close()
        self.out.close()
        self.cam.stop()
        core.quit()
        
    def run(self):
        for i in range(len(self.Q.agInitDir)):
            print 'Block %d'%(i+1)
            self.block=i
            if self.jumpToCalibration: continue  
            if waitForSpaceKey(): continue
            speed=self.Q.agInitSpeed
            mm=0
            self.skip=False
            for j in range(len(self.Q.stepSize)):
                self.trial=i*len(self.Q.stepSize)+j
                self.elem.setXYs(self.Q.agInitPos[i])
                norm=np.sqrt(np.square(self.Q.agInitDir[i]).sum())
                self.vel=self.Q.agInitDir[i]/norm*speed/self.Q.refreshRate
                
                self.showTrial()
                self.box.draw()
                self.win.flip()
                if self.skip: break
                gpx=np.convolve(self.dat[:self.k,X],np.ones(self.Q.fw)/float(self.Q.fw),mode='same')
                gpy=np.convolve(self.dat[:self.k,Y],np.ones(self.Q.fw)/float(self.Q.fw),mode='same')
                dist=np.sqrt(np.square(gpx-self.dat[:self.k,X+2])+np.square(gpy-self.dat[:self.k,Y+2]))
                vel=np.sqrt(np.square(np.diff(gpx))+np.square(np.diff(gpy)))*self.ET.refreshRate
                sel=dist<self.Q.maxDistance
                oldspeed=speed;ot=np.mean(sel);
                op=np.mean(vel[sel[:-1]]>self.Q.pursuitMinVel)
                speed=nextSpeed(speed,self.Q.stepSize[mm],ot,op,self.expID,self.Q)
                if speed!=oldspeed: mm+=1
                tlag=sap(self.ts,99)-sap(self.ts,1)
                msg='Trial %d, Tlag %.2f, Oldspeed=%.1f, Newspeed=%.1f, Step=%d, OnTarget=%.3f, Pursuit=%.3f'%(j+1,tlag,oldspeed,speed,self.Q.stepSize[j],ot,op)
                print msg
                self.ET.sendMsg(msg)
                time.sleep(0.8)
        self.win.flip()
        self.ET.calibrate(self.Q.refreshRate,win=self.win)
        waitForSpaceKey()
        self.terminate()
          
            
if __name__ == '__main__':
    exp=Experiment(calibVp=-9999)
    exp.run()
