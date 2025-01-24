import numpy as np
import os,pickle
from scipy.interpolate import interp1d
from scipy.stats import scoreatpercentile as sap

##########################################################
#
# CONSTANTS and DEFINITIONS
#
##########################################################

DPI=500 #dpi of figure in manuscript
a2d=np.atleast_2d
deg2cm=np.pi/180*70
 
MVPID,MCOH,M_2,M_3,MAGE,M_5,M_6,M_7,MCOND,MDFORM,MXMETA,METHZ,MCAL=range(13)
TS,TE,TSKIP,TOVEL,TSTEP,TTARG,TPURS,TFS,TFE=range(9)
GT,GC,GF,GLX,GLY,GRX,GRY,GBX,GBY,GSTIMX,GSTIMY,GTHETA=range(12)

CA=[[0,1,0],[0,0,1],[0,1,1],[0,3,13],
[1,3,13],[1,1,1],[0,1,1],[0,3,13],[1,3,13],
[0,3,13],[0,1,1],[1,1,1],[1,3,13],[0,3,13],
[2,-3,-13],[2,13,3],[0,13,3],[0,-3,-13],[2,-3,-13],
[0,-3,-13],[0,13,3],[2,13,3],[2,-3,-13],[0,-3,-13]]
CC=[[0,1,0],[0,0,1],[0,1,1],[0,3,13], [1,3,13],[1,1,1],
    [2,-3,-13],[2,13,3],[0,13,3],[0,-3,-13]]
CB=[[0,1,2,3],[4,5,2,3,4],[3,2,5,4,3],[6,7,8,9,6],[9,8,7,6,9]]
a=[item for sublist in CB for item in sublist]
for i in range(len(a)):
    assert(CC[a[i]]==CA[i])
DPATH='data/';FPATH='../publication/figs/'
OPATH=os.getcwd()[:-4]+'outputAnon'+os.path.sep+'pursuit'
CH=[[0,0,0,0],[1,1,0,0,1],[0,0,1,1,0],[2,2,0,0,2],[0,0,2,2,0]]
#0 causal, 1 reverse, 2 90deg, 3 no col
X=0;Y=1
NA=np.newaxis

def printRhat(w):
    from arviz import summary
    print('checking convergence')
    azsm=summary(w)
    nms=azsm.axes[0].to_numpy()
    rhat = azsm.to_numpy()[:,-1]
    nms=nms[np.argsort(rhat)]
    rhat=np.sort(rhat)
    stuff=np.array([nms,rhat])[:,::-1]
    print(stuff[:,:10].T)
    i=(rhat>1.1).nonzero()[0]
    nms=nms.tolist()
    nms.append('__lp')
    nms=np.array(nms)[np.newaxis,:]
    rhat=rhat.tolist()
    rhat.append(-1)
    rhat=np.array(rhat)[np.newaxis,:]
    return i.size>0,nms,rhat

def xy2dphi(x,y,trg=None,percentiles=[50,2.5,97.5]):
    ''' 
        translate from 2D euclidean (hor.+vert.) coordinates
            to radial coordinates (angle+distance)
        x,y - coordinates on horizontal and vertical axis
        trg - the center of the window for angular difference
        percentiles - request percentile estimates of the returned variables
        returns distance, angle
    '''   
    phi=((np.arctan2(y,x)+2*np.pi+(np.pi-trg))%(2*np.pi)-(np.pi-trg))/np.pi*180
    R=np.array([[np.cos(trg),np.sin(trg)],
                    [-np.sin(trg),np.cos(trg)]])
    tmp=np.concatenate([x[:,NA,:],y[:,NA,:]],axis=1)
    d=R.dot(tmp)[0,:,:]
    tmp=np.copy(phi)
    if len(percentiles):
        d=sap(d,percentiles,axis=0)
        phi=sap(phi,percentiles,axis=0)
    return d,phi,tmp

##########################################################
#
# DATA PREPROCESSING
#
##########################################################

# preprocessed information is put in variable D which is saved as 
# multiple files in the data folder, use D=loadData() to load D
# D[0] is a ndarray with Nx13 elements, the relevant columns are
#   0= subject id, 1=cohort (4,7 or 10),4=age in days,8=group, 
# D[2] is a ndarray with Nx11x13 elements, D[2][n,1:9,:] are eye-tracking 
#       data for each of nine stimuli that were shown during eye-tracker 
#       calibration, the columns along array's 3rd dimension are 0=time in s,
#       1-2 location of calibration target 3-6 median gaze left eye x,y 
#       and right eye x,y
# D[3] is a ndarray with Nx25x9 elements, stores trial information
#   columns are: 0= trial start in s, 1= trial end in s, 
#   2= if 1 trial was interrupted by experimenter otherwise 0, 
#   3=circle's nominal velocity deg/s,4-6=velocity-adjustment parameters, 
#   7-8= the index of D[4][n] when trial started and ended 
# D[4][n] is a ndarray with Tx12 elements
#   stores T gaze measurements up-sampled (nearest-neighbour) to 120 hz (monitor rate) 
#   columns are: 0= trial id (-1 = no trial shown),1-time of gaze measurment in sec.,
#   2=trial frame (1 to 1200), 3-6 gaze left eye x,y and right eye x,y, 
#   7-8= gaze both eyes x,y, 9-10= circle's position, 3-10 in deg of visual 
#   angle with origin at screen center, 11=circle's direction in rad
# D[5][n] is a ndarray with Nx6 elements, stores saccade exclusion stats
#   columns are: 0=total count,1=minus sacs outside trials,
#   2=minus sacs from interrupted trials,3=minus sacs too long, 
#   4=minus too far away from circle,4=minus multiple bounces,
#   5=minus lm sacs = bounce sacs
# D[6][n] is a ndarray with Sx10 elements where S is the number of 
#   events extracted by remodnav, columns are: 0 - event typ (0=saccade),
#   1 - time when event started (in sec since start of experiment)
#   2 - time when event ended (in sec since start of experiment)
# D[9][n] is a ndarray with Sx14 elements where S is the number of bounces
#   watched by participant i; 0- time when bounce occured; 1-trial; 
#   2- frame; 3-motion type; 4,5-bounce location;
#   6,7-old stim-movement direction; 8-13-new stim-movement direction assuming 
#   physics-based, inverted or 90-deg bounce
# D[10][n] is ndarray with Sx14 elements, stores catch-up saccades+circle info
#   columns are: 0-1= gaze pos rel to bounce origin, 2= angle of phys bounce, 3= velocity, 4= trial id, 5-6= stim rel to bounce origin, 7-8= gaze pos rel to stim pos, 9-10= location of origin, 11= saccade amplitude, 12-13= sac end relative to when bounce ocurred
def checkCollision(pos,vel,collisionType,verbose=False):
    '''compute agent's new position and velocity 
        pos - 2d array with agent's position in deg of visual angle
        vel - 2d array with agent's velocity in deg/s
        walls - list of walls, each element is ist a 2d array with coordinates of wall's vertices
        collisionType - 0=physical bounce, 1=inverse bounce, 2=rectangle bounce
        returns new position and new velocity
    '''
    walls=[[[-23.6385, 12.9285], [23.6385, 12.9285]], 
           [[23.6385, 12.9285], [23.6385, -12.9285]], 
           [[23.6385, -12.9285], [-23.6385, -12.9285]], 
           [[-23.6385, -12.9285], [-23.6385, 12.9285]]]
    if pos.ndim==1:pos=np.array(pos,ndmin=2)
    if vel.ndim==1:vel=np.array(vel,ndmin=2)
    vel1=np.copy(vel);pos1=pos+vel
    for w in walls:
        for c in [X,Y]:
            if (np.abs(pos[0,c])>np.abs(w[0][c]) and 
                w[0][c]==w[1][c] and np.sign(pos[0,c])==np.sign(w[0][c])):
                #backtrack to wall
                frac=(pos1[0,c]-w[0][c])/vel[0,c]
                pos1[0,:]=pos1[0,:]-vel[0,:]*frac
                if verbose: print(pos1,vel1,w[0][c],frac)
                if collisionType==0:#physical
                    vel1[0,c]*=-1
                    pos1[0,c]=2*w[0][c]-pos[0,c]
                elif collisionType==1:#deg180
                    vel1[0,:]*=-1
                elif collisionType==2:#deg90
                    vel1[0,:]=vel1[0,::-1]
                    if c==Y and w[0][Y]<0:
                        if vel[0,X]>0:vel1[0,X]*= -1#GUS
                        else: vel1[0,Y]*= -1#US
                    elif c==Y and w[0][Y]>0:
                        if vel[0,X]<0:vel1[0,X]*= -1#GUS
                        else: vel1[0,Y]*= -1#US
                    elif c==X and w[0][X]<0:
                        if vel[0,Y]<0:vel1[0,X]*= -1#GUS
                        else: vel1[0,Y]*= -1#US
                    elif c==X and w[0][X]>0:
                        if vel[0,Y]>0:vel1[0,X]*= -1#GUS
                        else: vel1[0,Y]*= -1#US
                if verbose: print(pos1,vel1)
                pos1=pos1+vel1*frac
                if verbose: print(pos1)
                return pos1,vel1
    return pos1, vel1
def saveData(D,ddir='data/',nm='D'):
    for k in list(range(4))+[5]:
        np.save(ddir+nm+'%d'%k,D[k])
    for k in [4]+list(range(6,11)):
        for i in range(len(D[k])):
            if D[k][i] is None: continue
            np.save(ddir+nm+'%d_%03d'%(k,i),D[k][i]) 
def loadData(ddir='data/',nm='D'):
    R=[[],[],[],[],[],[],[],[],[],[],[]]
    for k in list(range(4))+[5]:
        R[k]=np.load(ddir+nm+'%d.npy'%k)
    for k in [4]+list(range(6,11)):
        for i in range(R[0].shape[0]):
            R[k].append(None)
            try: R[k][-1]=np.load(ddir+nm+'%d_%03d.npy'%(k,i))
            except:pass
    return R  
def rlstsq(a,b,plot=[]):
    sel=np.logical_and(~np.isnan(a),~np.isnan(b))
    if sel.sum()<2:return
    a=a[sel];b=b[sel]
    a=np.array([np.ones(a.size),a]).T
    coef,c1,c2,c3=np.linalg.lstsq(a,b,rcond=None)
    if len(plot): 
        import pylab as plt
        plt.plot(plot,coef[0]+coef[1]*np.array(plot),'k',alpha=0.1)
    return coef
    
def getMetadata(showFigure=False):
    '''returns Metadata as ndarray with one row for each infant
        the columns are
    '''
    cpath=os.getcwd()[:-4]+'outputAnon'+os.path.sep
    infoa=np.int32(np.loadtxt(cpath+'vpinfo001.res',delimiter=','))
    infob=np.int32(np.loadtxt(cpath+'vpinfo002.res',delimiter=','))
    vpinfo=np.concatenate([infoa,infob])
    # determine exclusion (but keep the entries in vpinfo)
    sel=np.logical_and(vpinfo[:,MAGE]<30*11.5,vpinfo[:,MAGE]>30*3.5)# this line does nothing
    sel=np.logical_and(sel,vpinfo[:,MCOND]!=-1)
    # determine the output file formats (1-old,0-new)
    n=infoa.shape[0]+2;nn=vpinfo.shape[0]
    temp=np.array([np.arange(nn)<n,sel,np.zeros(nn)*np.nan,np.zeros(nn)],dtype=np.int32).T
    vpinfo=np.concatenate([vpinfo,temp],axis=1)
    if showFigure:
        import pylab as plt
        from matusplotlib import ndarray2latextable
        plt.hist(vpinfo[:,MAGE],bins=np.linspace(30*3.5,30*12,17))
        plt.gca().set_xticks(np.linspace(30*4,30*12,9))
        plt.xlabel('age in days');
        plt.figure(figsize=[12,6])
        lbls=[0,1,2,3,4]
        res=[[],[],[]]
        for i in range(3):
            ax=plt.subplot(1,3,i+1)
            sel=vpinfo[:,MCOH]==[4,7,10][i]
            for j in range(5):
                res[i].append(np.sum(vpinfo[sel,MCOND]==lbls[j]))
                plt.barh(j,res[i][-1],color='k')
            ax.set_yticks(range(5))
            ax.set_yticklabels(res[i])
            plt.title(['4M','7M','10M'][i])
            plt.xlim([0,40])
            res[i].append(np.sum(res[i]))
        res.append(np.sum(res,0))
        ndarray2latextable(np.array(res),decim=0)
        print(np.sum(res))
        plt.show()
    return vpinfo

def getFilename(vpin):
    if not vpin[MXMETA]: return None
    h=vpin[MCOND]+1
    if vpin[MDFORM]: fn='%02d%spursuit%02dvp%d.et'%(h,os.path.sep,h,vpin[MVPID])
    else: fn='%spursuitVp%dc%dM.log'%(os.path.sep,vpin[MVPID],vpin[MCOH])
    return fn

def checkFiles(vpinfo):
    '''checks if all data files for all infants in metadata file are available'''
    surp=False
    for suf in ['01','02','03','']:
        fnsall=os.listdir(OPATH+suf)
        fns=list(filter(lambda x: x[-3:]=='.et',fnsall))
        fns=fns+list(filter(lambda x: x[-4:]=='.log',fnsall))
        fns=np.sort(fns)
        for fn in fns:
            if not len(suf):
                vp=int(fn.rsplit('.')[0].rsplit('c')[0].rsplit('Vp')[1])
                m=int(fn.rsplit('.')[0].rsplit('c')[1][:-1])
                temp=np.logical_and(vpinfo[:,MVPID]==vp, vpinfo[:,MCOH]==m)
                temp=(np.logical_and(vpinfo[:,MDFORM]==0, temp)).nonzero()[0]
            else: 
                vp=int(fn.rsplit('.')[0].rsplit('-')[0].rsplit('vp')[1])
                temp=np.logical_and(vpinfo[:,MVPID]==vp,vpinfo[:,MDFORM]==1).nonzero()[0]
                if len(temp)>1: print(fn+' ID not unique')

            if not temp.size:
                surp=True
                print(fn+' surplus file')
    if not surp: print('No surpluss files (missing from vpinfo) found')            
    ok=True       
    for i in range(vpinfo.shape[0]):
        fn=getFilename(vpinfo[i,:])
        if fn is None: continue
        if not os.path.isfile(OPATH+fn): 
            print('Missing:',OPATH+fn)
            ok=False
        if vpinfo[i,MDFORM] and not os.path.isfile(OPATH+fn[:-3]+'.res'):
            print('Missing:',OPATH+fn[:-3]+'.res')
            ok=False
    if ok: print('No files are missing') 
def extractFromDataFiles():
    '''read eye-tracking data'''
    import warnings,ast,sys
    warnings.filterwarnings('ignore')
    print('Extracting ET data from text files')

    def _getcm(temp):
        temp=temp.replace(', ',',')
        for k in range(10):
            temp=temp.replace('[ ','[')
            temp=temp.replace('  ',' ')
        temp=temp.replace(' ',',')
        temp=ast.literal_eval(temp)
        return np.array(temp)

    vpinfo=getMetadata(showFigure=False)
    D=[vpinfo,[],[],[],[],[],[],[],[],[],[]]
    for i in range(D[0].shape[0]):
        for k in range(1,len(D)):D[k].append(None);
        D[1][-1]=np.ones((4,2))*np.nan
        fn=getFilename(D[0][i,:])
        D[2][-1]=np.ones((11,13))*np.nan
        D[3][-1]=np.ones((25,9))*np.nan
        D[10][-1]=None
        if not D[0][i,MXMETA]: continue
        sys.stdout.write('Finished %.0f %%   \r'%(i/4.1))
        sys.stdout.flush()
        if vpinfo[i,MDFORM]: stim=np.loadtxt(OPATH+fn[:-3]+'.res') 
        else: stim=[]
        f=open(OPATH+fn,'r') 
        lns=f.readlines()
        f.close()
        msg=[];et=[];
        for ln in lns:
            ln=ln.rstrip('\n').rstrip('\r')
            if ln[:2]=='##': 
                ws=ln[3:].rsplit(': ')
                if ws[0]=='Eyetracker Sampling Rate': D[0][i,METHZ]=int(float(ws[1]))
                elif ws[0]=='Calib Matrix': D[1][-1]=_getcm(ws[1])
            elif ln.count('MSG'): msg.append(ln.rsplit(';'))
            else: et.append(np.float64(ln.rsplit(';')).tolist())
        # process messages into trial info and calibration info
        ti=0
        for k in range(len(msg)):
            if msg[k][4][:6]=='Trial ':
                msg[k][4:]=msg[k][4].rsplit(', ')
                if msg[k][5][:4]=='Tlag':msg[k][5:]=msg[k][6:]
            elif msg[k][4][:3]=='Old':stop
                
            if msg[k][4]=='START calibration':D[2][-1][0,0]=msg[k][1]
            elif msg[k][4][:10]=='Target at ':#calibration target
                kk=11-np.isnan(D[2][-1][:,0]).sum()
                D[2][-1][kk,:3]=np.array([msg[k][1]]+msg[k][4][10:].rsplit(' '))
            elif msg[k][4]=='END calibration':
                D[2][-1][10,0]=msg[k][1]
                D[0][i,MCAL]=1
            elif msg[k][4][:7]=='trialon':
                if len(msg[k][4])>8: ti=int(msg[k][4][8:])
                D[3][-1][ti,TSKIP]=0
                D[3][-1][ti,TS]=float(msg[k][1])
            elif msg[k][4]=='trialoff': 
                assert(np.isnan(D[3][-1][ti,TE]))
                D[3][-1][ti,TE]=float(msg[k][1])
                ti+=1
            elif msg[k][4][:5]=='Trial':
                temp=np.array(list(map(lambda x: float(x.rsplit('=')[1]),list(msg[k][5:]))))
                if (ti-1)%5!=int(msg[k][4][6:])-1:
                    assert(int(msg[k][4][6:])==1)
                    tinew=(int( (ti-1)/5)+1)*5+1
                    D[3][-1][tinew-1,:]=D[3][-1][ti-1,:]
                    D[3][-1][ti-1,:]=np.nan
                    ti=tinew
                D[3][-1][ti-1,TOVEL]=temp[0]
                D[3][-1][ti-1,TSTEP:TFS]=temp[2:]
            elif msg[k][4]=='skiptrial' or msg[k][4][:4]=='Jump':D[3][-1][ti,TSKIP]=1
            elif msg[k][4][:4]=='flip':
                stim.append([msg[k][0],ti,msg[k][2]]+msg[k][4].rsplit(' ')[1:])
            else: print(msg[k][4]);stop
        D[2][-1]=np.array(D[2][-1],dtype=np.float64)
        assert(D[2][-1].shape[0]==11)
        stim=np.array(stim,dtype=np.float64)
        assert(stim.shape[1]==7)
        #determine stim movement changes
        D[9][i]=[]
        for k in range(25):
            tmp=stim[stim[:,1]==k,:]
            if not np.isnan(D[3][i][k,TS]):
                D[9][i].append([-999,k,-999,-1]+[-1]*10)
            assert(np.all(np.diff(tmp[:,2])==1))
            for f in np.diff(tmp[:,[5,6]],axis=0).sum(axis=1).nonzero()[0]:
                mt=CC[CB[D[0][i,MCOND]][int(k/5)]][0]
                D[9][i].append([-1,k,f,mt]+tmp[f,3:].tolist())
                if D[9][i][-2][6]==-1: 
                            D[9][i][-2][6]=D[9][i][-1][6]
                            D[9][i][-2][7]=D[9][i][-1][7]
                for h in range(3):
                    pos,vel=checkCollision(tmp[f,3:5],tmp[f,5:],h)
                    D[9][i][-1]+=vel[0].tolist() 
                assert(len(D[9][i][-1])==14)
        D[9][i]=np.array(D[9][i])
        D[9][i][:,6:]*=120#dg per sec
        #merge ET data and stim coords
        trl=-1;D[4][-1]=[]
        trid=np.unique(stim[:,1]).tolist()+[-1]
        cd=[[],[],[],[],[],[],[],[],[]]
        for k in range(len(et)):
            if D[2][-1][0,0]< et[k][GC] and D[2][-1][10,0]> et[k][GC]:
                for m in range(9):
                    if D[2][-1][1+m,0]+500000< et[k][GC] and D[2][-1][1+m,0]+4900000> et[k][GC]:
                        cd[m].append(et[k][GLX:7]+et[k][9:15])
                        continue
                continue
            if len(D[4][-1]) and et[k][GF]>-1 and D[4][-1][-1][GF]==-1:trl+=1
            if et[k][GF]!=-1:
                s=np.logical_and(trid[trl]==stim[:,1], et[k][GF]==stim[:,GF]).nonzero()[0]
                if not len(s) and et[k][GF]>=1199:continue
                assert(len(s)==1)
                temp=[stim[s[0],3],stim[s[0],4],np.arctan2(stim[s[0],6],stim[s[0],5])]
            else: temp=[np.nan,np.nan,np.nan]
            D[4][-1].append([trid[trl]]+et[k][1:7]+[np.nan,np.nan]+temp+et[k][9:15])
            if len(D[4][-1][-1])==16:D[4][-1][-1]=D[4][-1][-1][:GTHETA+1] 
            assert(len(D[4][-1][-1])==12 or len(D[4][-1][-1])==18)
            if len(D[4][-1][-1])==GTHETA+1:continue
            D[4][-1][-1]=np.array(D[4][-1][-1],dtype=np.float64)
            if D[4][-1][-1][GTHETA+3]==0:D[4][-1][-1][GTHETA+1:GTHETA+4]=np.nan
            if D[4][-1][-1][GTHETA+6]==0:D[4][-1][-1][GTHETA+4:GTHETA+7]=np.nan
        assert(trl==len(trid)-2)
        D[4][-1]=np.array(D[4][-1])
        # change time to sec.
        if fn=='/pursuitVp92c7M.log': #one file needs ad-hoc correction
            D[4][-1][D[4][-1][:,GC]<1000000000,GC]+=3700000000
            D[3][-1][D[3][-1][:,TS]<1000000000,TS]+=3700000000
            D[3][-1][D[3][-1][:,TE]<1000000000,TE]+=3700000000
        
        D[3][i][:,[TS,TE]]=(D[3][i][:,[TS,TE]]-D[4][i][0,GC])/1000000
        D[4][i][:,GC]=(D[4][i][:,GC]-D[4][i][0,GC])/1000000
        for k in range(D[3][i].shape[0]):
            if np.isnan(D[3][i][k,TS]):continue
            D[3][i][k,TFS]=np.min((D[4][i][:,GC]>D[3][i][k,TS]).nonzero()[0])
            D[3][i][k,TFE]=np.max((D[4][i][:,GC]<D[3][i][k,TE]).nonzero()[0])
        if D[0][i,MCAL]:
            for m in range(9):
                assert(len(cd[m])>1)
                temp=np.nanmedian(np.array(cd[m],dtype=np.float32),axis=0)
                D[2][-1][m+1,3:3+temp.size]=temp
        #compute time for stim movement changes
        tr=np.int32(D[9][i][:,1])
        s=np.int32(D[3][i][tr,TFS])
        e=np.int32(D[3][i][tr,TFE])
        for h in range(D[9][i].shape[0]):
            if D[9][i][h,0]==-999 and D[9][i][h,2]==-999:
                if np.isnan(D[3][i][int(D[9][i][h,1]),TS]):stop
                D[9][i][h,0]=D[3][i][int(D[9][i][h,1]),TS]
                D[9][i][h,2]=D[3][i][int(D[9][i][h,1]),TFS]
                continue
            G=D[4][i][s[h]:e[h],:]
            tf=interp1d(G[:,GF],G[:,GC],bounds_error=False)
            D[9][i][h,0]=tf(D[9][i][h,2])
    D[1]=np.array(D[1])
    D[2]=np.array(D[2])
    D[3]=np.array(D[3])
    D[5]=np.zeros(0)#reserved for rotateSaccades
    saveData(D)
    
def accuracyCorrection(method='no pooling',applyCor=True,exclude=False):
    '''correct gaze data from experiment based on gaze data from calibration trials'''
    def _partialPooling(dat,compileStan=True,runStan=False,ssmm=1):
        mdl='''
            data {
                int<lower=0> N;
                vector[2] y[N,9];
                vector[2] c[9];
            }parameters{
                vector[4] o[N];
                vector<lower=0,upper=100>[2] sy;
                vector[4] mo;
            vector<lower=0,upper=100>[4] so;
            corr_matrix[4] ro;
            } model {
            sy~cauchy(0,20);
            so~cauchy(0,20);
            for (n in 1:N){ o[n]~multi_normal(mo,quad_form_diag(ro,so));
            for (p in 1:9){
                if (! is_nan(y[n,p][1]))
                    c[p]~multi_normal(head(o[n],2)+tail(o[n],2).*y[n,p],diag_matrix(sy));       
            }}} '''
        import stan,pickle
        from matusplotlib import loadStanFit,saveStanFit
        dat['N']=dat['y'].shape[0]
        if compileStan:
            smsv=stan.StanModel(model_code=mdl)
            with open(DPATH+f'smlCalib.pkl', 'wb') as f: pickle.dump(smsv, f)
        
        if runStan:
            with open(DPATH+f'smlCalib.pkl','rb') as f: smsv=pickle.load(f)
            fit=smsv.sampling(data=dat,chains=6,n_jobs=6,
                thin=int(max(2*ssmm,1)),iter=int(ssmm*2000),warmup=int(ssmm*1000))
            saveStanFit(fit,'fitcalib')
            print(fit)
        w=loadStanFit('fitcalib')
        return np.median(w['o'],0),np.median(w['mo'],0)
    
    print('Performing linear accuracy correction')
    import ast
    D=loadData()
    # load omitted calibration locations
    mp={5:0,1:1,3:2,9:3,7:4,6:5,4:6,2:7,8:8}
    #mp maps from coding order (left to right, top to bot) to presentation order
    with open(OPATH[:-7]+'calib.sel','r') as f: dr=ast.literal_eval(f.read())
    res=list(dr.values());ids=list(dr.keys())
    for h in range(len(ids)):
        ids[h]=np.int32(ids[h].rsplit('_'))
    ids=np.array(ids,dtype=np.int32)
    for i in range(D[0].shape[0]):
        if not D[0][i,MCAL]:continue
        sel=np.logical_and(D[0][i,MVPID]==ids[:,0],D[0][i,MCOH]==ids[:,1]).nonzero()[0]
        assert(len(sel)==1)
        d=res[sel[0]]
        if len(d)==0: D[2][i,:,:]=np.nan
        elif len(d)==2:
            for h in range(2): 
                for dd in d[h]:
                    D[2][i,mp[dd]+1,(3+2*h):(5+2*h)]=np.nan
        else:stop
    # compute linear correction based on data from calibration routine
    if method=='no pooling':
        for e in range(2):
            g=[]
            for i in range(len(D[2])):
                d=D[2][i]
                ctrue=d[1:10,1:3]
                c=d[1:10,3+e*2:5+e*2]
                coef=[np.nan,np.nan,np.nan,np.nan,np.nan]
                assert(np.all(np.isnan(c[:,0])==np.isnan(c[:,1])))
                sel=~np.isnan(c[:,0])
                if np.isnan(c[:,0]).sum()>6:# don't apply correction when 2 or less calibration locations available
                    D[2][i][[0,-1][e],1:6]=coef
                    continue
                temp=0
                for k in range(2):
                    x=np.column_stack([np.ones(sel.sum()),c[sel,k]])
                    res =np.linalg.lstsq(x,ctrue[sel,k],rcond=None)
                    coef[k*2:(k+1)*2]=res[0]
                    temp+=res[1][0]
                coef[4]=temp**0.5/sel.sum()
                assert(np.all(np.isnan(coef))==np.any(np.isnan(coef)))
                D[2][i][[0,-1][e],1:6]=coef
                g.append(coef)
            gm=np.nanmean(g,0)
            for i in range(len(D[2])):
                if np.any(np.isnan(D[2][i][[0,-1][e],1:6])) and not exclude:
                    D[2][i][[0,-1][e],1:6]=gm
    elif method=='partial pooling':
        dat={'c':D[2][4][1:10,1:3],'y':[],'id':[]} 
        for e in range(2):
            dat['y']=[];dat['id']=[]
            for i in range(len(D[2])):
                c=D[2][i][1:10,3+e*2:5+e*2]
                assert(np.all(np.isnan(c[:,0])==np.isnan(c[:,1])))
                sel=~np.isnan(c[:,0])
                if np.isnan(c[:,0]).sum()<=5:# don't apply correction when 3 or less calibration locations available  
                    dat['y'].append(c)
                    dat['id'].append(i)
            dat['y']=np.array(dat['y'])
            dat['c']=np.array(dat['c'])
            np.save('c',dat['c'])    
            o,mo=_partialPooling(dat,compileStan=True,ssmm=1)
            for i in range(len(D[2])):
                if not exclude: D[2][i][[0,-1][e],1:5]=mo
            for k in range(len(dat['id'])):
                D[2][dat['id'][k]][[0,-1][e],1:5]=o[k,:]
    # apply linear correction
    for i in range(len(D[2])):
        if D[4][i] is None:continue
        for xy in range(2):
            for e in range(2):
                cor=D[2][i][[0,-1][e],1+xy*2:3+xy*2]
                if method=='external': cor=[[[13.1752,1.288],[-6.083,1.3044]],[[8.0115,1.2726],[-6.2817,1.3044]]][e][xy]
                if not exclude: assert(not np.any(np.isnan(cor)))
                temp=(D[4][i][:,GLX+xy+2*e]-D[1][i,2*e+xy,0])/D[1][i,2*e+xy,1]
                D[4][i][:,GLX+xy+2*e]=cor[1]*temp+cor[0]
    # compute drift correction
    tms=[]
    for i in range(1,3): 
        with open(OPATH[:-7]+f'dc{i}.sel','r') as f: tms.extend(ast.literal_eval(f.read()))
    for j in range(len(tms)):
        assert(len(tms[j])==10)
        i=np.logical_and(tms[j][0]==D[0][:,MVPID],tms[j][1]==D[0][:,MCOH]).nonzero()[0][0]
        G=D[4][i]
        if D[0][i,MCOND]==0:
            for e in range(2):
                if tms[j][2+2*e]==-1 or tms[j][6+2*e]==-1 or tms[j][3+2*e]==-1 or tms[j][7+2*e]==-1:
                    continue
                doneax=-1
                for ax in range(2):
                    ss=np.min((G[:,GC]>tms[j][2+2*e+4*ax]).nonzero()[0])
                    se=np.max((G[:,GC]<tms[j][3+2*e+4*ax]).nonzero()[0])
                    d1=np.nansum(np.abs(np.diff(G[ss:se,GSTIMY])))>0
                    d2=np.nansum(np.abs(np.diff(G[ss:se,GSTIMX])))>0
                    assert(d1 == (not d2))
                    if doneax==-1:doneax=int(d2)
                    else: assert(doneax!=int(d2))
                    corf=np.nanmedian(G[ss:se,GSTIMX+int(d2)]-G[ss:se,GLX+2*e+int(d2)])
                    assert(~np.isnan(corf))
                    D[2][i,[0,-1][e],11+int(d2)]=corf
        elif D[0][i,MCOND]>0:
            D[2][i,0,11:13]=0;D[2][i,-1,11:13]=0
            vels=[]
            for g in range(4):
                if tms[j][2+2*g]==-1:
                    assert(tms[j][3+2*g]==-1)
                    continue
                assert(tms[j][3+2*g]!=-1)
                if not (np.all(~np.logical_and(D[9][i][:,0]>=tms[j][2+2*g],
                                              D[9][i][:,0]<=tms[j][3+2*g]))):
                    gg=np.logical_and(D[9][i][:,0]>=tms[j][2+2*g], D[9][i][:,0]<=tms[j][3+2*g]).nonzero()[0][0]
                    print(tms[j][0],tms[j][2+2*g], D[9][i][gg,0],tms[j][3+2*g])
                    stop
                
                ss=np.min((G[:,GC]>tms[j][2+2*g]).nonzero()[0])
                se=np.max((G[:,GC]<tms[j][3+2*g]).nonzero()[0])
                assert(ss<se)
                ii=np.max((D[9][i][:,0]<tms[j][2+2*g]).nonzero()[0])
                er=G[ss:se,:][:,[GSTIMX,GSTIMY]]-G[ss:se,[GLX+2*(g%2),GLY+2*(g%2)]]
                er=er[~np.isnan(er[:,0]),:]
                tmp=D[9][i][ii,:]
                if tmp[3]==-1:vel=tmp[6:8]
                else: vel=tmp[int(tmp[3])*2+8:int(tmp[3])*2+10]
                
                if g>1 and len(vels)>g-2:
                    if (vels[g-2].dot(vel)!=0):# check if orthogonal
                        print(i,'LR'[g%2],vels[g-2].dot(vel))
                        stop
                else:vels.append(vel)
                phi=np.arctan2(vel[1],vel[0])
                R=np.array([[np.cos(phi),-np.sin(phi)],
                    [np.sin(phi),np.cos(phi)]])
                R2=np.array([[np.cos(phi),np.sin(phi)],
                    [-np.sin(phi),np.cos(phi)]])
                corf=R2.dot([np.median(R.dot(er.T).T),0])
                
                D[2][i,[0,-1][g%2],11:13]+=corf
        else: stop
    res=np.nanmean(np.linalg.norm(D[2][:,[0,-1],11:13],axis=2),axis=0)
    res=np.sqrt(np.square(res).sum()/2)   
    print('accuracy in degrees: ', res)
    if applyCor: saveData(D)
    
def changeUnits():
    '''translate gaze coordinates based on the eye-to-screen distance during each trial'''
    D=loadData()
    # change units
    MD=70# default monitor distance used throughout data recording
    # location of screen center in mm relative to ET at [0,0,0]
    def _chunits(H,dz,sc=[0,0,0]):
        return (H/180*np.pi*MD*10-a2d(sc[:2]))/(dz-sc[2])*180/np.pi
    for i in range(len(D[4])): 
        G=D[4][i]
        if G is None:continue
        for k in [GLX,GRX,GSTIMX]:
            if G.shape[1]==18:
                if k==GSTIMX:dz=np.nanmedian(np.nanmean(G[:,[14,17]]))
                else:dz=np.nanmedian(G[:,{GLX:14,GRX:17}[k]])
            else: dz=620
            G[:,[k,k+1]]=_chunits(G[:,[k,k+1]],dz)
        for k in range(5): 
            if k>0: tmp=_chunits(D[9][i][:,[4+2*k,5+2*k]],dz,sc=[0,0,25])
            else: tmp=_chunits(D[9][i][:,[4+2*k,5+2*k]],dz)
            D[9][i][:,[4+2*k,5+2*k]]=tmp
        D[3][i,:,TOVEL]=_chunits(D[3][i,:,TOVEL,NA],dz,sc=[0,0,25])[:,0]
    # compute binocular gaze by averaging over eyes
    def nanmean(a,b):
        res=np.copy(a)
        res[np.isnan(a)]=b[np.isnan(a)]
        sel=np.logical_and(~np.isnan(a),~np.isnan(b))
        res[sel]=a[sel]/2+b[sel]/2
    for i in range(len(D[2])):
        if D[4][i] is None:continue
        D[4][i]=np.array(D[4][i],dtype=np.float64)
        for xy in range(2):
            D[4][i][:,GBX+xy]=np.nanmean([D[4][i][:,GLX+xy],D[4][i][:,GRX+xy]],axis=0)
    saveData(D)
def extractSaccades():
    '''use REMODNAV to extract saccades
    '''
    from remodnav import EyegazeClassifier
    from sys import stdout
    D=loadData()
    def tseries2eventlist(tser):
        ''' translates from the time series to the event list
            representation '''
        tser=np.int32(tser)
        if tser.sum()==0: return []
        d=np.bitwise_and(tser,np.bitwise_not(np.roll(tser,1)))
        on = (d[1:].nonzero()[0]+1).tolist()
        d=np.bitwise_and(np.roll(tser,1),np.bitwise_not(tser))
        off=d[1:].nonzero()[0].tolist()
        if len(off)==0:off.append(tser.shape[0]-1)
        if len(on)==0: on.insert(0,0)
        if on[-1]>off[-1]: off.append(tser.shape[0]-1)
        if on[0]>off[0]: on.insert(0,0)
        if len(on)!=len(off): print('invalid fixonoff');raise TypeError
        out=np.array([on,off]).T
        return out.tolist()
    # extract events with remodnav
    D[6]=[];
    print('Extracting saccades with remodnav')
    for i in range(len(D[4])):   
        if D[4][i] is None:
            D[6].append(None)
            continue
        stdout.write('Finished %.0f %%   \r'%(i/4.1))
        stdout.flush()
        f=1/D[0][i,METHZ]#sample length in sec.
        s=np.arange(np.ceil(D[4][i][-1,GC]/f))*f
        MBD=0.1#minimum blink duration in seconds
        data={'x':None,'y':None}
        assert(np.all(np.isnan(D[4][i][:,GBX])==np.isnan(D[4][i][:,GBX+1])))
        
        tmp=np.copy(D[4][i][:,GBX])
        # interpolate to regular intervals using nearest-neighbour method
        # interpolate nans
        sel=np.ones(tmp.size,dtype=np.bool_)
        eg=tseries2eventlist(np.isnan(tmp))
        for ev in eg:
            if ev[1]-ev[0]<MBD*D[0][i,METHZ]:
                sel[ev[0]:(ev[1]+1)]=False
        for ax in range(2): data[['x','y'][ax]]=interp1d(D[4][i][sel,GC],tmp[sel],kind='nearest',bounds_error=False,fill_value=np.nan)(s)
        # run remodnav
        ec=EyegazeClassifier(1,D[0][i,METHZ],pursuit_velthresh=2,noise_factor=2.0,
            velthresh_startvelocity=60.0, min_intersaccade_duration=0.04,
            min_saccade_duration=0.02,max_initial_saccade_freq=2.0, saccade_context_window_length=1,max_pso_duration=0.04,min_fixation_duration=0.08,
            min_pursuit_duration=0.08,lowpass_cutoff_freq=4.0)
        p=ec.preproc(data,min_blink_duration=MBD,dilate_nan=0, median_filter_length=0.125,savgol_length=0.125,savgol_polyord=2,max_vel=1000.0)
        events=ec(p,classify_isp=True, sort_events=True)
        # put into new output format
        evs=[]
        blinks=tseries2eventlist(np.isnan(D[4][i][sel,GBX]))
        for bl in blinks:
            evs.append([-1,D[4][i][bl[0],GC],D[4][i][bl[1],GC],bl[0],bl[1]])
        for k in range(D[3][i].shape[0]):
            if k-1<0: es=[D[4][i][0,GC],0]
            else: es=[D[3][i,k-1,TE],D[3][i,k-1,TFE]]
            if not np.isnan([es[0]]):evs.append([-2,es[0],D[3][i,k,TS],es[1],D[3][i,k,TFS] ])
        for k in range(len(events)):
            if (events[k]['label'] in ['ILPS','IHPS'] or 
                events[k]['end_time']>D[4][i][-1,GC]):continue
            lbl2num={'PURS':2,'SACC':0,'ISAC':0,'FIX':1,'FIXA':1}
            evs.append([lbl2num[events[k]['label']],events[k]['start_time'],
                events[k]['end_time']])
            evs[-1].extend([np.max((evs[-1][1]>=D[4][i][:,GC]).nonzero()[0]),
                np.min((evs[-1][2]<=D[4][i][:,GC]).nonzero()[0])])
        evs2=[]
        lst=[0,0]
        while len(evs):
            im=np.argmin(np.array(evs)[:,1])
            ev=evs.pop(im)
            if ev[0]==-2: 
                lst=ev[4],ev[2]
                evs2.append(ev+[0,np.nan,np.nan,np.nan,np.nan])
            elif ev[2]>lst[1]: evs2.append(ev+[0,np.nan,np.nan,np.nan,np.nan])
        D[6].append(np.array(evs2))
    saveData(D)

def rotateSaccades(SACS=0.15,SACE=0,SACMAXL=0.2,CSDIST=7):
    '''select linear-motion and bounce saccades and rotate their targets'''
    D=loadData()
    out=[]
    D[5]=np.zeros((len(D[4]),8))
    for i in range(len(D[4])):
        out.append([]);
        if D[4][i] is None: continue
        S=D[6][i]
        for j in range(S.shape[0]):
            if not S[j,0]==0:continue # iterate over saccades
            D[6][i][j,5]=0#keep track of sac type for later
            tmp=D[4][i][np.int32(S[j,3:5]),GF]
            D[5][i,0]+=1
            if tmp[0]==-1 or tmp[1]==-1: continue
            D[5][i,1]+=1
            if tmp[0]==-1:trl=int(D[4][i][np.int32(S[j,4]),GT])
            else: trl=int(D[4][i][np.int32(S[j,3]),GT])
            if D[3][i][trl,TSKIP]: continue
            D[5][i,2]+=1
            a=D[4][i][np.int32(S[j,4]),[GBX,GBY]];
            if np.isnan(a[0]):continue
            D[5][i,3]+=1
            if S[j,2]-S[j,1]>SACMAXL:continue
            D[5][i,4]+=1
            dst=np.linalg.norm(a-D[4][i][np.int32(S[j,4]),[GSTIMX,GSTIMY]])
            if dst>CSDIST:continue
            D[5][i,5]+=1
            if type(SACS) is list and len(SACS)==2: sel=np.logical_and(S[j,1]-D[9][i][:,0]>SACS[0],S[j,1]-D[9][i][:,0]<SACS[1])
            else: sel=np.logical_and(S[j,1]-D[9][i][:,0]<SACS,S[j,2]-D[9][i][:,0]>SACE)
            sel=np.logical_and(sel,D[9][i][:,3]>-1)
            # compute physical bounce angle
            if sel.sum()>1:continue
            D[5][i,6]+=1
            # compute sac target relative to circle's location
            rr=D[4][i][np.int32(S[j,4])-1:np.int32(S[j,4])+1,[GSTIMX,GSTIMY]]
            th2=np.arctan2(rr[1,1],rr[1,0])
            rrout= (rr[1,:]/np.linalg.norm(rr[1,:]))
            assert(np.isclose(rrout[0],np.cos(th2)))
            th=D[4][i][np.int32(S[j,4]),GTHETA]
            R=np.array([[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]])#clockwise
            gazeRstim=np.squeeze(R.dot(a-rr[1,:])).tolist()
            rrout=R.dot(rrout).tolist()
            assert(len(rrout)==2)
            assert(np.all(~np.isnan(rrout)))
            assert(not np.isnan(D[3][i][trl,TOVEL]))
            sacamp=np.linalg.norm(np.diff(D[4][i][np.int32(S[j,3:5]),GBX:GBY+1],axis=0)) 
            if np.isnan(sacamp):
                sacamp=np.linalg.norm(np.diff(D[4][i][[np.int32(S[j,3])+1,np.int32(S[j,4])],GBX:GBY+1],axis=0))     
            if sel.sum()==0: 
                # put origin at the positive vertical half
                out[-1].append([np.nan,np.nan,np.nan,D[3][i][trl,TOVEL],trl,np.nan,np.nan]+gazeRstim+rrout+[sacamp,np.nan])
                D[6][i][j,5:]=np.array([1,-th,rr[1,0],rr[1,1],0])           
                continue
            D[5][i,7]+=1
            rp=np.squeeze(D[9][i][sel,:])
            th=np.arctan2(rp[7],rp[6]);thsave=np.copy(th)
            R=np.array([[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]])
            tmp=np.squeeze(R.dot((a-rp[4:6])))
            tmp2=np.squeeze(R.dot((D[4][i][np.int32(S[j,4]),[GSTIMX,GSTIMY]]-rp[4:6])))
            tmp3=np.squeeze(R.dot((rp[8:10])))
            if tmp3[1]<0: # mirror to upper half
                tmp[1]= -tmp[1]
                tmp2[1]= -tmp2[1]
                tmp3[1]= -tmp3[1]
                gazeRstim[1]= -gazeRstim[1]
                rrout[1]=-rrout[1]
            # check that the physical bounce dir is as expected
            th=np.arctan2(tmp3[1],tmp3[0])
            phi=np.abs(CC[CB[D[0][i,MCOND]][int(trl/5)]])
            phi=np.arctan2(phi[2],phi[1])
            assert(np.isclose(th,np.pi-2*phi) or np.isclose(th,2*phi))
            assert(not np.isnan(th))
            D[6][i][j,5:]=np.array([2+int(tmp3[1]<0),th,rp[4],rp[5],rp[0]])
            out[-1].append(np.array(tmp.tolist()+[th%np.pi,D[3][i][trl,TOVEL],trl]+tmp2.tolist()+gazeRstim+rrout+[sacamp, S[j,2]-rp[0]]))
        if len(out[-1]):out[-1]=np.array(out[-1])
    D[10]=out
    saveData(D)


##########################################################
#
# DATA ANALYSIS
#
##########################################################

def fitRegression(bounceSac=1,
    pCenter=0,pVelM=1,pVelS=0,yType=1,minSacNr=1,ssmm=1,suf=''):
    ''' fits hierarchical regression model with STAN
        bounceSac - if 1 bounce saccades as outcome, otherwise linear-motion saccades
        pCenter - if 1 includes direction of screen center as predictor
        pVelM - if 1 includes velocity as predictor of mean y
        pVelS - if 0 excludes velocity as predictor of variance of y 
            if 1 uses additive regression model
            if 2 uses multiplicative regression model
        yType - if 1 returns sac tar with bounce loc as origin; mean sac. targets, anticipated angle
            if 0 the stim loc is the origin; motion-parallel displacement
            if 2 takes amplitude as outcome; amplitude
        minSacNr - exclude infants with less than minSacNr of saccades
        ssmm - scale of STAN-simulations, 0.1 => 50 samples, 10 => 5000 samples
        suf - suffix added to the name of all ouput files
    '''
    D=loadData()
    def _prep2dat(prep,yType,minSacNr):
        y=[];xi=[];k=-1;sel=[];xc=[];xa=[];phis=[];xv=[];bts=[];stims=[];xo=[];vp=[]
        rs=[0,0]
        for i in range(len(prep)): 
            if len(prep[i])==0: continue
            if yType==2: ytar=prep[i][:,11,NA]
            elif yType==1:
                assert(bounceSac==1)
                ytar=prep[i][:,:2]
            elif yType==0: ytar=prep[i][:,[7,8]]
            if bounceSac==1: sel2= ~np.isnan(prep[i][:,2])
            else: sel2= np.isnan(prep[i][:,2])  
            if yType==2: sel2=np.logical_and(sel2,~np.isnan(ytar[:,0]))
            if sel2.sum()<minSacNr:continue
            k+=1
            vp.append(D[0][i,[MVPID,MCOH]])
            xa.append(D[0][i,MAGE]/30) 
            y.append(ytar[sel2,:])
            xi.append(k*np.ones(sel2.sum()));
            th=np.round(prep[i][sel2,2],4)
            trl=np.int32(prep[i][sel2,4]/5)
            bt=np.array(CC)[np.array(CB[D[0][i,MCOND]])[trl]][:,0]
            xc.append(th+bt*10)
            bts.append(bt)
            xv.append(prep[i][sel2,3])
            phis.append(np.round(prep[i][sel2,2],4))
            stims.append(prep[i][sel2,5:7])
            xo.append(prep[i][sel2,9:11]) 
        y=np.concatenate(y,0);
        stims=np.concatenate(stims,0)
        phis=np.concatenate(phis,0);bts=np.concatenate(bts,0)
        xi=np.int32(np.concatenate(xi))+1;xv=np.concatenate(xv,0);
        xc=np.concatenate(xc,0)+100;xo=np.concatenate(xo,0);
        xa=np.array(xa)
        k=np.unique(xc).tolist()
        if bounceSac==0:
            xc=np.int32(np.ones(xi.shape))
            dat={'N':len(xa),'xa':xa,'M':xi.size,'K':xo.shape[1],'KK':y.shape[1],
            'xi':xi,'y':y,'xv':xv,'xo':xo,'xc':xc,'L':np.max(xc),'vp':np.array(vp)}
            meta=[-1,-1,-1,-1,-1,xi.size, 
                xv.mean(),xv.std(),xa.size,xa.mean(),xa.std()]
            return dat,meta
        meta=[]#0-bounce code,1-angle of refl + angle of incid [rad],2- bounce type,
        # 3,4-circle's average position [deg,deg] 
        assert(phis.size==xc.size)
        for i in range(len(k)):
            sel=xc==k[i]
            xc[sel]=i
            j=sel.nonzero()[0]
            vpps=np.unique(xi[sel])
            assert(len(np.unique(phis[j]))==1)
            meta.append([k[i]-100,phis[j[0]],bts[j[0]]]+stims[sel,:].mean(0).tolist()+
                [sel.sum(),xv[sel].mean(),xv[sel].std(),vpps.size,
                xa[vpps-1].mean(),xa[vpps-1].std()])
        meta.append([-1,-1,-1,-1,-1,xi.size,xv.mean(),xv.std(),xa.size,xa.mean(),xa.std()])
        xc=np.int32(xc)+1
        if yType==2: xc[:]=1 #pool across conditions
        assert(xi.shape[0]==y.shape[0])
        dat={'N':xa.size,'xa':xa,'M':xi.size,'xi':xi,'K':xo.shape[1],'KK':y.shape[1],
            'y':y,'xc':xc,'L':np.max(xc),'xv':xv,'xo':xo,'vp':np.array(vp)}
        return dat,meta
   
    mdl='''
        data{{
            int N;int M;int L;int K;int KK;
            vector[N] xa;
            int<lower=1> xi[M];
            int<lower=1,upper=L> xc[M];
            real y[M,KK]; real xo[M,K];real xv[M];  
        }} parameters{{
            real<lower=-10,upper=10> b0[L,K];
            real<lower=-2,upper=2> b1[L,K];
            real<lower=-20,upper=20> g[L,K,N];
            real<lower=-10,upper=10> bo; 
            real<lower=0,upper=20> sn[K];
            real<lower=0,upper=30> sm[K,N];
            real<lower=0,upper=300> gsm[K,2];
            real<lower=-10,upper=10> mv[L,K];
            real<lower=-10,upper=10> bv[K];
            real<lower=0,upper=20> vs[K];
            //real<lower=-10,upper=10> v[L,K,N];
            //real<lower=0,upper=10> sv[K];
        }}model{{
            for (k in 1:KK){{
                for (n in 1:N){{
                    sm[k,n]~gamma(gsm[k,1],gsm[k,2]);
                    for (l in 1:L){{
                        g[l,k,n]~normal(b0[l,k]+b1[l,k]*xa[n],sn[k]);
                        //v[l,k,n]~normal(mv[l,k],sv[k]);
                }}}}   
                for (m in 1:M) y[m,k]~normal(g[xc[m],k,xi[m]]{pred},sm[k,xi[m]]{std});
            }}
        }}
        ''' 
    import stan,pickle
    cd='lb'[bounceSac]+f'{pCenter}{pVelM}{pVelS}{yType}{suf}'
    dat,meta=_prep2dat(D[10],yType,minSacNr)
    
    np.save(DPATH+'dat'+f'{cd}',dat,allow_pickle=True)
    np.save(DPATH+'meta'+f'{cd}',meta)
    dat.pop('vp')# don't need this anymore
    ml=mdl.format(pred=['','+bo*xo[m,k]'][pCenter]+['','+mv[xc[m],k]*xv[m]','+(mv[xc[m],k]+bv[k]*xa[xi[m]])*xv[m]','+v[xc[m],k,xi[m]]*xv[m]'][pVelM],
        std=['','+vs[k]*xv[m]','*pow(xv[m],vs[xc[m],k])', '*xv[m]','+vs[k]*(11-xa[xi[m]])'][pVelS])
    print(ml)
    sm=stan.build(program_code=ml,data=dat,random_seed=SEED)
    with open(DPATH+f'sm{cd}.pkl', 'wb') as f: pickle.dump(sm, f)
    
    fit=sm.sample(num_chains=6,num_thin=int(max(ssmm*1,1)),
        num_samples=int(ssmm*500),num_warmup=int(ssmm*500)) 
    converged,nms,rhat=printRhat(fit)
    w={'nms':nms,'rhat':rhat}
    for k in fit.keys():w[k]=np.rollaxis(fit[k],-1,0)
    with open(DPATH+f'sm{cd}.wfit','wb') as f: pickle.dump(w,f,protocol=-1)
    
def computeFreqAmp():
    '''compute saccade frequency'''
    import pylab as plt
    D=loadData()
    plt.figure(figsize=(16,9))
    bs=np.linspace(-0.3,0.7,51)
    wdur=(bs[-1]-bs[0])/(bs.size-1)
    G=np.zeros((len(D[6]),len(bs)-1,7,7))*np.nan
    for i in range(len(D[6])):
        if D[6][i] is None: continue
        #determine bounce type
        a=np.arctan2(-D[9][i][:,7],-D[9][i][:,6])
        b=np.arctan2(D[9][i][:,9],D[9][i][:,8])
        tmp=np.abs((a-b+np.pi)%(2*np.pi)-np.pi)
        for k in range(5):tmp[tmp<[0.3,1.2,1.8,2.9,10][k]]=[10,11,12,13,10][k]
        tmp-=10
        tmp=np.int32(tmp)
        tmp[np.logical_and(D[9][i][:,3]==1,tmp==1)]=4
        tmp[np.logical_and(D[9][i][:,3]==1,tmp==2)]=5
        tmp[D[9][i][:,3]==2]=6
        sel=D[6][i][:,0]==0
        ampl=np.linalg.norm(D[4][i][np.int32(D[6][i][sel,4]),GBX:GBY+1]
            -D[4][i][np.int32(D[6][i][sel,3]),GBX:GBY+1],axis=1)
        dur=D[6][i][sel,2]-D[6][i][sel,1]
        for t in range(7):
            for  j in range(3):
                assert(np.all(D[6][i][sel,1]<D[6][i][sel,2]))
                m=[D[6][i][:,1],D[6][i][:,2],D[6][i][:,1:3].mean(axis=1)][j]
                temp=(m[sel,NA]-D[9][i][NA,tmp==t,0]).flatten()
                a,b=np.histogram(temp,bins=bs,density=False)
                G[i,:,t,j]=a/wdur 
            for k in range(len(bs)-1):
                of=wdur/2+bs[k]
                a=np.logical_and(D[9][i][NA,tmp==t,0]+of>D[6][i][sel,NA,1],
                    D[9][i][NA,tmp==t,0]+of<D[6][i][sel,NA,2]) 
                G[i,k,t,3]=np.nansum(a)
                G[i,k,t,4]=np.nansum(a*ampl[:,NA])
                G[i,k,t,6]=a.shape[1]
            G[i,:,t,5]=(tmp==t).sum()
    np.save(DPATH+'G',G) 
    return
    for h in range(5):    
        plt.subplot(3,2,h+1)
        plt.title(['sac start','sac end','sac mid','occurence','amplitude'][h])
        clrs=['k','b','m','c','g','lime','r']
        for t in range(7):
            if h==4: dn=G[:,t,6]
            else: dn=G[:,t,5]
            plt.plot(bs[:-1]+wdur/2,G[:,t,h]/dn,color=clrs[t])
        plt.grid(True)
        plt.xlabel('Time in sec, bounce at 0 sec')
        plt.ylabel('Saccades per sec')
    plt.savefig(FPATH+'suppSacTime.png',dpi=DPI,bbox_inches='tight',pad_inches=0)

def hypothesisTest(suf,ML=False,BALANCED=False):
    from scipy.stats import norm
    with open(f'data/sm{suf}.wfit','rb') as f: w=pickle.load(f)
    dat=np.load(f'data/dat{suf}.npy',allow_pickle=True)
    dat=dict(dat.tolist())
    na=np.newaxis
    vels=np.linspace(0,40,81)
    ages=np.linspace(4,10,61)
    x=np.linspace(0,20,101)
    p=np.zeros((ages.size,vels.size,5))
    angls=[[180]*7,[0]*7,[90]*7,[180,26,90,154,90,154,154],[180,26,90,154,180,180,90]]
    for h in range(len(angls)):
        for k in range(len(angls[0])):
            angl=angls[h][k]/180*np.pi
            xx=np.array([np.cos(angl)*x,np.sin(angl)*x]) 
            if not BALANCED:
                sel=dat['xc']==k+1
                mu=dat['y'][sel,:]
                if ML:
                    if int(suf[3])==3: sig=np.median(w['sm'],0).T[dat['xi'][sel]-1,:]*dat['xv'][sel,na]
                    tmp=np.max(norm.logpdf(xx[:,:,na],mu.T[:,na,:],sig.T[:,na,:]).sum(0),0)
                    p[0,0,h]+=tmp.sum()
                else:
                    if int(suf[3])==3: sig=w['sm'][:,:,dat['xi'][sel]-1]*dat['xv'][na,na,sel]
                    elif int(suf[3])==4:
                        sig=w['sm'][:,:,dat['xi'][sel]-1]+w['vs'][:,:,na]*(11-dat['xa'][na,na,dat['xi'][sel]-1])
                    tmp=norm.pdf(xx.T[:,na,:,na],mu.T[na,na,:,:],sig[na,:,:,:])
                    p[0,0,h]+=np.log(tmp.mean((0,1))).sum()
                continue
            for i in range(ages.size):
                for j in range(vels.size):
                    age=ages[i]
                    vel=vels[j]
                    if ML:
                        mu=np.median(w['b0'],0)[k,:]+np.median(w['b1'],0)[k,:]*age +np.median(w['mv'],0)[k,:]*vel
                        if int(suf[3])==3: sig= np.sqrt(np.square(vel*np.median(w['gsm'][:,:,0]/w['gsm'][:,:,1],0))+np.square(np.median(w['sn'],0)))
                        #use median xx?
                        p[i,j,h]+=np.product(norm.pdf(xx,mu[:,na],sig[:,na]).mean(1))/7
                    else:
                        S=w['b0'].shape[0]
                        for s in range(S):
                            mu=w['b0'][s,k,:]+w['b1'][s,k,:]*age +w['mv'][s,k,:]*vel
                            if int(suf[3])==3:sig= np.sqrt(np.square(vel*w['gsm'][s,:,0]/w['gsm'][s,:,1])+np.square(w['sn'][s,:]))
                            p[i,j,h]+=np.product(norm.pdf(xx,mu[:,na],sig[:,na]).mean(1))/7
                        #p[i,j,h]/=S does not affect results and was omited
    if not BALANCED: 
        p=p[0,0,:]
        p=p-np.max(p)
        p=np.exp(p)/np.exp(p).sum()
    np.save(DPATH+f'ht{int(ML)}{int(BALANCED)}',p)

##########################################################
#
# FIGURES IN MANUSCRIPT & SUPPLEMENT
#
##########################################################

def plotTerms(ax=None):
    '''plot Figure 1'''
    from matusplotlib import subplotAnnotate,subplot
    import pylab as plt
    from matplotlib.patches import Wedge
    if ax is None: 
        plt.figure(figsize=(8,5))
        ax=plt.gca()
    ss=0.015
    phi=180-np.arctan(0.4-ss)/np.pi*180
    ax.add_patch(Wedge([0,ss],0.4,180-phi,90,ec='r',ls='--',fc='none'))
    ax.add_patch(Wedge([0,ss],0.5,90,phi,ec='r',fc='none'))
    ax.add_patch(Wedge([0,ss],0.7,10,phi,ec='g',fc='none'))
    ax.plot([-1,0],[0.4,ss],'c')
    ax.plot([1,0],[0.4,ss],'c--')
    ax.plot([-1,1],[0,0],color='gray',lw=5)
    ax.legend(['angle of reflection','angle of incidence','anticipated angle',
        'old trajectory','new trajectory','barrier'],ncol=2,loc=[.17,0.75],fontsize=11)
    ax.plot([0,0],[ss,2],color='k')
    ax.plot(0.69,.155,'xg')
    ax.set_xlim([-.8,.8])
    ax.set_ylim([0,1])
    ax.set_aspect(1)
    ax.set_axis_off() 
    plt.savefig(FPATH+'terms.png',dpi=DPI,bbox_inches='tight',pad_inches=0)
    
def plotTrajectory():
    '''plot Figure 2'''
    wx=37.3/2;wy=21/2
    def getBounceLocs(bounceType=2,phi=257,vel=7,initPos=[0,0]):
        pos=np.array(initPos);bt=bounceType;maxdur=10
        out=[pos];phis=[phi]
        dur=0
        k=0
        while True:
            if phi>90 and phi<270: inv=-1
            else: inv=1
            bpx=np.array([inv*wx,np.tan(phi*np.pi/180)*(inv*wx-out[-1][0])+out[-1][1]])
            if phi>180: inv=-1
            else: inv=1
            bpy=np.array([(inv*wy-out[-1][1])/np.tan(phi*np.pi/180)+out[-1][0],inv*wy])
            hithorw= np.linalg.norm(bpx-out[-1])>np.linalg.norm(bpy-out[-1])
            out.append([bpx,bpy][int(hithorw)])
            dur+=np.linalg.norm(out[-1]-pos)/vel
            if dur>=maxdur:
                dur-=np.linalg.norm(out[-1]-pos)/vel
                out.pop()
                break
            pos=out[-1]
            if bt==1: phi=(phi+180)%360
            elif bt==0: 
                if hithorw:phi=(phi+2*(180-phi)+360)%360
                else: phi=(phi+(180-2*phi)+360)%360
            elif bt==2:
                if ((phi<90 or phi>180 and phi<270) and hithorw or 
                    (phi>90 and phi<180 or phi>270) and not hithorw):
                    phi=(phi+270)%360
                else:phi=(phi+90)%360
            phis.append(phi)
            k+=1
        out.append(out[-1]+(maxdur-dur)*vel*np.array([np.cos(phi*np.pi/180),np.sin(phi*np.pi/180)]))   
        return np.array(out),np.array(phis) 
    D=np.array([[0,90,45,77,-1],[1077,1045,45,77,1077],[77,45,1045,1077,77],
            [2257,2013,13,257,2257],[257,13,2013,2257,257]])
    import pylab as plt
    plt.figure(figsize=[8,5])
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if D[i,j]==-1:continue
            plt.subplot(5,5,(4-i)*5+j+1)
            out,phis=getBounceLocs(bounceType=D[i,j]//1000,phi=D[i,j]%1000,vel=14)
            plt.plot(out[:,0],out[:,1],'k')
            plt.xlim([-wx,wx]);plt.ylim([-wy,wy])
            ax=plt.gca()
            ax.set_xticks([]);ax.set_yticks([]);
            ax.set_aspect(1)
            for p in range(1,out.shape[0]-1):
                if D[i,j]==0 or D[i,j]==90:clr='k'
                elif D[i,j]==1045:clr='lime'
                elif D[i,j]==1077:clr='g'
                elif D[i,j]>2000:clr='r'
                elif D[i,j]<1000:
                    if (phis[p]-phis[p-1]+360)%90==0:clr='m'
                    elif (phis[p]-phis[p-1]+360)%360==26:clr='c'
                    else:clr='b' 
                c=plt.Circle(out[p,:],2,ec=clr,fc=clr,zorder=2)
                ax.add_patch(c)
            plt.arrow(out[0,0],out[0,1],2*np.cos(phis[0]/180*np.pi),
                      2*np.sin(phis[0]/180*np.pi),width=1,color='k')
            if i==4:plt.title('Block '+str(j+1))
            if j==0:plt.ylabel('Group '+str(i+1))
    plt.tight_layout()
    plt.savefig(FPATH+'trajectory.png',dpi=DPI,bbox_inches='tight',pad_inches=0)
        
def plot_AA_MPD(suf,sufmpd=None, axs=[],mspeed=20,mage=7):
    ''' plot Figure 3A-3D
        suf - suffix of target file with STAN estimates
        mage - mean age in months
        mspeed - mean object speed in deg/s
    '''
    def _computeMST(suf,mage=7,mspeed=20):
        ''' compute mean saccade targets and their percentage intervals'''
        cis=[[],[],[],[],[],[],[]]
        mage=np.array(mage,ndmin=1)
        mspeed=np.array(mspeed,ndmin=1)
        meta=np.load(DPATH+f'meta{suf}.npy')
        meta[meta[:,1]==0,1]=np.pi
        with open(DPATH+f'sm{suf}.wfit','rb') as f: w=pickle.load(f)
        z=[]
        b0=w['b0'][:,:,:,NA];b1=w['b1'][:,:,:,NA];
        if int(suf[2])>0: 
            b2=w['mv'][:,:,:,NA]
            if suf[2]==2: b2=b2+w['bv'][:,:,:,NA]*mage[NA,NA,NA,:]
        else: b2=0
        tmp=b0+mage[NA,NA,NA,:]*b1+b2*mspeed[NA,NA,NA,:]
        clrs=[];
        ta=[0.45,1.57,2.69,3.14,3.14,3.14,1.57]
        for i,k in enumerate(list(A2CLR.keys())):#draw physical bounce trajs
            ox,oy=[(0,0),(0,0),(0,0),(0,0),(0,.3),(0,-.3),(.3,0)][i]  
        for k in range(tmp.shape[1]):#draw mean sac target
            y=tmp[:,k,:,:]
            a=10*meta[k,2]+np.round(meta[k,1],2)
            cis[list(A2CLR.keys()).index(a)].append(y)
        return cis
    if sufmpd is None:sufmpd=suf
    from pylab import Polygon
    sv=len(axs)==0
    A2CLR={0.45:'c',1.57:'m',2.69:'blue',3.14:'k',
           11.57:'lime',12.69:'g',22.69:'r'}    
    if sv:
        import pylab as plt
        fig, axs =plt.subplots(figsize=(12,5),nrows=3, ncols=2)
        axs=np.reshape(axs,-1)
    for k in range(3):
        for s in range(2):
            vel=[np.linspace(5,40,101),mspeed][s]
            ax=axs[k*2+s]
            age=[mage,np.linspace(4,10,101)][s]
            cis=_computeMST('b'+[sufmpd,sufmpd,suf][k]+str([0,0,1][k]),None,mage=age,mspeed=vel)
            b1=0;xtcks=[]
            for i in range(len(cis)):
                d=np.array(cis[i])[0,:]
                if i<4:trg=list(A2CLR.keys())[i]
                else: trg=[np.pi,np.pi,np.pi/2][i-4]
                clr=list(A2CLR.values())[i]
                if k<2: 
                    r=sap(-d[:,k,:],[50,2.5,97.5],axis=0)
                elif k==2:
                    dst,r,rd=xy2dphi(d[:,0,:],d[:,1,:],trg)
                    r=180-r
                    ax.plot([i-.5,i+.5],[180-trg/np.pi*180,180-trg/np.pi*180],'--',c=clr) 
                     
                gap=.2
                x=[(vel-5)/35,(age-4)/7][s]*(1-gap)+i-.5+gap/2
                xtcks.extend([x[0],x[x.size//2],x[-1]])
                ax.plot(x,r[0,:],c=clr)
                if k==2: ax.plot(x[0]-.1,180-(list(A2CLR.keys())[i]%10)/np.pi*180,'>',c=clr)
                
                if k<2 and s==0:ax.plot([x[0],x[-1]],[vel[0]*0.15,vel[-1]*.15],'y-')
                elif k<2 and s==1: 
                    ax.plot([x[0],x[-1]],[vel*0.15,vel*.15],'y-')
                u=r[2,:];l=r[1,:]
                if s==0 and k==2:
                    for ii,tm in enumerate([180-trg/np.pi*180,180-(list(A2CLR.keys())[i]%10)/np.pi*180]):
                        hr=np.logical_or(tm<u,tm>l).nonzero()[0]
                        if hr.size>0:print(['true','phys'][ii],clr,vel[np.min(hr)],vel[np.max(hr)])
                if s==0 and k<2:
                    tm =vel*0.15
                    hr=(tm>u).nonzero()[0]
                    if hr.size>0:print('mpd',clr,vel[np.min(hr)],vel[np.max(hr)])
                xx=np.concatenate([x,x[::-1]])
                ci=np.concatenate([u,l[::-1]])
                ax.add_patch(Polygon(np.array([xx,ci]).T,
                            alpha=0.2,fill=True,fc=clr,ec=clr));
            if k==2: 
                ax.set_ylim([-60,180])
                ax.set_yticks([-45,0,45,90,135,180])
                if s==1:ax.set_ylabel('Anticipated Angle in °');
            else:
                tmp=np.max(np.abs(ax.set_ylim()))
                ax.set_ylim([-5,3])
                if s==1:
                    ax.set_ylabel('Motion-parallel\nDisplacement in deg');
                    ax.set_xlabel('Infant\'s age')
                else: ax.set_xlabel('Circle\'s speed') 
                
            ax.set_xticks([])
            ax.grid(True,axis='y')   
    if sv: plt.savefig(FPATH+suf+f'_{mspeed}_{mage}.png',dpi=DPI,bbox_inches='tight',pad_inches=0)

def plotSacFrequency(axs=[],alpha=0.05):
    '''plot Figure 3E'''
    sv=len(axs)==0
    if sv:
        import pylab as plt 
        fig, ax = plt.figure()
    else: ax=axs[0]
    G=np.load(DPATH+'G.npy')
    bs=np.linspace(-0.3,0.7,51);wdur=(bs[-1]-bs[0])/(bs.size-1)
    clrs=['k','b','m','c','g','lime','r']
    for t in range(7):
        ax.plot(bs[:-1]+wdur/2,np.nansum(G[:,:,t,2],0)/np.nansum(G[:,:,t,6],0),color=clrs[t],alpha=0.5)
    ax.grid(True)
    ax.set_xlabel('Time in sec with bounce at 0 sec')
    ax.set_ylabel('# saccades per sec')
    from scipy import stats
    for t in  range(7):
        m=wdur*np.nansum(G[:,:,t,2],0)/np.nansum(G[:,:,t,6],0)
        r=stats.norm.ppf(1-alpha/2)*np.sqrt(m*(1-m)/np.nansum(G[:,:,t,6]))
        h=wdur
        print(f'sacFreq CI {clrs[t]} {np.max(r)/h:.3f}');
    if sv: plt.savefig(FPATH+'sacFreq.png',dpi=DPI,bbox_inches='tight')  
    
def plotLegend(axs=[],speed=20):
    '''plot legend in Figure 3 (bottom left)'''
    import pylab as plt
    if len(axs)==0:plt.figure(figsize=(12,5))
    D=[[0,-9,193,'c',8,9,167,0,'78-physical'],
        [0,-9,225,'m',3,-5,135,0,'45-physical'],
      [0,-9,257,'b',-6,-3,103,0,'12-physical'],
      [0,-9,270,'k',-6,-3,90,0,'0-physical'],
       [0,-9,-45,'lime',3,-5,135,0,'45-inverse'],
      [0,-9,-77,'g',-6,-3,103,0,'12-inverse'],
      [0,-9,193,'r',-6,-3,103,0,'12-perpendicular']]  
    for i in range(len(D)):
        a=np.array(D[i][:2])
        b=-speed*np.array([np.cos(D[i][2]/180*np.pi),np.sin(D[i][2]/180*np.pi)])
        c=D[i][4:6]
        G=[a,a+b,c]
        if len(D[i])>6:
            d=speed*np.array([np.cos(D[i][6]/180*np.pi),np.sin(D[i][6]/180*np.pi)])+a
            e=speed*np.array([np.cos(D[i][7]/180*np.pi),np.sin(D[i][7]/180*np.pi)])+a
            G.extend([d,e])
        if len(axs)==0:
            plt.subplot(3,3,1+i)
            ax=plt.gca()
        else: ax=axs[i]
        ax.plot([G[0][0],G[1][0]],[G[0][1],G[1][1]],color=D[i][3])
        ax.set_xlim([-10,10]);ax.set_ylim([-10,0]);
        
        ax.set_aspect(1);
        ax.plot([G[0][0],G[3][0]],[G[0][1],G[3][1]],'-',color=D[i][3])
        ax.plot([G[4][0],2*G[0][0]-G[4][0]],[G[4][1],2*G[0][1]-G[4][1]],color='gray',zorder=-1)
        ax.set_title(D[i][8])
        ax.set_axis_off()
    if len(axs)==0:plt.savefig(FPATH+'lgnd.png',dpi=DPI,bbox_inches='tight',pad_inches=0) 
   
def plotMain(suf,sufmpd=None):
    '''plot Figure 3'''
    if sufmpd is None:sufmpd=suf
    from matusplotlib import formatAxes, subplotAnnotate
    import pylab as plt
    fig2, da = plt.subplots(figsize=(9,9),nrows=2, ncols=1)
    r1=3*'A'+3*'B'+'\n';r2=3*'C'+3*'D'+'\n';r3=3*'E'+'\n'
    fig=plt.figure(figsize=(9,9),layout="constrained")
    axs=fig.subplot_mosaic(3*r1+3*r2+'..g'+r3+'def'+r3+'abc'+r3)
    formatAxes(list(axs.values()))
    plotSacFrequency(axs=[axs['E']]) 
    plot_AA_MPD(f'{suf}',sufmpd=sufmpd,axs=[axs['C'],axs['D'],
        da[1],da[0],axs['A'],axs['B']]) 
    for i in range(5): subplotAnnotate(loc='se',nr=i,ax=axs['ABCDE'[i]])
    fig.tight_layout()
    plotLegend(list(map(lambda x: axs[x],'abcdefg')))
    plt.savefig(FPATH+f'main{suf}.png',dpi=DPI,bbox_inches='tight',pad_inches=0)

def plotExplanation(speed=20):
    '''plot Figure S1'''
    import pylab as plt
    plt.figure(figsize=(12,7.5))
    def rotate(p,phi):
        phi=phi/180*np.pi
        return np.array(p).dot(np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]]))
    D=[[6,1,40,'k',1,-1],[-4,-8,250,'k',-9,-6],[1,7,113,'k',-1,8],[6,0,45,'m',3,-5,135,90],
      [0,-9,257,'b',-6,-3,103,0],[0,7,13,'c',8,9,347,0]]
    for i in range(len(D)):
        a=np.array(D[i][:2])
        b=-speed*np.array([np.cos(D[i][2]/180*np.pi),np.sin(D[i][2]/180*np.pi)])
        c=D[i][4:6]
        G=[[a,a+b,c]]
        if len(D[i])>6:
            d=speed*np.array([np.cos(D[i][6]/180*np.pi),np.sin(D[i][6]/180*np.pi)])+a
            e=speed*np.array([np.cos(D[i][7]/180*np.pi),np.sin(D[i][7]/180*np.pi)])+a
            G[0].extend([d,e])
        G.append(np.array(G[0])-a[np.newaxis,:])
        G.append(rotate(np.copy(G[1]),D[i][2]).tolist())
        G.append(np.copy(G[-1]))
        G[-1][:,1]*=-1
        for j in range(4):
            if i<4 and j==3:continue
            plt.subplot(4,len(D),j*len(D)+1+[3,4,5,0,1,2][i])
            plt.plot([G[j][0][0],G[j][1][0]],[G[j][0][1],G[j][1][1]],color=D[i][3])
            plt.xlim([-10,10]);plt.ylim([-10,10]);
            plt.gca().set_aspect(1);
            if len(G[j])<=3:plt.plot(G[j][0][0],G[j][0][1],'o'+D[i][3])
            plt.plot(G[j][2][0],G[j][2][1],'x'+D[i][3])
            plt.gca().set_xticks([]);plt.gca().set_yticks([])
            if len(G[j])>3:
                plt.plot([G[j][0][0],G[j][3][0]],[G[j][0][1],G[j][3][1]],'--',color=D[i][3])
                plt.plot([G[j][4][0],2*G[j][0][0]-G[j][4][0]],[G[j][4][1],2*G[j][0][1]-G[j][4][1]],color='gray',zorder=-1)
                
            if j==0:
                plt.title('Saccade '+str([3,4,5,0,1,2][i]+1))
                plt.xlabel('↓shift↓')
            elif j==1:plt.xlabel('↓rotate↓')
            elif j==2 and i>=4: plt.xlabel('↓mirror↓')
    plt.savefig(FPATH+'expl.png',dpi=DPI,bbox_inches='tight',pad_inches=0)
    
def plotMST_regCoeffs(suf='b113',sufmpd=None,mage=[7],mspeed=[20],axs=[]):
    ''' plot Figure S2 (mean saccade targets) and S3 (reg. coefficients)'''
    if sufmpd is None:sufmpd=suf
    from matusplotlib import subplotAnnotate
    a2clr={0.45:'c',1.57:'m',2.69:'blue',3.14:'k',
           11.57:'lime',12.69:'g',22.69:'r'} 
    sv=len(axs)==0
    if sv: 
        import pylab as plt
        fig, axs =plt.subplots(figsize=(16,8),nrows=len(mage), ncols=len(mspeed))
        axs=np.reshape(axs,-1)
    with open(DPATH+f'sm{suf}1.wfit','rb') as f: w=pickle.load(f)
    if 'rhat' in w.keys():
        if not np.all(w['rhat'][0,:-1]<1.1): print(f'WARNING: {suf}, no convergence')
        print(f'{suf} max rhat:', np.max(w['rhat'][0,:-1]),w['nms'][0,np.argmax(w['rhat'][0,:-1])])
    meta=np.load(DPATH+f'meta{suf}1.npy')
    meta[meta[:,0]==0,0]=np.pi
    ordr=[3,0,1,2,4,5,6]
    #mean saccade target
    for m in range(len(mage)):
        for j in range(len(mspeed)):
            if int(suf[2])>0: 
                b2=w['mv']
                if suf[2]==2: b2=b2+w['bv']*mage[m]
            else: b2=0
            mst=w['b0']+mage[m]*w['b1']+b2*mspeed[j]
            ax=axs[m*3+j]
            for i in range(mst.shape[1]):
                k=ordr[i]
                c=a2clr[np.round(meta[i,0],2)]
                if i<4:trg=list(a2clr.keys())[i]
                else: trg=[np.pi,np.pi,np.pi/2][i-4]
                dst,r,rd=xy2dphi(mst[:,i,0,NA],mst[:,i,1,NA],trg)
                dst,r,rd=xy2dphi(mst[:,i,0,NA],mst[:,i,1,NA],r[0,0]/180*np.pi)
                NN=101
                if dst[1,0]>dst[2,0]:stop
                dsts=np.linspace(dst[1,0],dst[2,0],2)
                rs=np.mod(np.linspace(r[1,0],r[2,0],NN)/180*np.pi+2*np.pi,2*np.pi)
                d=sap(mst,[50,2.5,97.5],axis=0)
                mdist=np.linalg.norm(d[0,i,:])
                ax.plot(np.cos(rs)*mdist,np.sin(rs)*mdist,c)
                mang=np.arctan2(d[0,i,1],d[0,i,0])
                ax.plot(np.cos(mang)*dsts,np.sin(mang)*dsts,c)
            if j==0:ax.set_ylabel(f'Infant\'s Age: {mage[m]} months');
            if m==0:ax.set_title(f'Circle\'s Speed: {mspeed[j]} deg/s')
            ax.set_aspect(1)
            ta=[0.45,1.57,2.69,3.14,3.14,3.14,1.57]
            scl=.01
            for ii,kk in enumerate(list(a2clr.keys())):#draw physical bounce trajs
                ox,oy=[(0,0),(0,0),(0,0),(0,0),(0,scl),(0,-scl),(scl,0)][ii]
                ax.plot([ox,ox+np.cos(ta[ii])*15],[oy,oy+np.sin(ta[ii])*15],'--',c=a2clr[kk],alpha=0.3)
            ax.set_xlim([-10,6]);
            ax.set_ylim([-3,5]) 
            ax.grid(True,which='major',alpha=.3)
            ax.set_xticks(np.arange(ax.get_xlim()[0],ax.get_xlim()[1]+1,1))
            ax.set_yticks(np.arange(ax.get_ylim()[0],ax.get_ylim()[1]+1,1))
    for ax in axs: subplotAnnotate(loc='sw',nr=np.nan,ax=ax)       
    if sv: 
        plt.savefig(FPATH+suf+f'MST.png',dpi=DPI,bbox_inches='tight',pad_inches=0)
        fig, axs =plt.subplots(figsize=(11,6),nrows=2, ncols=2)
        axs=np.reshape(axs,-1)
    with open(DPATH+f'sm{sufmpd}0.wfit','rb') as f: w0=pickle.load(f)
    for s in range(2):
        dd=[w['mv'],w['b1']][s]
        d=sap(dd,[50,2.5,97.5],axis=0)
        dd0=[w0['mv'],w0['b1']][s]
        d0=sap(dd0,[50,2.5,97.5],axis=0)
        for i in range(dd.shape[1]):
            k=ordr[i]
            ax=axs[2*s]
            c=a2clr[np.round(meta[i,0],2)]
            if i<4:trg=list(a2clr.keys())[i]
            else: trg=[np.pi,np.pi,np.pi/2][i-4]
            
            dst,r,rd=xy2dphi(dd[:,i,0,NA],dd[:,i,1,NA],trg)
            dst,r,rd=xy2dphi(dd[:,i,0,NA],dd[:,i,1,NA],r[0,0]/180*np.pi)
            NN=101
            if dst[1,0]>dst[2,0]:stop
            dsts=np.linspace(dst[1,0],dst[2,0],2)
            rs=np.mod(np.linspace(r[1,0],r[2,0],NN)/180*np.pi+2*np.pi,2*np.pi)
            mdist=np.linalg.norm(d[0,i,:])
            ax.plot(np.cos(rs)*mdist,np.sin(rs)*mdist,c)
            mang=np.arctan2(d[0,i,1],d[0,i,0])
            ax.plot(np.cos(mang)*dsts,np.sin(mang)*dsts,c)
            ax=axs[2*s+1]
            vl=sap(-d0[:,i,0],[50,2.5,97.5])
            ax.plot(k,vl[0],'.',color=c)
            ax.plot([k,k],vl[1:],color=c)
        ax=axs[2*s]
        ax.set_aspect(1)
        ta=[0.45,1.57,2.69,3.14,3.14,3.14,1.57]
        scl=[.005,.002][s]
        for i,k in enumerate(list(a2clr.keys())):#draw physical bounce trajs
        
            ox,oy=[(0,0),(0,0),(0,0),(0,0),(0,scl),(0,-scl),(scl,0)][i]
            ax.plot([ox,ox+np.cos(ta[i])*15],[oy,oy+np.sin(ta[i])*15],'--',c=a2clr[k],alpha=0.3)
        ax.set_xlim([[-.25,.15],[-.8,.8]][s]);ax.set_ylim([[-.1,.2],[-.6,.4]][s]) 
        if s==0:ax.set_title('Outcome: Mean Saccade Target')
        ax.grid(True,which='major',alpha=.3)
        scl=[.05,.2][s]
        ax.set_xticks(np.arange(ax.get_xlim()[0],ax.get_xlim()[1]+scl,scl))
        ax.set_yticks(np.arange(ax.get_ylim()[0],ax.get_ylim()[1]+scl,scl))
        unt='$\\frac{\\mathrm{deg}}{'+['\\mathrm{deg}/s}$','\\mathrm{month}}$'][s]
        ax.set_ylabel(unt);
        ax.set_ylabel([f'Predictor: Circle\'s Speed',f'Predictor: Infant\'s Age'][s]);
        ax=axs[2*s+1]
        
        ax.set_ylim([[-.15,.15],[-.6,.6]][s]) 
        scl=[.05,.2][s]
        ax.set_yticks(np.arange(ax.get_ylim()[0],ax.get_ylim()[1]+scl,scl))
        ax.set_xticks([]) 
        if s==0: ax.set_title('Outcome: Displacement')
        ax.set_ylabel(unt);
        ax.grid(True,which='major',alpha=.3)
    for ax in axs: subplotAnnotate(loc='nw',nr=np.nan,ax=ax)
    if sv: plt.savefig(FPATH+suf+f'RC.png',dpi=DPI,bbox_inches='tight',pad_inches=0)
##############################################
  
def printStats(suf,sufmpd=None):
    ''' print sample information (incl. Table 1) and estimates 
        described in Methods, Results and Discussion and in Supplementary Materials'''
    if sufmpd is None:sufmpd=suf
    np.set_printoptions(suppress=True)
    print('''print sample statistics''')
    from matusplotlib import ndarray2latextable,figure
    R=np.zeros((10,7),dtype=object)
    R[2:,1:]=np.load(DPATH+f'metab{suf}1.npy')[:,5:][[-1,0,3,2,1,4,5,6],:]
    R[1,1:]=np.load(DPATH+f'metal{suf}0.npy')[5:]
    R[0,:]=np.array([' ','Nr. Saccades','Mean Velocity','Std Velocity','Nr. Participants','Mean Age','Std Age'])
    R[:,0]=np.array([' ','LM Saccades','Bounce Saccades','  Black','  Blue','  Magenta','  Cyan','  Lime','  Green','  Red'])
    ndarray2latextable(R,decim=[0,0,1,1,0,1,1])
    print('hypothesis test')
    p=np.load('data/ht00.npy')
    print(f'p(H_inv)={p[0]}, p(H_none)={p[1]},p(H_phys)={p[3]},p(H_true)={p[4]}')
    print('''standard deviation''')
    if int(suf[2]) in (1,4):
        for sf in [f'b{suf}1',f'b{sufmpd}0',f'l{suf}0']:
            with open(DPATH+f'sm{sf}.wfit','rb') as f: w=pickle.load(f)
            tmp=w['gsm'][:,:,0]/w['gsm'][:,:,1]
            for k in range(2):
                print(f'{sf}, E[sm_{k}|a=10m]= ',np.round(sap(tmp[:,k]+w['vs'][:,k],[50,2.5,97.5],axis=0),2))
                print(f'{sf}, E[sm_{k}|a=4m]=',np.round(sap(tmp[:,k]+7*w['vs'][:,k],[50,2.5,97.5],axis=0),2))
            if int(sf[1]):print(f'{sf}, bo= ',sap(w['bo'],[50,2.5,97.5],axis=0)[:,0])
    print('''motion-parallel displacement''')
    with open(DPATH+f'sml{suf}0.wfit','rb') as f: w=pickle.load(f)
    if 'rhat' in w.keys() and not np.all(w['rhat'][0,:-1]<1.1): 
        print(f'WARNING: sml{suf}0, no convergence')
    print(f'sml{suf}0 max rhat:', np.max(w['rhat'][0,:-1]),w['nms'][0,np.argmax(w['rhat'][0,:-1])])
    def sacTar(a=7,v=20,percs=[50,2.5,97.5]): 
        res= w['b0']+w['b1']*a +v*[0,w['mv'],w['bv']*a+w['mv'],w['mv']][int(suf[1])]
        res=np.round(sap(res[:,0,0],percs,axis=0),2).T
        print(f'age: {a}, vel: {v} out: {res}') 
    sacTar(a=7,v=20)
    sacTar(a=10)
    sacTar(v=5)
    sacTar(v=40)
    sacTar(v=0)
    print('age deg/month:',np.round(sap(w['b1'][:,0,0],[50,2.5,97.5]),2))
    if int(suf[1])==2: 
        print(np.round(sap(w['mv'][:,0,0]+w['bv'][:,0,0]*7,[50,2.5,97.5]),2))
        print(np.round(sap(w['bv'][:,0,0]*(11-4)*1000,[50,2.5,97.5]),1))
    elif int(suf[1]) in (1,3): 
        print('vel deg/(10 deg/s):', np.round(sap(w['mv'][:,0,0]*10,[50,2.5,97.5]),2))
        
    dat=np.load(DPATH+f'datl{suf}0.npy',allow_pickle=True)
    dat=dict(dat.tolist())
    vp=dat['vp'][:,0];coh=(dat['vp'][:,1]-4)//3
    g=w['g'][:,0,0,:]+w['mv'][:,0,0,NA]*20
    res=np.zeros((np.max(vp)+1,3,g.shape[0]))*np.nan
    for i in range(vp.shape[0]):res[vp[i],coh[i],:]=g[:,i]
    def mscorrcoef(x,y):
        sel=np.logical_and(~np.isnan(x[:,0]),~np.isnan(y[:,0]))
        r=np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            r[i]= np.corrcoef(x[sel,i],y[sel,i])[0,1] 
        return r,sel.sum()
    un,cn=np.unique(vp,return_counts=True)
    print('n_tot= ',un.size)
    print('n_twice= ',(cn==2).sum())
    print('n_thrice= ',(cn==3).sum())
    for i in range(3): print(f'coh {i*3+4}m, n=',(coh==i).sum())
    
    for ij in [[0,1],[1,2],[0,2]]:
        tmp2=mscorrcoef(res[:,ij[0],:],res[:,ij[1],:])
        tmp=np.round(sap(tmp2[0],[50,2.5,97.5]),2)
        print(f'cor between coh {ij[0]*3+4}m,{ij[1]*3+4}m: ',tmp,tmp2[1])    
    print('''print amplitude stats''')
    for sf in ['l0'+suf[1]+'02','b0'+suf[1]+'02']:
        print(['linear motion','bounce'][sf[0]=='b'])
        with open(DPATH+f'sm{sf}.wfit','rb') as f: w=pickle.load(f)
        assert(np.all(w['rhat'][0,:-1]<1.1))
        sacTar(a=7,v=20)
        print('age deg/month:',np.round(sap(w['b1'][:,0,0],[50,2.5,97.5]),2))
        if int(suf[1])==2:
            print(np.round(sap(w['mv'][:,0,0]+w['bv'][:,0,0]*7,[50,2.5,97.5]),2))
            print(np.round(sap(w['bv'][:,0,0]*(11-4),[50,2.5,97.5]),1))
        elif int(suf[1])==1: print('vel deg/(deg/s):', np.round(sap(w['mv'][:,0,0]*10,[50,2.5,97.5]),2))
        #additional stats for discussion section
        sacTar(a=4,v=24)
        sacTar(a=5,v=20)
       
if __name__=='__main__':
    # loading and preprocessing
    vpinfo=getMetadata(showFigure=False)
    checkFiles(vpinfo)
    extractFromDataFiles()
    accuracyCorrection(method='no pooling',applyCor=True,exclude=False)
    changeUnits()
    extractSaccades()
    # main statistical analyses 
    rotateSaccades()
    # mean saccade target & anticipated angle
    fitRegression(bounceSac=1,pCenter=1,pVelM=1,pVelS=4,yType=1,ssmm=10)#b1141
    # motion-parallel displacement
    fitRegression(bounceSac=1,pCenter=1,pVelM=1,pVelS=4,yType=0,ssmm=10)#b1140
    fitRegression(bounceSac=0,pCenter=1,pVelM=1,pVelS=4,yType=0,ssmm=10)#l1140
    #bounce saccade frequency
    computeFreqAmp()
    # saccade amplitude
    fitRegression(bounceSac=0,pCenter=0,pVelM=1,pVelS=0,yType=2,ssmm=1)#b0102
    fitRegression(bounceSac=1,pCenter=0,pVelM=1,pVelS=0,yType=2,ssmm=10)#l0102
    # supplementary statistical analyses
    rotateSaccades(CSDIST=10);
    fitRegression(bounceSac=1,pCenter=1,pVelM=1,pVelS=4,yType=1,ssmm=10, suf='csdist10')#b1141csdist10
    rotateSaccades(SACS=[-.2,0]);
    fitRegression(bounceSac=1,pCenter=1,pVelM=1,pVelS=4,yType=1,ssmm=10, suf='sacs200')#b1141sacs200
    rotateSaccades(SACMAXL=1);
    fitRegression(bounceSac=1,pCenter=1,pVelM=1,pVelS=4,yType=1,ssmm=10, suf='sacmaxl1')#b1141sacmaxl1
    rotateSaccades()
    # results presentation
    plotTerms()
    plotTrajectory() 
    plotMain('114',sufmpd='014')
    printStats('114',sufmpd='014')

    # create figures from supplementary materials
    plotExplanation();
    plotMST_regCoeffs('b114',sufmpd='b014',mage=[4,7,10],mspeed=[5,20,40])
    # further analyses briefly summarized in supplementary materials
    hypothesisTest(suf='b1141',ML=False,BALANCED=False)# bayesian hypothesis test
    plotMST_AA_MPD('b1141sacs200') # same analyses, but bounce sac. start upto 0.2 s after bounce allowed
    plotMST_AA_MPD('b1141csdist10') # same analyses, but allow catch-up saccades with sac-circle distance of 10 deg instead of 7 deg 
    plotMST_AA_MPD('b1141sacmaxl1') # same analyses, but allow catch-up saccades with max. length of 1 s instead of  
