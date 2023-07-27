from scipy.interpolate import interp1d
import numpy as np
import os,pickle

SEED=8
NCORE=6
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


# preprocessed information is put in variable D which is saved as 
# multiple file in the data folder, use D=loadData() to load D
# D[6][i] is a ndarray with Sx10 element where S is the number of 
#   events extracted by remodnav, columns are: 0 - event typ (0=saccade),
#   1 - time when event started (in sec since start of experiment)
#   2 - time when event ended (in sec since start of experiment)
# D[9][i] is a ndarray with Sx14 elements where S is the number of bounces
#   watched by participant i; 0- time when bounce occured; 1-trial; 
#   2- frame; 3-motion type; 4,5-bounce location;
#   6,7-old stim-movement direction; 8-13-new stim-movement direction assuming 
#   physics-based, inverted or 90-deg bounce
def checkCollision(pos,vel,collisionType,verbose=False):
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
    for k in range(4):
        np.save(ddir+nm+'%d'%k,D[k])
    for k in range(4,11):
        for i in range(len(D[k])):
            if D[k][i] is None: continue
            np.save(ddir+nm+'%d_%03d'%(k,i),D[k][i]) 
def loadData(ddir='data/',nm='D'):
    R=[[],[],[],[],[],[],[],[],[],[],[]]
    for k in range(4):
        R[k]=np.load(ddir+nm+'%d.npy'%k)
    for k in range(4,11):
        for i in range(R[0].shape[0]):
            R[k].append(None)
            try: R[k][-1]=np.load(ddir+nm+'%d_%03d.npy'%(k,i))
            except:pass
    return R  
def rlstsq(a,b,plot=[]):
    sel=np.logical_and(~np.isnan(a),~np.isnan(b))
    if sel.sum()<2:return
    a=a[sel];b=b[sel]
    #if plot: plt.figure();plt.plot(a,b,'.k')
    a=np.array([np.ones(a.size),a]).T
    #print(a.shape,b.shape)
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
    surp=False
    for suf in ['01','02','03','']:
        fnsall=os.listdir(OPATH+suf)
        fns=list(filter(lambda x: x[-3:]=='.et',fnsall))
        fns=fns+list(filter(lambda x: x[-4:]=='.log',fnsall))
        fns=np.sort(fns)
        #vpns=np.concatenate([infoa,infob])[:,[0,1]]
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

    vpinfo=getMetadata(showFigure=False)#[375:,:]
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
        #ts=min(float(msg[0][1]),float(et[0][1]))
        #if i==9:stop
        # process messages into trial info and calibration info
        ti=0
        for k in range(len(msg)):
            #print('k=',k,ti,msg[k][4])
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
        #np.save('stim/%03d'%i,stim);continue
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
            #D[4][-1][-1][GBX]=np.nanmean([float(et[k][GLX]),float(et[k][GRX])])
            #D[4][-1][-1][GBY]=np.nanmean([float(et[k][GLY]),float(et[k][GRY])])
        assert(trl==len(trid)-2)
        D[4][-1]=np.array(D[4][-1])
        # change time to sec.
        if fn=='/pursuitVp92c7M.log': #one file needs correction
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
    saveData(D)
    
def partialPooling(dat,compileStan=True,runStan=False,ssmm=1):
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
        fit=smsv.sampling(data=dat,chains=NCORE,n_jobs=NCORE,
            thin=int(max(2*ssmm,1)),iter=int(ssmm*2000),warmup=int(ssmm*1000))
        saveStanFit(fit,'fitcalib')
        print(fit)
    w=loadStanFit('fitcalib')
    return np.median(w['o'],0),np.median(w['mo'],0)
        
def accuracyCorrection(method='complete pooling',applyCor=True,exclude=False):
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
    if method=='complete pooling':
        for e in range(2):
            g=[]
            for i in range(len(D[2])):
                d=D[2][i]
                ctrue=d[1:10,1:3]
                c=d[1:10,3+e*2:5+e*2]
                coef=[np.nan,np.nan,np.nan,np.nan,np.nan]
                assert(np.all(np.isnan(c[:,0])==np.isnan(c[:,1])))
                sel=~np.isnan(c[:,0])
                if np.isnan(c[:,0]).sum()>5:# don't apply correction when 3 or less calibration locations available
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
            #print(dat['y'].shape,dat['c'].shape);bla  
            np.save('c',dat['c'])    
            o,mo=partialPooling(dat,compileStan=False,ssmm=1)
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
                    #print(tms[j][0],e,int(d1),corf)
                    D[2][i,[0,-1][e],11+int(d2)]=corf
                    
                    #D[4][tms[j][0]][:,GLX+2*e+int(d1)]+=corf
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
                #D[4][tms[j][0]][:,[GLX+2*e,GLY+2*e]]+=corf 
        else: stop
    res=np.nanmean(np.linalg.norm(D[2][:,[0,-1],11:13],axis=2),axis=0)
    res=np.sqrt(np.square(res).sum()/2)   
    print('accuracy in degrees: ', res)
    if applyCor: saveData(D)
def changeUnits():
    D=loadData()
    # change units
    MD=70# default monitor distance used throughout data recording
    # location of screen center in mm relative to ET at [0,0,0]
    def _chunits(H,dz,sc=[0,0,0]):#sc=[0,-165,25]):
        return (H/180*np.pi*MD*10-a2d(sc[:2]))/(dz-sc[2])*180/np.pi
    for i in range(len(D[4])): 
        G=D[4][i]
        if G is None:continue
        for k in [GLX,GRX,GSTIMX]:
            if G.shape[1]==18:#
                if k==GSTIMX:dz=np.nanmedian(np.nanmean(G[:,[14,17]]))
                else:dz=np.nanmedian(G[:,{GLX:14,GRX:17}[k]])
            else: dz=620
            G[:,[k,k+1]]=_chunits(G[:,[k,k+1]],dz)
        for k in range(5): 
            if k>0: tmp=_chunits(D[9][i][:,[4+2*k,5+2*k]],dz,sc=[0,0,25])
            else: tmp=_chunits(D[9][i][:,[4+2*k,5+2*k]],dz)
            D[9][i][:,[4+2*k,5+2*k]]=tmp
        D[3][i,:,TOVEL]=_chunits(D[3][i,:,TOVEL,NA],dz,sc=[0,0,25])[:,0]
        #D[2][i,[0,-1],11:13]=_chunits(D[2][i,[0,-1],11:13],dz,sc=[0,0,25])
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
    from remodnav import EyegazeClassifier
    #from nslr_hmm import classify_gaze
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
        #D[6].append([]);
        f=1/D[0][i,METHZ]#sample length in sec.
        s=np.arange(np.ceil(D[4][i][-1,GC]/f))*f
        MBD=0.1#minimum blink duration in seconds
        e=2#for e in range(3):
        data={'x':None,'y':None}
        assert(np.all(np.isnan(D[4][i][:,GBX])==np.isnan(D[4][i][:,GBX+1])))
        
        tmp=np.copy(D[4][i][:,GBX])
        # interpolate to regular intervals using nearest-neighbour method
        # interpolate nans
        sel=np.ones(tmp.size,dtype=np.bool)
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
            elif ev[2]>lst[1]:
                #if ev[1]<lst[1]:
                #    ev[1]=lst[1]
                #    ev[4]=lst[0]
                evs2.append(ev+[0,np.nan,np.nan,np.nan,np.nan])
        D[6].append(np.array(evs2))
    # extract events with nslr
    D[5]=[];
    for i in []:#range(len(D[4])):
        if D[4][i] is None:
            D[5].append(None)
            continue
        #D[5].append([]);
        #for e in range(3):
        e=2
        sel=~np.isnan(D[4][i][:,GLX+2*e])
        if sel.sum()<=1:
            D[5].append([])
            continue
        sc,segs,sgc=classify_gaze(D[4][i][sel,GC],D[4][i][sel,GLX+2*e:GRX+2*e])
        #structural_error=1.3, optimize_noise=False)
        evs=[]
        cd={1:1,2:0,4:2,3:0}
        for k in range(len(segs.segments)):
            rs=segs.segments[k]
            #sel=np.logical_and(rs.t[0]<D[4][i][:,GC],D[4][i][:,GC]<rs.t[1])
            #nn=0#=np.isnan(D[4][i][sel,GLX+2*e]).mean()
            a=np.max((rs.t[0]>=D[4][i][:,GC]).nonzero()[0])
            b=np.min((rs.t[1]<=D[4][i][:,GC]).nonzero()[0])
            #tmp=np.roll(D[4][i][a:b,GLX+2*e],1);tmp2=tmp=np.roll(D[4][i][a:b,GLX+2*e],-1)
            #sel=(~np.isnan(D[4][i][a:b,GLX+2*e]+tmp+tmp2)).nonzero()[0]
            #sel=(~np.isnan(D[4][i][a:b,GLX+2*e])).nonzero()[0]
            #if len(sel)==0:continue
            #b=np.max(sel)+a
            #a=np.min(sel)+a
            if b-a>1:
                evs.append(np.concatenate([[cd[sgc[k]]],D[4][i][[a,b],GC],[a,b]]))
        #print(i,e,len(evs))
        D[5].append(np.array(evs))
    saveData(D)

def plotGaze(D,i,g,xlm=None,showEvents=[]):
    import pylab as plt
    s=D[3][i][g*5,TFS];f=D[3][i][g*5:g*5+5,TFE]
    assert(~np.isnan(s))
    if np.isnan(f[-1]):print('trialblock not completed')
    f=np.int32(f[~np.isnan(f)][-1])
    s=int(s)
    for e in range(3):
        #plt.fig
        ure(figsize=(D[4][i][-1,GC]/10*8,4))
        plt.figure(figsize=(16,4))
        for ax in range(2):
            plt.ylabel(['L','R','B'][e])
            plt.plot(D[4][i][s:f,GC],D[4][i][s:f,GSTIMX+ax],color='gr'[ax],alpha=0.5)
            plt.plot(D[4][i][s:f,GC],D[4][i][s:f,GLX+ax+2*e],lw=2,color='gr'[ax])
        plt.grid()
        if e==2:
            for k in showEvents:#range(2):
                h=0
                for v in range(D[5+k][i].shape[0]):
                    tmp=D[5+k][i][v,:]
                    if tmp[1]>D[4][i][f,GC] or tmp[2]<D[4][i][s,GC]:continue

                    if tmp[0]==0:plt.plot(tmp[1:3],[48+k*7,53+k*7],color='k')
                    else: 
                        plt.plot(tmp[1:3],np.zeros(2)+48+k*7+h,lw=1,color=['r','g'][int(tmp[0])-1])
                        h=(h+1)%5
        if not xlm is None: plt.xlim(xlm)
        sel=~np.isnan(D[4][i][:,GSTIMX])
        print(1-np.isnan(D[4][i][sel,GLX+ax+2*e]).mean())
    plt.title(D[0][i,:2]);
    #plt.savefig('figs/evseg%03d%d'%(i,e),dpi=100,bbox='tight')

def rotateSaccades(SACS=0.15,SACE=0,SACMAXL=0.2,CSDIST=7):
    D=loadData()
    out=[]# 0:2 - gaze pos rel to bounce origin, 2 - angle of phys bounce, 3- velocity, 4 - trial id, 5:7- stim rel to bounce origin, 7:9 - gaze pos rel to stim pos, 9:11- location of origin, 11 - saccade amplitude, 12:14 - sac end relative to when bounce ocurred
    for i in range(len(D[4])):
        out.append([]);
        if D[4][i] is None: continue
        S=D[6][i]
        for j in range(S.shape[0]):
            if not S[j,0]==0:continue # iterate over saccades
            sacamp=np.linalg.norm(np.diff(D[4][i][np.int32(S[j,3:5]),GBX:GBY+1],axis=0))
            D[6][i][j,5]=0#keep track of sac type for later
            tmp=D[4][i][np.int32(S[j,3:5]),GF]
            if tmp[0]==-1 or tmp[1]==-1: continue
            if tmp[0]==-1:trl=int(D[4][i][np.int32(S[j,4]),GT])
            else: trl=int(D[4][i][np.int32(S[j,3]),GT])
            if D[3][i][trl,TSKIP]: continue
            a=D[4][i][np.int32(S[j,4]),[GBX,GBY]];
            dst=np.linalg.norm(a-D[4][i][np.int32(S[j,4]),[GSTIMX,GSTIMY]])
            if dst>CSDIST:continue
            
            if type(SACS) is list and len(SACS)==2: sel=np.logical_and(S[j,1]-D[9][i][:,0]>SACS[0],S[j,1]-D[9][i][:,0]<SACS[1])
            else: sel=np.logical_and(S[j,1]-D[9][i][:,0]<SACS,S[j,2]-D[9][i][:,0]>SACE)
            sel=np.logical_and(sel,S[j,2]-S[j,1]<SACMAXL)
            sel=np.logical_and(sel,D[9][i][:,3]>-1)
            # compute physical bounce angle
            if sel.sum()>1:continue
            # compute sac target relative to circle's location
            rr=D[4][i][np.int32(S[j,4])-1:np.int32(S[j,4])+1,[GSTIMX,GSTIMY]]
            th2=np.arctan2(rr[1,1],rr[1,0])
            rrout= (rr[1,:]/np.linalg.norm(rr[1,:]))
            assert(np.isclose(rrout[0],np.cos(th2)))
            #th=np.arctan2(rr[1,1]-rr[0,1],rr[1,0]-rr[0,0])
            th=D[4][i][np.int32(S[j,4]),GTHETA]
            R=np.array([[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]])#clockwise
            gazeRstim=np.squeeze(R.dot(a-rr[1,:])).tolist()
            rrout=R.dot(rrout).tolist()
            assert(len(rrout)==2)
            assert(np.all(~np.isnan(rrout)))
            assert(not np.isnan(D[3][i][trl,TOVEL])) 
            if sel.sum()==0: 
                # put origin at the positive vertical half
                #if (-th2+2*np.pi+th)%(2*np.pi)>np.pi:gazeRstim[1]= -gazeRstim[1]
                out[-1].append([np.nan,np.nan,np.nan,D[3][i][trl,TOVEL],trl,np.nan,np.nan]+gazeRstim+rrout+[sacamp,np.nan])
                D[6][i][j,5:]=np.array([1,-th,rr[1,0],rr[1,1],0])
                continue
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
            assert(np.isnan(tmp[0])==np.isnan(tmp[1]))
            D[6][i][j,5:]=np.array([2+int(tmp3[1]<0),th,rp[4],rp[5],rp[0]])
            
            out[-1].append(np.array(tmp.tolist()+[th%np.pi,D[3][i][trl,TOVEL],trl]+tmp2.tolist()+gazeRstim+rrout+[sacamp, S[j,2]-rp[0]]))
            
           
        if len(out[-1]):out[-1]=np.array(out[-1])
    D[10]=out
    saveData(D)

def computeSacStats(tartrl=range(0,25),compileStan=True,suf='',bounceSac=1,
    poolCond=True,pCenter=0,pVelM=1,pVelS=0,yType=1,minSacNr=1,ssmm=1):
    ''' tartrl - list with ids of trials that will be analyzed, default: all trials
        bounceSac - if 1 analyze bounce saccades, otherwise linear-motion saccades
        poolCond - pool data across conditions (based on angle of incidence)
        pCenter - if 1 includes direction of screen center as predictor
        pVelM - if 1 includes velocity as predictor of mean y
        pVelS - if 0 excludes velocity as predictor of variance of y 
            if 1 uses additive regression model
            if 2 uses multiplicative regression model
        yType - if 1 returns sac tar with bounce loc as origin
            if 0 the stim loc is the origin, if 2 takes amplitude as outcome,
        minSacNr - exclude infants with less than minSacNr of saccades
    '''
    D=loadData()
    def _prep2dat(prep):
        y=[];xi=[];k=-1;sel=[];xc=[];xa=[];phis=[];xv=[];bts=[];stims=[];xo=[];
        for i in range(len(prep)): 
            if len(prep[i])==0: continue
            if yType==2: ytar=prep[i][:,11,NA]
            elif yType==0 or bounceSac==0: ytar=prep[i][:,7:9]
            else: ytar=prep[i][:,:2]
            if bounceSac==1: sel2= ~np.isnan(prep[i][:,2])
            else: sel2= np.isnan(prep[i][:,2])
            sel2=np.logical_and(sel2,~np.isnan(ytar[:,0]))
            sel2=np.logical_and(sel2,np.isin(prep[i][:,4],tartrl))
            if sel2.sum()<minSacNr:continue
            k+=1
            xa.append(D[0][i,MAGE]/30) 
            y.append(ytar[sel2,:])
            xi.append(k*np.ones(sel2.sum()));
            th=np.round(prep[i][sel2,2],4)
            trl=np.int32(prep[i][sel2,4]/5)
            bt=np.array(CC)[np.array(CB[D[0][i,MCOND]])[trl]][:,0]
            if not poolCond:xc.append(D[0][i,MCOND]*np.ones(sel2.sum())*10+th)
            else: xc.append(th+bt*10)
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
        #print(np.unique(xc));bla
        k=np.unique(xc).tolist()
        K=y.shape[1]
        #y=np.squeeze(y)
        print('M=',xi.size)
        #np.save('xi',xi);stop
        if bounceSac==0:
            dat={'N':len(xa),'xa':xa,'M':xi.size,'K':K, 'xi':xi,'y':y,'xv':xv,'xo':xo}
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
            print(xa.shape,vpps)
            meta.append([k[i]-100,phis[j[0]],bts[j[0]]]+stims[sel,:].mean(0).tolist()+
                [sel.sum(),xv[sel].mean(),xv[sel].std(),vpps.size,
                xa[vpps-1].mean(),xa[vpps-1].std()])
        meta.append([-1,-1,-1,-1,-1,xi.size,xv.mean(),xv.std(),xa.size,xa.mean(),xa.std()])
        xc=np.int32(xc)+1
        #sel=xc==1
        #xc=xc[sel];xi=xi[sel];xv=xv[sel];y=y[sel,:];xo=xo[sel,:]
        if yType==2: xc[:]=1
        assert(xi.shape[0]==y.shape[0])
        dat={'N':xa.size,'xa':xa,'M':xi.size,'xi':xi,'K':K,
            'y':y,'xc':xc,'L':np.max(xc),'xv':xv,'xo':xo}
        return dat,meta

    mdlb='''
        data{{
            int N;int M;int L;int K;
            vector[N] xa;
            int<lower=1> xi[M];
            int<lower=1,upper=L> xc[M];
            real y[M,K];real xo[M,K];
        }} parameters{{
            real<lower=-20,upper=20> b0[L,K];
            real<lower=-5,upper=5> b1[L,K];
            real<lower=-50,upper=50> g[L,K,N];
            real<lower=-20,upper=20> bo;
            real<lower=0,upper=30> sn[K];
            real<lower=0,upper=40> sm[L,K,N];
            real<lower=0,upper=100> gsm[L,K,2];
        }}model{{
            for (k in 1:K){{
                //sn[k]~cauchy(0,sK[3]);
                for (n in 1:N){{
                    for (l in 1:L)
                        sm[l,k,n]~gamma(gsm[l,k,1],gsm[l,k,2]);
                        g[l,k,n]~normal(b0[l,k]+b1[l,k]*xa[n],sn[k]);
                }}    
                for (m in 1:M) y[m,k]~normal(g[xc[m],k,xi[m]]{pred},{std});
            }}
        }}
        '''
        
    mdlbv='''
        data{{
            int N;int M;int L;int K;
            vector[N] xa;
            int<lower=1> xi[M];
            int<lower=1,upper=L> xc[M];
            real y[M,K]; real xo[M,K];real xv[M];  
        }} parameters{{
            real<lower=-10,upper=10> b0[L,K];
            real<lower=-2,upper=2> b1[L,K];
            real<lower=-20,upper=20> g[L,K,N];
            real<lower=-10,upper=10> bo;
            real<lower=0,upper=20> sn[K];
            real<lower=0,upper=30> sm[K,N];
            real<lower=0,upper=300> gsm[K,2];
            
            //real<lower=-10,upper=10> v[L,K,N];
            real<lower=-10,upper=10> mv[L,K];
            real<lower=-10,upper=10> bv[L,K];
            //real<lower=0,upper=20> vs[L];
            //real<lower=0,upper=10> sv[L,K];
            //real<lower=0,upper=100> gsv[2];
            
        }}model{{
            for (k in 1:K){{
                //sn[k]~cauchy(0,sK[3]);
                //for (l in 1:L){{vs[l,k]~cauchy(0,vss[k]);}}
                //for (l in 1:L){{
                //    gsm[l,k,1]~cauchy(0,sgsm);
                //    gsm[l,k,2]~cauchy(0,sgsm);
                //    sv[l,k]~gamma(gsv[1],gsv[2]);
                //}}
                for (n in 1:N){{
                    sm[k,n]~gamma(gsm[k,1],gsm[k,2]);
                    for (l in 1:L){{
                        //sm[l,k,n]~gamma(gsm[k,1],gsm[k,2]);
                        g[l,k,n]~normal(b0[l,k]+b1[l,k]*xa[n],sn[k]);
                        //v[l,k,n]~normal(mv[l,k],sv[l,k]); v[xc[m],k,xi[m]]
                }}}}   
                for (m in 1:M) y[m,k]~normal(g[xc[m],k,xi[m]]{pred},sm[k,xi[m]]{std});
            }}
        }}
        ''' 
        
    mdlL='''
    data{{
        int N;int M;int K;
        vector[N] xa;
        int<lower=1> xi[M];
        real xv[M];real y[M,K]; real xo[M,K];
    }} parameters{{
        real<lower=-10,upper=10> b0[K];
        real<lower=-2,upper=2> b1[K];
        real<lower=-20,upper=20> g[K,N];
        real<lower=-10,upper=10> bo;
        real<lower=0,upper=20> sn[K];
        real<lower=0,upper=30> sm[K,N];
        real<lower=-2,upper=2> v[K,N];
        real<lower=-10,upper=10> mv[K];
        real<lower=0,upper=2> sv[K];
        real<lower=-10,upper=10> bv[K];
        real<lower=-3,upper=3> vs[K];
        real<lower=0,upper=300> gsm[K,2];
    }}model{{
        for (k in 1:K){{
            //vs[k]~cauchy(0,1);
            //sn[k]~cauchy(0,sK[3]);
            for (n in 1:N){{
                sm[k,n]~gamma(gsm[k,1],gsm[k,2]);
                g[k,n]~normal(b0[k]+b1[k]*xa[n],sn[k]);
                //v[k,n]~normal(mv[k]+bv[k]*xa[n],sv[k]);
            }}    
            for (m in 1:M) y[m,k]~normal(g[k,xi[m]]{pred},{std});
        }}
    }}
    ''' 
 
    import stan,pickle
    from matusplotlib import fit2dict
    cd=f'{pCenter}{pVelM}{pVelS}{yType}'
    dat,meta=_prep2dat(D[10])
    np.save(DPATH+'dat'+'lb'[bounceSac]+f'{cd}{suf}',dat,allow_pickle=True)
    np.save(DPATH+'meta'+'lb'[bounceSac]+f'{cd}{suf}',meta)
    if bounceSac==0: 
        if compileStan:
            ml=mdlL.format(
                pred=['','+bo*xo[m,k]'][pCenter]+['','+v[k,xi[m]]*xv[m]','+xv[m]*(mv[k]+bv[k]*xa[xi[m]])'][pVelM],
                std=['sm[k,xi[m]]','sm[k,xi[m]]+vs[k]*xv[m]','sm[k,xi[m]]*pow(xv[m],vs[k])','sm[k,xi[m]]*xv[m]'][pVelS])
            print(ml)
            smsv=stan.build(model_code=ml)
            with open(DPATH+f'sml{cd}.pkl', 'wb') as f: pickle.dump(smsv, f)
        with open(DPATH+f'sml{cd}.pkl','rb') as f: smsv=pickle.load(f)
        fit=smsv.sampling(data=dat,chains=NCORE,n_jobs=NCORE,
            thin=int(max(ssmm*1,1)),iter=int(ssmm*1000),warmup=int(ssmm*500))
        w=fit2dict(fit)
        with open(DPATH+f'sml{cd}{suf}.wfit','wb') as f: pickle.dump(w,f,protocol=-1)
        return

    if compileStan:
        ml=mdlbv.format(pred=['','+bo*xo[m,k]'][pCenter]+['','+mv[xc[m],k]*xv[m]','+(mv[xc[m],k]+bv[xc[m],k]*xa[xi[m]])*xv[m]'][pVelM],
            std=['','+vs[xc[m],k]*xv[m]','*pow(xv[m],vs[xc[m],k])', '*xv[m]','','','','',''][pVelS])
        print(ml)
        sm=stan.build(program_code=ml,data=dat,random_seed=SEED)
        with open(DPATH+f'smb{cd}.pkl', 'wb') as f: pickle.dump(sm, f)
    with open(DPATH+f'smb{cd}.pkl','rb') as f: sm=pickle.load(f)
    
    w=sm.sample(num_chains=NCORE,num_thin=int(max(ssmm*1,1)),
        num_samples=int(ssmm*500),num_warmup=int(ssmm*500)) 
    converged,nms,rhat=printRhat(w)
    w2={'nms':nms,'rhat':rhat}
    for k in w.keys():w2[k]=np.rollaxis(w[k],-1,0)
    with open(DPATH+f'smb{cd}{suf}.wfit','wb') as f: pickle.dump(w2,f,protocol=-1)
    
    
    

def plotExpl():
    from matusplotlib import subplotAnnotate,subplot
    import pylab as plt
    plt.figure(figsize=(12,7.5))
    def rotate(p,phi):
        phi=phi/180*np.pi
        return np.array(p).dot(np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]]))
    al=20
    D=[[6,1,40,'k',1,-1],[-4,-8,250,'k',-9,-6],[1,7,113,'k',-1,8],[6,0,45,'m',3,-5,135,90],
      [0,-9,257,'b',-6,-3,103,0],[0,7,13,'c',8,9,347,0]]
    for i in range(len(D)):
        a=np.array(D[i][:2])
        b=-al*np.array([np.cos(D[i][2]/180*np.pi),np.sin(D[i][2]/180*np.pi)])
        c=D[i][4:6]
        G=[[a,a+b,c]]
        if len(D[i])>6:
            d=al*np.array([np.cos(D[i][6]/180*np.pi),np.sin(D[i][6]/180*np.pi)])+a
            e=al*np.array([np.cos(D[i][7]/180*np.pi),np.sin(D[i][7]/180*np.pi)])+a
            G[0].extend([d,e])
        G.append(np.array(G[0])-a[np.newaxis,:])
        G.append(rotate(np.copy(G[1]),D[i][2]).tolist())
        G.append(np.copy(G[-1]))
        G[-1][:,1]*=-1
        for j in range(4):
            if i<4 and j==3:continue
            plt.subplot(4,len(D),j*len(D)+1+i)
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
                plt.title('Saccade '+str(i+1))
                plt.xlabel('↓shift↓')
            elif j==1:plt.xlabel('↓rotate↓')
            elif j==2 and i>=4: plt.xlabel('↓mirror↓')
            
    from matplotlib.patches import Wedge
    ax=subplot(4,3,10)
    ss=0.035
    phi=180-np.arctan(0.4-ss)/np.pi*180
    ax.add_patch(Wedge([0,ss],0.4,180-phi,90,ec='r',ls='--',fc='none'))
    ax.add_patch(Wedge([0,ss],0.5,90,phi,ec='r',fc='none'))
    ax.add_patch(Wedge([0,ss],0.7,10,phi,ec='g',fc='none'))

    #for c in ['r','r--','g']:plt.plot(3,3,c)
    plt.plot([-1,0],[0.4,ss],'c')
    plt.plot([1,0],[0.4,ss],'c--')
    plt.plot([-1,1],[0,0],color='gray',lw=5)
    plt.legend(['angle of reflection','angle of incidence','saccade angle',
        'old trajectory','new trajectory','barrier'],ncol=2,loc=[-.1,0.65],fontsize=8)
    plt.plot([0,0],[ss,2],color='k')
    plt.plot(0.69,.155,'xg')
    plt.xlim([-1,1])
    plt.ylim([0,1.2])
    ax.set_aspect(1)
    ax.set_axis_off() 
    plt.savefig(FPATH+'expl.png',dpi=400,bbox_inches='tight',pad_inches=0)

def plotSacStats(suf='b1131',axs=[]):
    import pylab as plt
    from scipy.stats import scoreatpercentile as sap
    from matusplotlib import figure,subplot,subplotAnnotate
    a2clr={0.45:'c',1.57:'m',2.69:'blue',3.14:'k',
           11.57:'lime',12.69:'g',22.69:'r'}
    def plotSmps(suf,xshift=0,age=np.linspace(4,11,101),mspeed=40,lim=None,crc=False):
        '''mspeed = speed in deg/s'''
        cis=[[],[],[],[],[],[],[]]
        import os.path
        if not os.path.exists(DPATH+f'sm{suf}.wfit'):return cis
        meta=np.load(DPATH+f'meta{suf}.npy')
        meta[meta[:,1]==0,1]=np.pi
        with open(DPATH+f'sm{suf}.wfit','rb') as f: w=pickle.load(f)
        if 'rhat' in w.keys():
            print(f'{suf} max rhat:', np.max(w['rhat'][0,:-1]),w['nms'][0,np.argmax(w['rhat'][0,:-1])])
            if int(suf[1])==0: w['rhat'][:,w['nms'].tolist().index('bo')]=1
            #TODO assert(np.all(w['rhat'][0,:-1]<1.1))
        if int(suf[1]):print(f'{suf}, bo= ',sap(w['bo'],[50,2.5,97.5],axis=0))
        z=[]
        b0=w['b0'][:,:,:,NA];b1=w['b1'][:,:,:,NA];
        if 'mv' in w.keys(): 
            b2=w['mv'][:,:,:,NA]
            if suf[2]==2: b2=b2+w['bv'][:,:,:,NA]*age[NA,NA,NA,:]
        else: b2=0
        tmp=b0+age[NA,NA,NA,:]*b1+b2*mspeed
        clrs=[];
        for k in list(a2clr.keys()):#draw physical bounce trajs
            o=.1*(k/10)
            if int(suf[4])==1:plt.plot([o,o+np.cos(k%10)*15],[o,o+np.sin(k%10)*15], '--', c=a2clr[k])
        if int(suf[4])==0:
            plt.plot([0,-5],[0,0],color='grey')
            plt.plot(0,0,'o',color='grey')
        
        for k in range(tmp.shape[1]):#draw avg sac target
            y=tmp[:,k,:,:]
            a=10*meta[k,2]+np.round(meta[k,1],2)
            y-= np.array([xshift,0])[NA,:,NA]
            cis[list(a2clr.keys()).index(a)].append(y)
            if int(suf[3])==0:
                d4=np.linalg.norm(y[:,:,0],axis=1)
                d11=np.linalg.norm(y[:,:,-1],axis=1)
                print('11m-7m dist= ',a2clr[a],sap(d11-d4,[50,2.5,97.5]))
            y=np.median(y,0)
            print(k,a, (a%10)/np.pi*180,a2clr[a])
            plt.plot(y[0,:],y[1,:],c=a2clr[a])
            plt.plot(y[0,0],y[1,0],'x',c=a2clr[a])
            if crc: plt.plot(meta[k,3],meta[k,4],'o',c=a2clr[a])
            if lim is None:
                if int(suf[4])==1: 
                    plt.xlim([-10,10]);
                    plt.ylim([-4,14]) 
            else:plt.xlim(lim[0]);plt.ylim(lim[1]) 
            plt.gca().set_aspect(1)
        return cis
        
    def xy2dphi(x,y,trg=None,percentiles=[50,2.5,97.5]):
        ''' x,y - coordinates on horizontal and vertical axis
            trg - the center of the window for angular difference
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
    def computeDisplacement(x,y):
        phi=np.mod(360-np.arctan2(np.median(y),np.median(x))/np.pi*180,360)-180
        phi=np.arctan2(np.median(y),np.median(x))
        print('phi',phi)
        dst,r,rd=xy2dphi(x[:,NA],y[:,NA],trg=phi)
        #print(dst.shape);stop
        return dst[:,0]
 
    sv=len(axs)==0
    if sv: 
        fig, axs = plt.subplots(figsize=(12,5),nrows=2, ncols=4)
        axs=np.reshape(axs,-1)

    print(len(axs))
    with open(DPATH+f'sm{suf}.wfit','rb') as f: w=pickle.load(f)
    #TODO uncomment assert(np.all(w['rhat'][0,:-1]<1.1))
    meta=np.load(DPATH+f'meta{suf}.npy')
    meta[meta[:,0]==0,0]=np.pi
    for s in range(2):
        dd=[w['mv']*10,w['b1']][s]
        d=sap(dd,[50,2.5,97.5],axis=0)
        print(d.shape)
        #print(d[0,2,:],np.median(w['mv']*1000,0)[2,:]);bla
        for i in range(d.shape[1]):
            plt.sca(axs[2+s])
            c=a2clr[np.round(meta[i,0],2)]
            plt.plot(d[0,i,0],d[0,i,1],'.',color=c)
            #vs=[[d[0,i,0],d[1,i,1]],[d[2,i,0],d[0,i,1]],
            #    [d[0,i,0],d[2,i,1]], [d[1,i,0],d[0,i,1]]]
            #plt.gca().add_patch(plt.Polygon(vs,alpha=0.2,fill=True,fc=c,ec=c))
            plt.plot([d[0,i,0],d[0,i,0]],d[1:,i,1],alpha=.3,color=c)
            plt.plot(d[1:,i,0],[d[0,i,1],d[0,i,1]],alpha=.3,color=c)
            plt.sca(axs[6+s])
            if int(suf[4])==1: 
                dst=computeDisplacement(dd[:,i,0],dd[:,i,1])
            else: dst=sap(d[:,i,0],[50,2.5,97.5])
            #print(dst.shape);stop
            k=[3,0,1,2,4,5,6][i]
            plt.plot(k,dst[0],'.',color=c)
            plt.plot([k,k],dst[1:],color=c)
        plt.sca(axs[2+s])
        plt.grid(False)
        ax=plt.gca()
        ax.set_aspect(1)
        for k in list(a2clr.keys()):#draw physical bounce trajs
            o=2*(k/10)/1000
            plt.plot([o,o+np.cos(k%10)*5],[o,o+np.sin(k%10)*5], '--', c=a2clr[k])
        plt.xlim([-4,4])
        plt.ylim([-2,4])
        unt='$\\frac{\\mathrm{deg}}{'+['10 \\mathrm{deg}/s}$','\\mathrm{month}}$'][s]
        plt.xlabel(unt);
        plt.ylabel(unt);
        subplotAnnotate(loc='ne',nr=np.nan)
        plt.sca(axs[6+s])
        ax=plt.gca()
        #if s: plt.ylim([-.5,1.7])
        #else: plt.ylim([-.100,.500])
        ax.set_xticks([])
        #plt.ylim([-.200,.400])
        if int(suf[4])==0: plt.xlabel('Bounce type');
        plt.ylabel('Displacement from origin in '+unt);
        subplotAnnotate(loc='ne',nr=np.nan)
    
    for s in range(2):
        vel=[10,40][s]
        plt.sca(axs[s])
        cis=plotSmps(suf,xshift=0,lim=None,crc=0,mspeed=vel)
        plt.grid(False)
        #plt.xlabel('Degrees of visual angle');plt.ylabel('Degrees of visual angle')
        #plt.xlim([-10,10]);plt.ylim([-4,14]);
        subplotAnnotate(loc='ne',nr=np.nan)
    
        age=np.linspace(4,10,101)
        b1=0;xtcks=[]
        
        for i in range(len(cis)):
            d=np.array(cis[i])[0,:]
            if i<4:trg=list(a2clr.keys())[i]
            else: trg=[np.pi,np.pi,np.pi/2][i-4]
            clr=list(a2clr.values())[i]
            #print(i,trg,clr)
            plt.sca(axs[s+4])
            if int(suf[4])==0: 
                r=sap(d[:,0,:],[50,2.5,97.5],axis=0)
                k=[3,0,1,2,4,5,6][i]
                bbb=r.shape[1]//2
                plt.plot(k,r[0,bbb],'.',color=clr)
                plt.plot([k,k],r[1:,bbb],color=clr)
            else:
                dst,r,rd=xy2dphi(d[:,0,:],d[:,1,:],trg)
                r=180-r
                plt.plot([i-.5,i+.5],[180-trg/np.pi*180,180-trg/np.pi*180],':',c=clr)
            #plt.plot(i,r[0,r.shape[1]//2],'o',c=clr)
            #plt.plot([i,i],[r[2,r.shape[1]//2],r[1,r.shape[1]//2]],c=clr)
                gap=.2
                x=(age-4)/7*(1-gap)+i-.5+gap/2
                xtcks.extend([x[0],x[x.size//2],x[-1]])
                plt.plot(x,r[0,:],c=clr)
                u=r[2,:];l=r[1,:]
                xx=np.concatenate([x,x[::-1]])
                ci=np.concatenate([u,l[::-1]])
                plt.gca().add_patch(plt.Polygon(np.array([xx,ci]).T,
                            alpha=0.2,fill=True,fc=clr,ec=clr));
        ax=plt.gca()
        if int(suf[4])==1: 
            plt.ylim([-60,180])
            ax.set_yticks([-45,0,45,90,135,180])
            if s==0:plt.ylabel('Anticipated Angle in Degrees');
        else:
            tmp=np.max(np.abs(plt.ylim()))
            plt.ylim([-tmp,tmp])
            if s==0:plt.ylabel('XX Displacement along Circle\'s Trajectory');
            plt.xlabel('Bounce Type')
        ax.set_xticks([])
        plt.grid(True,axis='y')
        subplotAnnotate(loc='ne',nr=np.nan)
        
    if sv:plt.savefig(FPATH+suf+'.png',dpi=400,bbox_inches='tight',pad_inches=0)


def plotMain():
    import pylab as plt
    fig, discax = plt.subplots(figsize=(12,12),nrows=3, ncols=3)
    discax=np.reshape(discax,-1).tolist()
    fig, axs = plt.subplots(figsize=(12,9),nrows=3, ncols=4)
    from matusplotlib import formatAxes
    axs=np.reshape(axs,-1).tolist()
    formatAxes(axs)
    plotSacStats('b1131',axs=axs[:3]+[discax[0]]+axs[4:8]) 
    plotSacStats('b1130',axs=discax[:4]+axs[8:]) 
    plt.savefig(FPATH+'predsac.png',dpi=400,bbox_inches='tight',pad_inches=0)
    
def plotSacfrequency(axs=[]):
    sv=len(axs)==0
    if sv: 
        fig, axs = plt.subplots(figsize=(12,5),nrows=2, ncols=4)
        axs=np.reshape(axs,-1)
    plt.figure()
    G=np.load(DPATH+'G.npy')
    bs=np.linspace(-0.3,0.7,101);wdur=(bs[-1]-bs[0])/(bs.size-1)
    clrs=['k','b','m','c','g','lime','r']
    for t in range(7):
        plt.plot(bs[:-1]+wdur/2,G[:,t,3]/G[:,t,5],color=clrs[t])
    plt.grid(True)
    plt.xlabel('Time in sec with bounce at 0 sec')
    plt.ylabel('# saccades per sec')
    #subplotAnnotate(loc='ne',nr=np.nan)
    for t in range(7):
        r=1.96*np.sqrt(G[:,t,3]/G[:,t,5]*(1-G[:,t,3]/G[:,t,5])/G[:,t,5])
        print(np.nanmin(r),np.nanmax(r))   
    plt.savefig(FPATH+'sacFreq.png',dpi=400,bbox_inches='tight')  

def plotSupplement(compute=True):
    from matusplotlib import ndarray2latextable,figure
    import pylab as plt
    D=loadData()
    R=np.zeros((6,5),dtype=np.int32)
    for i in range(D[0].shape[0]):
        R[D[0][i,MCOND]+1,int((D[0][i,MCOH]-4)/3)+1]+=1
    R.T
    R[:,-1]=R.sum(1)
    R[:,0]=np.arange(6)
    R=R.astype(object)
    R[0,:]=np.array(['Group','Age 4.0-5.5 m.','Age 5.5-8.5 m.','Age 8.5-11.0 m.','Total'])
    ndarray2latextable(R.T,decim=0)
    
    figure()
    xi=np.load('xi.npy')
    a,b=np.histogram(xi,bins=np.sort(np.unique(xi)))
    N=410-np.unique(xi).size
    a,b=np.histogram(a.tolist()+[0]*N,bins=np.linspace(-5,80,18)+0.5)
    plt.bar(b[:-1]+4.5-2.5,a,width=5,ec='w')
    plt.gca().set_xticks(b[:-1]+4.5);
    plt.ylabel('Nr. of infants')
    plt.xlabel('Nr. of bounce saccades per infant')
    plt.savefig(FPATH+'suppHist.png',dpi=400,bbox_inches='tight',pad_inches=0)
    
    return 
    #TODO
    cs=[]#TODO move suplement method
    figure(figsize=[12,15])
    for i in [0,1,2,3,4]:
        for h in range(2):
            subplot(5,2,1+h+2*i)
            plt.grid(False)
            cis=plotSmps('b001n'+str(i),xshift=[0,-XSHIFT][h])
        cs.append(cis)
    plt.savefig(FPATH+'predsacBlock.png',dpi=400,bbox_inches='tight',pad_inches=0)
    
    clrs='rgbkc'
    for k in range(5):
        subplot(2,3,4)
        spd=[5,10,20,30,40][k]
        cis=plotSmps(suf,xshift=0,lim=None,crc=0,mspeed=spd)
        
        rds=[]
        for i in range(len(cis)):
            d=np.array(cis[i])[0,:]
            dst,r,rd=xy2dphi(d[:,0,:],d[:,1,:],trg)
            rd=(180-trg/np.pi*180)-(180-rd)
            rds.append(rd)
        subplot(2,3,6)
        m=np.abs(rds).mean(0).mean(0)
        u=sap(np.abs(rds).mean(0),97.5,axis=0)
        plt.plot(age,m,color=clrs[k])
        plt.plot(age, u,':'+clrs[k])
        print(spd,m[m.size//2],u[m.size//2])

def computeFreqAmp():
    import pylab as plt
    D=loadData()
    plt.figure(figsize=(16,9))
    bs=np.linspace(-0.3,0.7,101)
    wdur=(bs[-1]-bs[0])/(bs.size-1)
    G=np.zeros((len(bs)-1,7,7))
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
                G[:,t,j]+=a/wdur 
            for k in range(len(bs)-1):
                of=wdur/2+bs[k]
                a=np.logical_and(D[9][i][NA,tmp==t,0]+of>D[6][i][sel,NA,1],
                    D[9][i][NA,tmp==t,0]+of<D[6][i][sel,NA,2])    
                G[k,t,3]+=np.nansum(a/dur[:,NA]*wdur)/wdur#/a.sum(1)[:,NA])
                G[k,t,4]+=np.nansum(a*ampl[:,NA])
                G[k,t,6]+=a.sum()
            G[:,t,5]+=(tmp==t).sum() 
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
    np.save(DPATH+'G',G)
    plt.savefig(FPATH+'suppSacTime.png',dpi=400,bbox_inches='tight',pad_inches=0)
    
def printStats():
    print('''print sample statistics''')
    from matusplotlib import ndarray2latextable,figure
    import pylab as plt
    R=np.zeros((10,7),dtype=object)
    R[2:,1:]=np.load('data/metab1131.npy')[:,5:][[-1,0,3,2,1,4,5,6],:]
    R[1,1:]=np.load('data/metal1131.npy')[5:]
    R[0,:]=np.array([' ','Nr. Saccades','Mean Velocity','Std Velocity','Nr. Participants','Mean Age','Std Age'])
    R[:,0]=np.array([' ','LM Saccades','Bounce Saccades','  Black','  Blue','  Magenta','  Cyan','  Lime','  Green','  Red'])
    ndarray2latextable(R,decim=[0,0,1,1,0,1,1])
    stop
    print('''print statistics of linear-motion saccades''')
    from scipy.stats import scoreatpercentile as sap
    with open(DPATH+f'sml01.wfit','rb') as f: w=pickle.load(f)
    w['rhat'][:,w['nms'].tolist().index('bo')]=1
    assert(np.all(w['rhat'][0,:-1]<1.1))

    def sacTar(a=7,v=20,percs=[50,2.5,97.5]): 
        res= w['b0']+w['b1']*a+(w['bv']*a+w['mv'])*v
        res=np.round(sap(-res[:,0],percs,axis=0),2).T
        print(f'age: {a}, vel: {v}, gaze: {res}')
    sacTar(a=4)
    sacTar(a=11)
    sacTar(v=5)
    sacTar(v=40)
    print(np.round(sap(-w['mv'][:,0]-w['bv'][:,0]*7,[50,2.5,97.5])*1000,1))
    print(np.round(sap(-w['bv'][:,0]*(11-4)*1000,[50,2.5,97.5]),1))
    sacTar(v=0)
    print('''print amplitude stats''')
    for suf in ['l02','b012']:
        print(['linear motion','bounce'][suf[0]=='b'])
        with open(DPATH+f'sm{suf}.wfit','rb') as f: w=pickle.load(f)
        print(f'rhat', np.nanmax(w['rhat'][0,:-1]),w['nms'][np.argmax(w['rhat'][0,:-1])])
        if suf[0]=='b': bv=0
        else: bv=w['bv'][:,0]
        def sacTar(a=7,v=20,percs=[50,2.5,97.5]): 
            res= w['b0'][:,0]+w['b1'][:,0]*a+(w['mv'][:,0]+bv*a)*v
            res=np.round(sap(res,percs,axis=0),2).T
            print(f'age: {a}, vel: {v}, ampl: {res}')
        sacTar(a=7,v=20)
        print(np.round(sap(w['b1'][:,0],[50,2.5,97.5]),2))
        print(np.round(sap(w['mv'][:,0]+bv*7,[50,2.5,97.5]),2))
        #additional stats for discussion section
        sacTar(a=4,v=24)
        sacTar(a=5,v=20)

def plotTrajectory():
    wx=37.3/2;wy=21/2
    def getBounceLocs(bounceType=2,phi=257,vel=7,initPos=[0,0]):
        pos=np.array(initPos);bt=bounceType;maxdur=10
        out=[pos];phis=[phi]
        dur=0
        k=0
        while True:
            #print(k,pos,phi)
            if phi>90 and phi<270: inv=-1
            else: inv=1

            bpx=np.array([inv*wx,np.tan(phi*np.pi/180)*(inv*wx-out[-1][0])+out[-1][1]])
            #if k==6:stop
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
            #print(i,j,phis)
            for p in range(1,out.shape[0]-1):#np.unique(out[1:-1,:],axis=0).tolist():
                if D[i,j]==0 or D[i,j]==90:clr='k'
                elif D[i,j]==1045:clr='lime'
                elif D[i,j]==1077:clr='g'
                elif D[i,j]>2000:clr='r'
                #elif D[i,j]==45:clr='m'
                elif D[i,j]<1000:
                    if (phis[p]-phis[p-1]+360)%90==0:clr='m'
                    elif (phis[p]-phis[p-1]+360)%360==26:clr='c'
                    #elif (phis[p]-phis[p-1]+360)%360==154:clr='b'
                    else:clr='b'
                    
                c=plt.Circle(out[p,:],2,ec=clr,fc=clr,zorder=2)
                ax.add_patch(c)
                #if np.abs(p[0])==wx: ha=['left','right'][int(p[0]>0)]
                #else: ha='center'
                #if np.abs(p[1])==wy: va=['bottom','top'][int(p[1]>0)]
                #else: va='center'
                #plt.text(p[0],p[1],str(n),ha=ha,va=va,zorder=3)
            plt.arrow(out[0,0],out[0,1],2*np.cos(phis[0]/180*np.pi),
                      2*np.sin(phis[0]/180*np.pi),width=1,color='k')
            #ax.add_patch(a)
            if i==4:plt.title('Block '+str(j+1))
            if j==0:plt.ylabel('Group '+str(i+1))
    plt.tight_layout()
    plt.savefig(FPATH+'trajectory.png',dpi=400,bbox='tight',pad_inches=0)

def hypothesisTest(fn='b1131',ML=False,BALANCED=True):
    from scipy.stats import norm
    with open(f'data/sm{fn}.wfit','rb') as f: w=pickle.load(f)
    dat=np.load(f'data/dat{fn}.npy',allow_pickle=True)
    dat=dict(dat.tolist())
    na=np.newaxis
    vels=np.linspace(0,40,81)
    ages=np.linspace(4,10,61)
    x=np.linspace(0,20,101)
    p=np.zeros((ages.size,vels.size,5))
    angls=[[180]*7,[0]*7,[90]*7,[180,26,90,154,90,154,154],[180,26,90,154,180,180,90]]
    for h in range(len(angls)):
        for k in range(len(angls[0])):
            #print(h,k)
            angl=angls[h][k]/180*np.pi
            xx=np.array([np.cos(angl)*x,np.sin(angl)*x]) 
            if not BALANCED:
                sel=dat['xc']==k+1
                mu=dat['y'][sel,:]
                if ML:
                    sig= np.median(w['sm'],0).T[dat['xi'][sel]-1,:]*dat['xv'][sel,na]
                    tmp=np.max(norm.logpdf(xx[:,:,na],mu.T[:,na,:],sig.T[:,na,:]).sum(0),0)
                    p[0,0,h]+=tmp.sum()
                else:
                    sig= w['sm'][:,:,dat['xi'][sel]-1]*dat['xv'][na,na,sel]
                    tmp=norm.pdf(xx.T[:,na,:,na],mu.T[na,na,:,:],sig[na,:,:,:])
                    p[0,0,h]+=np.log(tmp.mean((0,1))).sum()
                continue
            for i in range(ages.size):
                for j in range(vels.size):
                    age=ages[i]
                    vel=vels[j]
                    if ML:
                        mu=np.median(w['b0'],0)[k,:]+np.median(w['b1'],0)[k,:]*age +np.median(w['mv'],0)[k,:]*vel
                        sig= np.sqrt(np.square(vel*np.median(w['gsm'][:,:,0]/w['gsm'][:,:,1],0))+np.square(np.median(w['sn'],0)))
                        #TODO use median xx
                        p[i,j,h]+=np.product(norm.pdf(xx,mu[:,na],sig[:,na]).mean(1))/7
                    else:
                        S=w['b0'].shape[0]
                        for s in range(S):
                            mu=w['b0'][s,k,:]+w['b1'][s,k,:]*age +w['mv'][s,k,:]*vel
                            sig= np.sqrt(np.square(vel*w['gsm'][s,:,0]/w['gsm'][s,:,1])+np.square(w['sn'][s,:]))
                            p[i,j,h]+=np.product(norm.pdf(xx,mu[:,na],sig[:,na]).mean(1))/7
                        #p[i,j,h]/=S does not affect results
    if not BALANCED: 
        p=p[0,0,:]
        p=p-np.max(p)
        p=np.exp(p)/np.exp(p).sum()
    np.save(DPATH+f'ht{int(ML)}{int(BALANCED)}',p)

if __name__=='__main__':
    computeSacStats(bounceSac=0,pVelM=1,pVelS=3,yType=1,pCenter=1,ssmm=10);stop
    plotSacStats('b1130')
    plotMain();stop
    if False:
        # loading and preprocessing
        vpinfo=getMetadata(showFigure=False)
        checkFiles(vpinfo)
        extractFromDataFiles()
        accuracyCorrection(method='partial pooling',applyCor=True,exclude=False)
        #accuracyCorrection(method='none',applyCor=True)
        changeUnits()
        extractSaccades()
        # statistical analyses
        rotateSaccades();
        # saccade amplitude
        #computeSacStats(bounceSac=False,yType=2,pVelM=1,pVelS=0,pCenter=0,ssmm=1)
        #computeSacStats(bounceSac=True, yType=2,pVelM=1,pVelS=0,pCenter=0,ssmm=1)  
    #printStats()
    plotSacStats('b1231');stop
    computeSacStats(bounceSac=1,pVelM=1,pVelS=3,yType=1,pCenter=1,ssmm=10)
    computeSacStats(bounceSac=0,pVelM=2,pVelS=3,yType=1,pCenter=1,ssmm=10)#sml1121
    stop;
    for i in range(3):
        computeSacStats(list(range(i*5,(i+1)*5)),suf='n'+str(i),poolCond=False,
            bounceSac=1,pVelM=0,pVelS=0,yType=1,pCenter=0,ssmm=10,minSacNr=2, compileStan=i==0)#smb0001n
    if False:
        rotateSaccades(CSDIST=10);
        computeSacStats(bounceSac=1,pVelM=1,pVelS=2,yType=1,pCenter=1,ssmm=10, suf='csdist10')#sml1121csdist10
        rotateSaccades(SACS=[-.2,0]);
        computeSacStats(bounceSac=1,pVelM=1,pVelS=2,yType=1,pCenter=1,ssmm=10, suf='sacs200')#sml1121sacs200
        #printStats()
        computeFreqAmp()
        hypothesisTest(fn='b1131',ML=True,BALANCED=False)
        hypothesisTest(fn='b1131',ML=False,BALANCED=False)
        #hypothesisTest(fn='b1131',ML=False,BALANCED=True)
        #hypothesisTest(fn='b1131',ML=True,BALANCED=True)
        # results presentation
        #plotExpl()
        #plotTrajectory() 
        #printStats()
        plotSacStats()
        #plotSupplement();
        
    
    
