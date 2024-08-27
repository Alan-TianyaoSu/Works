from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pickle
import pandas as pd
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




# a=torch.load('D:/bishe/Graph-WaveNet-master/garage31/aptonly/_exp1_best_0.11.pth')
# m1=np.random.rand(125,10)
# m2=np.random.rand(10,125)

# for i in range(0,125):
#     for j in range(0,10):
#         m1[i][j]=a['nodevec1'][i][j].item()

# for i in range(0,10):
#     for j in range(0,125):
#         m2[i][j]=a['nodevec2'][i][j].item()       
# m=np.dot(m1,m2)
#seaborn.heatmap(m,cmap="")

#关系矩阵
a=torch.load('D:/bishe/Graph-WaveNet-master/winter71/doubletransition/_exp1_best_0.1.pth')
m1=np.random.rand(125,10)
m2=np.random.rand(10,125)

st=4
for i in range(0,125):
    for j in range(0,10):
        m1[i][j]=a['nodevec1'][i][j].item()

for i in range(0,10):
    for j in range(0,125):
        m2[i][j]=a['nodevec2'][i][j].item()       
m=np.dot(m1,m2)
p=np.random.rand(25,25)
for i in range(st,125,5):
    for j in range(st,125,5):
        p[int(i/5)][int(j/5)]=m[i][j]
p=p*(1e39)

#seaborn.heatmap(p,cmap="YlGnBu")





#精度
f=open('D:/bishe/Graph-WaveNet-master/winter71/doubletransition/data/predict.pkl','rb')
preds = pickle.load(f, encoding='latin1')
print(preds.shape)
f.close()
f=open('D:/bishe/Graph-WaveNet-master/winter71/doubletransition/data/real.pkl','rb')
labels = pickle.load(f, encoding='latin1')
print(labels.shape)
f.close()

preds=torch.tensor(preds)
labels=torch.tensor(labels)
loss = torch.abs(preds-labels)
loss=np.array(loss)
loss=np.mean(loss,axis=0)
loss=loss[0]
loss=np.mean(loss,axis=1)

# a=np.random.rand(16,125)
# for i in range(0,16):
#     for j in range(0,125):
#         a[i][j]=loss[i][0][j][0]

# loss=np.average(a,axis=0)




c=pd.read_csv('D:/bishe/Graph-WaveNet-master/tem.csv')
col=c.columns
col=col[1:]
da=[]
for i in range(st,125,5):
    t=[]
    t.append(float(col[i].split(':')[0]))
    t.append(float(col[i].split(':')[1]))
    t.append(loss[i])
    da.append(t)

matrix='Doubletransition'
deep='199.79'
dep='5'

def point(da):
    fig= plt.figure(figsize=(10, 10))
    ax=fig.add_subplot(1,1,1)
    m = Basemap(projection='lcc', resolution=None,
                width=2E6, height=2E6, 
                lat_0=13, lon_0=110,)
    #这里投影模式llc是圆锥投影
    m.etopo(scale=0.5, alpha=0.5)
    #其实也就加了这一句...scale原理同上文bluemarble中的scale，alpha是透明度图层
    # Map (long, lat) to (x, y) for plotting
    
    for item in da:
        x,y=m(item[0],item[1])
        if item[2]>=0.15:
            plt.plot(x, y, 'o', markersize=12,color='red')
        elif item[2]>=0.14:
            plt.plot(x, y, 'o', markersize=12,color='orange')
        elif item[2]>=0.13:
            plt.plot(x, y, 'o', markersize=12,color='green')
        elif item[2]>=0.12:
            plt.plot(x, y, 'o', markersize=12,color='blue')
        elif item[2]>=0.11:
            plt.plot(x, y, 'o', markersize=12,color='purple')
        elif item[2]>=0.10:
            plt.plot(x, y, 'o', markersize=12,color='pink')
        else:
            plt.plot(x, y, 'o', markersize=12,color='black')
    plt.title('MATRIX:'+matrix+' DEEP:'+deep,fontsize=30)
    plt.legend(loc=0)
    plt.savefig('D:/bishe/Graph-WaveNet-master/winter71/doubletransition/mat/deep'+dep+'/point.png')
    plt.show()
    


    

def init(da,p):
    fig= plt.figure(figsize=(10, 10))
    ax=fig.add_subplot(1,1,1)
    m = Basemap(projection='lcc', resolution=None,
                width=2E6, height=2E6, 
                lat_0=13, lon_0=110,)
    #这里投影模式llc是圆锥投影
    m.etopo(scale=0.5, alpha=0.5)
    #其实也就加了这一句...scale原理同上文bluemarble中的scale，alpha是透明度图层
    # Map (long, lat) to (x, y) for plotting
    
    for item in da:
        x,y=m(item[0],item[1])
        if item[2]>=0.15:
            plt.plot(x, y, 'o', markersize=12,color='red')
        elif item[2]>=0.14:
            plt.plot(x, y, 'o', markersize=12,color='orange')
        elif item[2]>=0.13:
            plt.plot(x, y, 'o', markersize=12,color='green')
        elif item[2]>=0.12:
            plt.plot(x, y, 'o', markersize=12,color='blue')
        elif item[2]>=0.11:
            plt.plot(x, y, 'o', markersize=12,color='purple')
        elif item[2]>=0.10:
            plt.plot(x, y, 'o', markersize=12,color='pink')
        else:
            plt.plot(x, y, 'o', markersize=12,color='black')
            
    for i in range(0,25):
        x1,y1=m(da[i][0],da[i][1])
        for j in range(0,25):
            x2,y2=m(da[j][0],da[j][1])
            if abs(p[i][j]<=1):
                if p[i][j]>0:
                    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0,0,0.2], shrink=0.05))
                else:
                    ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0,0,0.2], shrink=0.05))
            if abs(p[i][j]<=1.5  and p[i][j]>1):
                if p[i][j]>0:
                    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0,0,0.5], shrink=0.05))
                else:
                    ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0,0,0.5], shrink=0.05))
            if abs(p[i][j]>1.5 and p[i][j]<=2):
                if p[i][j]>0:
                    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0.5,0,0.5], shrink=0.05))
                else:
                    ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0.5,0,0.5], shrink=0.05))
            if abs(p[i][j]>2 and p[i][j]<=2.5):
                if p[i][j]>0:
                    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0.6,0.2,0], shrink=0.05))
                else:
                    ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0.6,0.2,0], shrink=0.05))
            if abs(p[i][j]>2.5 and p[i][j]<=3):
                if p[i][j]>0:
                    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0,0.6,0.3], shrink=0.05))
                else:
                    ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0,0.6,0.3], shrink=0.05))
            if abs(p[i][j]>3 and p[i][j]<=4.5):
                if p[i][j]>0:
                    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0,0.6,0.7], shrink=0.05))
                else:
                    ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0,0.6,0.7], shrink=0.05))
            if abs(p[i][j]>4.5 and p[i][j]<=5.5):
                if p[i][j]>0:
                    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0,0.6,0.9], shrink=0.05))
                else:
                    ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0,0.6,0.9], shrink=0.05))
            if abs(p[i][j]>5.5):
                if p[i][j]>0:
                    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0.5,0,1], shrink=0.05))
                else:
                    ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0.5,0,1], shrink=0.05))
    
    plt.title('MATRIX:'+matrix+' DEEP:'+deep,fontsize=30)
    plt.legend(loc=0)
    plt.savefig('D:/bishe/Graph-WaveNet-master/winter71/doubletransition/mat/deep'+dep+'/com.png')
    plt.show()
        
    
    
    
        
    
def test(case,da,p):
    fig= plt.figure(figsize=(10, 10))
    ax=fig.add_subplot(1,1,1)
    m = Basemap(projection='lcc', resolution=None,
                width=2E6, height=2E6, 
                lat_0=13, lon_0=110,)
    #这里投影模式llc是圆锥投影
    m.etopo(scale=0.5, alpha=0.5)
    #其实也就加了这一句...scale原理同上文bluemarble中的scale，alpha是透明度图层
    # Map (long, lat) to (x, y) for plotting
    
    for item in da:
        x,y=m(item[0],item[1])
        if item[2]>=0.15:
            plt.plot(x, y, 'o', markersize=12,color='red')
        elif item[2]>=0.14:
            plt.plot(x, y, 'o', markersize=12,color='orange')
        elif item[2]>=0.13:
            plt.plot(x, y, 'o', markersize=12,color='green')
        elif item[2]>=0.12:
            plt.plot(x, y, 'o', markersize=12,color='blue')
        elif item[2]>=0.11:
            plt.plot(x, y, 'o', markersize=12,color='purple')
        elif item[2]>=0.10:
            plt.plot(x, y, 'o', markersize=12,color='pink')
        else:
            plt.plot(x, y, 'o', markersize=12,color='black')
            
    if case==1:
        c=0
        for i in range(0,25):
            x1,y1=m(da[i][0],da[i][1])
            for j in range(0,25):
                x2,y2=m(da[j][0],da[j][1])
                if abs(p[i][j]<=1):
                    if p[i][j]>0:
                        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0,0,0.2], shrink=0.05))
                        if y2<y1:
                            c=c+1
                    else:
                        if y1<y2:
                            c=c+1
                        ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0,0,0.2], shrink=0.05))
        print('down1:',c)
        count=0
        for i in range(0,25):
            for j in range(0,25):
                if abs(p[i][j]<=1):
                    count=count+1
        print('case1:',count)
    elif case==2:
        c=0
        for i in range(0,25):
            x1,y1=m(da[i][0],da[i][1])
            for j in range(0,25):
                x2,y2=m(da[j][0],da[j][1])
                if abs(p[i][j]<=1.5  and p[i][j]>1):
                    if p[i][j]>0:
                        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0,0,0.5], shrink=0.05))
                        if y2<y1:
                            c=c+1
                    else:
                        ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0,0,0.5], shrink=0.05))
                        if y1<y2:
                            c=c+1
        print('down2:',c)
        count=0
        for i in range(0,25):
            for j in range(0,25):
                if abs(p[i][j]<=1.5  and p[i][j]>1):
                    count=count+1
        print('case2:',count)
    elif case==3:
        c=0
        for i in range(0,25):
            x1,y1=m(da[i][0],da[i][1])
            for j in range(0,25):
                x2,y2=m(da[j][0],da[j][1])
                if abs(p[i][j]>1.5 and p[i][j]<=2):
                    if p[i][j]>0:
                        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0.5,0,0.5], shrink=0.05))
                        if y2<y1:
                            c=c+1
                    else:
                        ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0.5,0,0.5], shrink=0.05))
                        if y1<y2:
                            c=c+1
        print('down3:',c)
        count=0
        for i in range(0,25):
            for j in range(0,25):
                if p[i][j]>1.5 and p[i][j]<=2:
                    count=count+1
        print('case3:',count)
    elif case==4:
        c=0
        for i in range(0,25):
            x1,y1=m(da[i][0],da[i][1])
            for j in range(0,25):
                x2,y2=m(da[j][0],da[j][1])
                if abs(p[i][j]>2 and p[i][j]<=2.5):
                    if p[i][j]>0:
                        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0.6,0.2,0], shrink=0.05))
                        if y2<y1:
                            c=c+1
                    else:
                        ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0.6,0.2,0], shrink=0.05))
                        if y1<y2:
                            c=c+1
        print('down4:',c)
        count=0
        for i in range(0,25):
            for j in range(0,25):
                if abs(p[i][j]>2 and p[i][j]<=2.5):
                    count=count+1
        print('case4:',count)
    elif case==5:
        c=0
        for i in range(0,25):
            x1,y1=m(da[i][0],da[i][1])
            for j in range(0,25):
                x2,y2=m(da[j][0],da[j][1])
                if abs(p[i][j]>2.5 and p[i][j]<=3):
                    if p[i][j]>0:
                        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0,0.6,0.3], shrink=0.05))
                        if y2<y1:
                            c=c+1
                    else:
                        ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0,0.6,0.3], shrink=0.05))
                        if y1<y2:
                            c=c+1
        print('down5:',c)
        count=0
        for i in range(0,25):
            for j in range(0,25):
                if abs(p[i][j]>2.5 and p[i][j]<=3):
                    count=count+1
        print('case5:',count)
    elif case==6:
        c=0
        for i in range(0,25):
            x1,y1=m(da[i][0],da[i][1])
            for j in range(0,25):
                x2,y2=m(da[j][0],da[j][1])
                if abs(p[i][j]>3 and p[i][j]<=4.5):
                    if p[i][j]>0:
                        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0,0.6,0.7], shrink=0.05))
                        if y2<y1:
                            c=c+1
                    else:
                        ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0,0.6,0.7], shrink=0.05))
                        if y1<y2:
                            c=c+1
        print('down6:',c)
        count=0
        for i in range(0,25):
            for j in range(0,25):
                if abs(p[i][j]>3 and p[i][j]<=4.5):
                    count=count+1
        print('case6:',count)
    elif case==7:
        c=0
        for i in range(0,25):
            x1,y1=m(da[i][0],da[i][1])
            for j in range(0,25):
                x2,y2=m(da[j][0],da[j][1])
                if abs(p[i][j]>4.5 and p[i][j]<=5.5):
                    if p[i][j]>0:
                        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0,0.6,0.9], shrink=0.05))
                        if y2<y1:
                            c=c+1
                    else:
                        ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0,0.6,0.9], shrink=0.05))
                        if y1<y2:
                            c=c+1
        print('down7:',c)
        count=0
        for i in range(0,25):
            for j in range(0,25):
                if abs(p[i][j]>4.5 and p[i][j]<=5.5):
                    count=count+1
        print('case7:',count)
    elif case==8:
        c=0
        for i in range(0,25):
            x1,y1=m(da[i][0],da[i][1])
            for j in range(0,25):
                x2,y2=m(da[j][0],da[j][1])
                if abs(p[i][j]>5.5):
                    if p[i][j]>0:
                        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(facecolor=[0.5,0,1], shrink=0.05))
                        if y2<y1:
                            c=c+1
                    else:
                        ax.annotate('',xy=(x1,y1),xytext=(x2,y2),arrowprops=dict(facecolor=[0.5,0,1], shrink=0.05))
                        if y1<y2:
                            c=c+1
        print('down8:',c)
        count=0
        for i in range(0,25):
            for j in range(0,25):
                if abs(p[i][j]>5.5):
                    count=count+1
        print('case8:',count)
    
    plt.title('MATRIX:'+matrix+' DEEP:'+deep+' rel pair:%d'%(count),fontsize=25)
    plt.legend(loc=0)
    plt.savefig('D:/bishe/Graph-WaveNet-master/winter71/doubletransition/mat/deep'+dep+'/case'+str(case)+'.png')
    plt.show()
    
    
    
    
    


#-----------------------------------------------------------------
point(da)

init(da,p)

for i in range(1,9):
    test(i,da,p)
