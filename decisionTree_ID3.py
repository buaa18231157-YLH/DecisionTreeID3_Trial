import numpy as np
import csv
import copy
import random
from math import log
featureLabel=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']#四个特征
class data_unit:#定义数据单元
    def __init__(self,fea=None,label=None):
        self.fea=fea
        self.label=label
class Tnode:#定义节点单元
    def __init__(self,label=None,pnext=None):
        #该节点可用来决策的特征集
        self.label=label#该节点的类
        self.cls=[]#用来决策特征以及特征取值
        self.next=pnext
def data_Init():#数据预处理，将数据值分成三个区间
    data_backup=[]
    feaValue=[]
    with open('E:\模式识别与智能系统技术\机器学习\决策树\iris.csv','r') as file:
        reader = csv.reader(file)
        for rows in reader:
            feature_data=rows[1:5] #中间四个是数据，最后一个是种类
            feature_data_temp=[]
            for each in feature_data:
                feature_data_temp.append(float(each))
            data_backup.append(data_unit(feature_data_temp,rows[-1]))
    for each_data_unit in data_backup:
        feaValue.append(each_data_unit.fea)
    m = np.array(feaValue).T #转置,让每个特征的数据排成一行
    maxValue=[]
    minValue=[]
    subValue=[]
    for i in range(0,len(m)): #150个
        maxValue.append(max(m[i]))#得到四个最大的特征的值，1*4矩阵
        minValue.append(min(m[i]))
        subValue.append((max(m[i])-min(m[i]))/3)#强制分为3个区间
    for data_each in data_backup:
        for i in range(0,len(data_each.fea)):#遍历特征每一个值
            for j in range(0,4):#遍历每一个区间
                a=minValue[i]+subValue[i]*j-0.001
                b=minValue[i]+subValue[i]*(j+1)
                if(data_each.fea[i]>a and data_each.fea[i]<b):
                    data_each.fea[i]=j
                    break
    branch=[[0,1,2],[0,1,2],[0,1,2],[0,1,2]] #三分类法
    indexTrain = random.sample(range(0,len(data_backup)),round(len(data_backup)*0.7))
    indexTest = list(set(range(len(data_backup)))-set(indexTrain))
    dataTrain=[]
    dataTest=[]
    for i in indexTrain:
        dataTrain.append(data_backup[i])
    for i in indexTest:
        dataTest.append(data_backup[i])
    return dataTrain,dataTest,branch
def calcEntropy(dataSample):#计算集合的熵
    num=len(dataSample)
    labelCounts={}
    for it in dataSample:
        if it.label not in labelCounts.keys():
            labelCounts[it.label]=0
        labelCounts[it.label] += 1
    #计算熵
    entropy=0
    for key in labelCounts:
        prob = float(labelCounts[key])/num
        if(prob==0):
            continue
        entropy=entropy - prob*log(prob,2)
    return entropy
def getLabel(dataSample):#取得该集合中数量最多的类
    num=len(dataSample)
    labelCounts={}
    for it in dataSample:
        if it.label not in labelCounts.keys():
            labelCounts[it.label]=0
            labelCounts[it.label] += 1
    return max(zip(labelCounts.values(),labelCounts.keys()))[1]
 
def decision(node,dataD,feature):#生成树
    if(dataD==[]):
        node0=Tnode()
        node0.label="None"
        return node0
    if(feature==[]):#决策可用特征已经用完
        node1=Tnode()
        node1.label=getLabel(dataD)
        return node1
    stad=dataD[0].label
    num1=len(dataD)
    num2=len([x for x in dataD if x.label==stad])
    if(num1==num2):#dataD全为一类
        node2 = Tnode()
        node2.label = dataD[0].label
        return node2
    dataNext=[]#存放决策分类后的数据
    node.next=[]#
    Etpy=[]#存放不同特征的熵
    #选取dataD信息增益最大的特征
    EtpyPre=calcEntropy(dataD)#初始熵值
    for i in feature:#遍历每一个决策可用特征
        EtpyPre_=EtpyPre
        for j in featClass[i]:#遍历该特征所有取值
            mid=[x for x in dataD if x.fea[i]==j]
            #这里必须要新建对象，否则是对同一片内存空间操作
            EtpyPre_=EtpyPre_-len(mid)/len(dataD)*calcEntropy(mid)
        Etpy.append(EtpyPre_)
    maxIndex=Etpy.index(max(Etpy))#得到信息增益最大特征在可用特征集中所对应的索引
    FEA=feature[maxIndex]#转换为特征
    node.cls=[]
    node.cls.append(FEA)
    del feature[maxIndex]#从待选特征中去除该特征
    featureNext=copy.deepcopy(feature)#深拷贝对象
    for i in featClass[FEA]:
        mid=[x for x in dataD if x.fea[FEA]==i]
        dataNext.append(mid)#将dataD按决策特征的值分类   
        node.next.append(Tnode)
    for i in range(0,len(node.next)):
        node.next[i]=decision(Tnode(),dataNext[i],featureNext)
        node.next[i].cls=[]
        node.next[i].cls.append(FEA)#该子节点决策所用特征
        node.next[i].cls.append(featClass[FEA][i])#该子节点决策所用特征取值
    return node

def classify(data,TREE):#递归分类
    if(TREE.next==None):
        if(TREE.label==data.label):
            return 1
        else:
            return 0
    a=TREE.next[0].cls[0]#决策节点的特征
    b=data.fea[a]#该样本在该特征中的取值
    if(b>=len(TREE.next)):#如果出现了训练集中没有考虑到的样本，以分错处理
        return 0
    judge=classify(data,TREE.next[b])
    return judge

def draw(nod):#画字典形式的树
    treeMid={}
    if(nod.next==None):#说明是叶节点
        treeMid[0]=nod.label
        return treeMid
    for i in range(0,len(nod.next)):
        treeMid[i]=draw(nod.next[i])
    return treeMid    
def correct_rate(dataT,TREE):#准确率
    count=0
    for i in dataT:
        count = count + classify(i,TREE)
    rate=count/len(dataT)
    return rate
dataTrain,dataTest,featClass=data_Init()
root=Tnode()
jkl=decision(root,dataTrain,[0,1,2,3])
crtRate=correct_rate(dataTest,jkl)
print("决策树结构：",draw(jkl))
print('正确率:{:.2f}%'.format(crtRate*100))

