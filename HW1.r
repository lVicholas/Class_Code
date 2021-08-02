# Homework 1

# P2
load("~/_Importance_/TextbooksClassesPapers/StatLearning/R/HW1P2.dat")
library(class)
library(e1071)

x1 = data$x1
x2 = data$x2
xrandom = data$xrandom
y = data$y

train.x1 = data$x1[1:500]
train.x2 = data$x2[1:500]
train.y = y[1:500]

test.x1  = data$x1[501:1000]
test.x2  = data$x2[501:1000]
test.y  = y[501:1000]

prob = function(x1, x2){
  return (pnorm(.5*x1 - .4*x2))
}

simpleClassify = function(x1,x2){
  y=c()
  for(i in 1:length(x1))
    y[i] = as.numeric(prob(x1[i],x2[i]) >= .5)
  return (y)
}

trainYhat = simpleClassify(train.x1,train.x2)
testYhat = simpleClassify(test.x1, test.x2)

trainError = mean(trainYhat != train.y)
testError = mean(testYhat != test.y)

set.seed(777)
knn3 = knn(train=cbind(train.x1,train.x2), test=cbind(test.x1,test.x2), k=3, cl=train.y)
knnPred = as.numeric(knn3) - 1
mean(knnPred != test.y)

set.seed(777)
rates=c()
m = c(1,10)
for(i in 1:100){
  
  mod=knn(train=cbind(train.x1,train.x2), 
          test=cbind(test.x1,test.x2), k=i, cl=train.y)
  pre=as.numeric(mod)-1
  rates[i] = mean(pre != test.y)
  
}
ks = which(rates %in% sort(rates)[1:5]); ks
rates[ks]
plot(rates, xlab="k", ylab="Test Error Rate")

conf.train=cbind(train.x1,train.x2,xrandom[1:500,])
conf.test=cbind(test.x1,test.x2,xrandom[501:1000,])

knnConf = knn(train=conf.train,test=conf.test,k=40,cl=train.y)
mean(as.numeric(knnConf)-1 != test.y)

# P3
library(ISLR)
data("Smarket")

y2002 = rep(c(0,1,0),times=c(242,252,252*3))
y2003 = rep(c(0,1,0),times=c(242+252,252,252*2))
y2004 = rep(c(0,1,0),times=c(242+252*2,252,252))
y2005 = rep(c(0,1),times=c(242+252*3,252))

lr.mod=lm(Today~Lag1+Lag2+Lag3+
       Lag4+Lag5+Volume+y2002+y2003+y2004+y2005, 
       data=Smarket)
lr2=lm(Today~poly(Lag1, 3)+Lag2+Lag3+
         Lag4+Lag5+Volume+y2002+y2003+
         y2004+y2005, data=Smarket)
summary(lr2)

Smarket$Direction = (as.numeric(Smarket$Direction))-1
set.seed(777)
sample=sample(seq_len(nrow(Smarket)), size=floor(.5*nrow(Smarket)))
train.smarket=Smarket[sample,]
test.smarket=Smarket[-sample,]

min = c(1,10)
for (i in 1:100){
  
  mod=knn(train=train.smarket[,1:8], test=test.smarket[,1:8], 
          k=i, cl=train.smarket[,9])
  rate=mean((as.numeric(mod)-1) != test.smarket[,9])
  
  if(rate<min[2])
    min=c(i,rate)
  
}
min