# HW3

library(class)
library(caret)
library(e1071)
library(MASS)

#P1
d = read.table("Crabs.dat.txt", header = T)
d$y=as.factor(d$y)
d$color=as.factor(d$color)
d$spine=as.factor(d$spine)

# (ii)
set.seed(1.618)
t=sort(sample(1:nrow(d), size=floor(.5*nrow(d))))
train=d[t,]
valid=d[-t,]

G=10^(-2:2)
C=c(0.0001,0.001,0.01,0.1,1,2,3,5,10,25,50,100)

cost=c()
gamma=c()
trainError=c()
testError=c()

# Fit SVMs with radial kernel
for(i in 1:length(G)) {
  
  tune = tune(svm, y~., data=train, kernel="radial",
            ranges=list(cost=C, gamma=c(G[i])))
  cost[i]=tune$best.model$cost
  gamma[i]=tune$best.model$gamma
  trainError[i]=tune$best.performance
  testError[i]=mean(predict(
    tune$best.model, newdata = valid[,-1]) != valid[,1])
  
}

ii = data.frame(Gamma=gamma, Cost=cost, 
                Train=round(trainError,3), 
                Test=round(testError,3))
ii

# (iii)
set.seed(1.618)
cost=c()
degree=c()
trainError=c()
testError=c()
for(i in 1:5){
  
  tune=tune(svm, y~., data=train, kernel="polynomial", 
            ranges=list(cost=C, degree=c(i)))
  cost[i]=tune$best.model$cost
  degree[i]=tune$best.model$degree
  trainError[i]=tune$best.performance
  testError[i]=mean(predict(
    tune$best.model, newdata = valid[,-1]) != valid[,1])
  
}

iii=cbind(degree, cost, 
          Train=round(trainError,3), 
          Test=round(testError,3))
iii

# (iv)
set.seed(1.618)
indices=list()
I = 1:nrow(d)

# Get 10 random samples
for (i in 1:10){
  
  indices[[i]] = sort(sample(I, size=17+as.numeric(i >= 8)))
  I = I[!is.element(I, indices[[i]])]
  
}

metaK = c()
for (m in 1:100) {
  # For each k, get average error over 10 samples
  meanErrors=c()
  for(k in 1:17) {
    errorsOfk=c()
    for(i in 1:length(indices)) {
      knn=knn(train=d[-indices[[i]], -1], 
              test=d[indices[[i]], -1], 
              cl=d[-indices[[i]], 1], k = k)
      errorsOfk[i] = mean(knn != d[indices[[i]], 1])
    }
    # Average error for k over 10 samples
    meanErrors[k] = mean(errorsOfk)
  }
  # Get which k had min average error over 10 samples
  metaK[m] = which.min(meanErrors)
}
# Over 100 attempts, get frequencies of which k had min average error
table(metaK)

# (v)
G=c(.01,.1,1,10)
C=c(.01,.1,1,2,5,10,50,100)
D=1:4
E=matrix(NA, 100, 12)

for (i in 1:100){
  
  set.seed(sqrt(3) * i)
  
  # Runtime is about 10 mins
  
  s=sort(sample(1:nrow(d), size=25))
  
  for(j in 1:4){
    
    rad.svm = tune(svm, y~., data=d[-s,], kernel="radial", 
                   ranges=list(cost=C, gamma=G[j]))$best.model
    
    pol.svm = tune(svm, y~., data=d[-s,], kernel="polynomial", 
                   ranges=list(cost=C, degree=D[j]))$best.model
    
    E[i,j] = mean(predict(rad.svm, newdata=d[s,-1]) != d[s,1])
    E[i,j+4] = mean(predict(pol.svm, newdata=d[s,-1]) != d[s,1])
    
  }
  
  glm=glm(y~., data=d[-s,], family=binomial)
  lda=lda(y~., data=d[-s,])
  qda=qda(y~., data=d[-s,])
  knn = knn(train=d[-s,-1], test=d[s,-1], cl=d[-s,1], k=9)
  
  glm.p = as.numeric(predict(glm, newdata=d[s,-1], type="response") >= .5)
  
  E[i,9]=mean(glm.p != d[s,1])
  E[i,10]=mean(predict(lda, newdata=d[s,-1])$class != d[s,1])
  E[i,11]=mean(predict(qda, newdata=d[s,-1])$class != d[s,1])
  E[i,12]=mean(knn != d[s,1])
  
}

# Radial SVMs (Y=.01,.1,1,10)
apply(E[,1:4], 2, mean)

# Polynomial SVMs (d=1,2,3,4)
apply(E[,5:8], 2, mean)

# GLM, LDA, QDA, KNN
apply(E[,9:12], 2, mean)

boxplot(E[,1:4], names=c("Y=.01","Y=.1","Y=1","Y=10"),
        main="Rad SVM Mean Error Rates")
boxplot(E[,5:8], names=c("d=1","d=2","d=3","d=4"),
        main="Poly SVM Mean Error Rates")
boxplot(E[,9:12], names=c("Log Reg", "LDA", "QDA", "KNN9"),
        main="Non-SVM Mean Error Rates")
boxplot(E, main="Mean Error Rates")

# P2

# (iii)
set.seed(1.618)
X = sort(runif(100, min=0, max=10))
theta1=c()
for (i in 1:1000) {
  
  s = sort(sample(X, size=100, replace = T))
  theta1[i] = s[80]
  
}

theta1=sort(theta1)
CI = c(theta1[25], theta1[975]); CI

# (iv)
S=c(0,0)
# Test 1000 intervals of each method
# Runtime about 15 seconds
for (i in 1:1000) {
  
  set.seed(1.618*i)
  X = sort(runif(100, min=0, max=10))
  theta2=c()
  
  # Each interval based on B=200 estimates of q_.8
  for (j in 1:200) {
    
    s = sort(sample(X, size=100, replace = T))
    theta2[j] = s[80]
    
  }
  
  # Vector of q_.8 estimates
  theta2 = sort(theta2)
  
  S[1] = S[1] + as.numeric(theta2[5]<8 & 8<theta2[195])
  
  m = mean(theta2)
  SE = sqrt(1/199 * sum((theta2 - m)^2))
  S[2] = S[2] + as.numeric(X[80]-1.96*SE<8 & 8<X[80]+1.96*SE)
  
}
# Percentile method
S[1]/1000
# Standard error method
S[2]/1000

# (v)
S=c(0,0)
# Test 1000 intervals of each method
# Runtime about 12 seconds
for (i in 1:1000) {
  
  set.seed(3.141*i)
  X = sort(runif(100, min=0, max=10))
  theta3=c()
  
  
  # Each interval based on B=200 estimates of q_.8
  for (j in 1:200) {
    
    s = sort(sample(X, size=100, replace = T))
    theta3[j] = s[99]
    
  }
  
  # Vector of q_.8 estimates
  theta3 = sort(theta3)
  
  S[1] = S[1] + as.numeric(theta3[5]<9.9 & 9.9<theta3[195])
  
  m = mean(theta3)
  SE = sqrt(1/199 * sum((theta3 - m)^2))
  S[2] = S[2] + as.numeric(X[99]-1.96*SE<9.9 & 9.9<X[99]+1.96*SE)
  
}
# Bootstrap estimates for 80th percentile of population
# Percentile method
S[1]/1000
# Standard error method
S[2]/1000
