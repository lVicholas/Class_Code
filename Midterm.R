# Orriols Midterm


# P3
library(pls)
library(glmnet)

rm(list=ls())
set.seed(-7)

p=100

# X1 denotes set where lasso outperforms PCR
# X2 denotes set where PCR outperforms lasso

X1.test=matrix(NA, 100*100, 100)
X2.test=matrix(0,  100*100, 100)

for(j in 1:p) {
  
  # Response for X1 is sum of covariates
  # Make only 1st 10 covariates predictive
  if(j <= 10)
    X1.test[,j]=rnorm(nrow(X1.test),j,sqrt(3))
  
  # Rest of covariates have small effect on response
  else X1.test[,j]=rnorm(nrow(X1.test),0,.1)
  
  # Response is mean of first 10 covariates
  # First 5 covariates are significant and variable
  if(j<=5)
    X2.test[,j]=rnorm(nrow(X2.test),50,sqrt(250))

  # Rest of covariates have small effect on response
  else X2.test[,j]=rnorm(nrow(X2.test),0,sqrt(.1))
  
}

# Make responses as function of testing predictors
Y1.test=rowSums(X1.test)
Y2.test=rowSums(X2.test[,1:5])

n=p
nsim=1000

# Each row stores MSEs of 4 models
MSEs=matrix(NA, nsim, 4)

for(sim in 1:nsim) {
  
  #set.seed(7*sim)
  
  X1.train=matrix(NA,n,p)
  X2.train=matrix(0 ,n,p)
  
  for(j in 1:p) {
    
    # Make training predictors in same manner as testing data
    if(j <= 10)
      X1.train[,j]=rnorm(n,j,sqrt(3))
    else X1.train[,j]=rnorm(n,0,.1)
    
    
    if(j<=5)
      X2.train[,j]=rnorm(n,50,sqrt(250))
    else X2.train[,j]=rnorm(n,0,sqrt(.1))
    
      
  }
  
  # Make training responses in same manner as testing data
  Y1.train=rowSums(X1.train)
  Y2.train=rowSums(X2.train[,1:5])
  
  # Fit models for data where lasso outperforms PCR
  # Use CV to get best lambda for lasso
  las1=cv.glmnet(X1.train, Y1.train, family="gaussian", alpha=1)
  pcr1=pcr(Y1.train~., data=data.frame(cbind(X1.train,Y1.train)))
  
  # Get models' predictions for data where lasso performs better
  las1.pred=predict(las1, newx=X1.test)
  pcr1.pred=predict(pcr1, newdata=X1.test, ncomp=5)
  
  # Fit models for data where PCR outperforms lasso
  las2=cv.glmnet(X2.train, Y2.train, family="gaussian", alpha=1)
  pcr2=pcr(Y2.train~., data=data.frame(cbind(X2.train,Y2.train)))
  
  # Get models' predictions for data where PCR performs better
  las2.pred=predict(las2, newx=X2.test)
  pcr2.pred=predict(pcr2, newdata=X2.test, ncomp=5)
  
  # MSEs of good lasso (data set 1) is 1st column
  MSEs[sim,1]=mean((las1.pred - Y1.test)^2)
  
  # MSEs of bad PCR (data set 1) is 2nd column
  MSEs[sim,2]=mean((pcr1.pred - Y1.test)^2)
  
  # MSEs of bad lasso (data set 2) is 3rd column
  MSEs[sim,3]=mean((las2.pred - Y2.test)^2)
  
  # MSEs of good pcr (data set 2) is 4th column
  MSEs[sim,4]=mean((pcr2.pred - Y2.test)^2)
  
}

# Lasso (left) should outperform PCR (right)
round(colMeans(MSEs)[1:2],5)

# PCR (right) should outperform Lasso (left)
round(colMeans(MSEs)[3:4],5)

rm(list=ls())

# Problem 4

# Read in data
train = read.csv("MidtermP4Train.csv")
train[,1]=as.factor(train[,1])
test=read.csv("MidtermP4Test.csv")
test[,1]=as.factor(test[,1])

# (i)
# Fit glm based on training data
glm=glm(Y~., data=train, family=binomial)

# Get glm predictions for test data, out-of-sample MSE
glm.test = as.numeric(predict(glm, newdata = test[,-1], type="response") > .5)
mean(glm.test != test[,1])

# (ii)
set.seed(9)

# Use bootstrap to estimate B_1 + exp(B_2)
# Runtime about 1 min

B = 10000
estimates=c()
for (b in 1:B) {
  
  # Get random indices of training data
  index=sort(sample(1:nrow(train), size=nrow(train), replace = T))
  
  # Make model based on random sample of training data
  glm.bs=glm(Y~., data=train[index,], family=binomial)
  
  # Record estimate for each model
  estimates[b]=glm.bs$coefficients[2] + exp(glm.bs$coefficients[3])
  
}

estimates=sort(estimates)

# This is sample estimate from model in part (i)
thetaHat=glm$coefficients[2]+exp(glm$coefficients[3])

# Gives percentile method CI
CI = c(estimates[.06*B], estimates[.94*B]); CI

# Gives standard error estimate CI
CI = thetaHat + c(-1,1)*sd(estimates)*qnorm(.94); CI

# (iii)
# Performs BSR on model
# Trace parameter suppresses lengthy stepwise models output
bsr=step(glm, trace = 0)

# Get MSE of BSR model
bsr.pred=as.numeric(predict(bsr, newdata=test[,-1], type="response") > .5)
mean(bsr.pred != test[,1])

# (iv)
# Make string for formula
V=paste("poly(X.",1:15, sep="")
for(i in 1:15)
  V[i]=paste(V[i], ",3)", sep="")
terms=paste(V, collapse = "+")
f=as.formula(paste("Y~",terms, sep=""))

# Fit model with poly(X.j,3) for j=1,...15
glm3=glm(f, data=train, family=binomial)

# Do BSR on this model
bsr.3=step(glm3)

# MSE of minimum AIC model with up to degree 3 terms
bsr.3.pred=as.numeric(
  predict(bsr.3, newdata = test[,-1], type="response")>.5)
mean(bsr.3.pred != test[,1])

# (vii)
library(e1071)
set.seed(9)

# List of gamma values to try
G = 10^(-3:3)

# List of cost values to try
C = c(.001,.01,.1,1,3,5,10,25,50,75,100)

# Does 10-fold CV on G and C, returns best (min MSE) SVM
rad.svm = tune(svm, Y~., data=train,
               ranges=list(cost=C, gamma=G))$best.model
rad.svm$cost; rad.svm$gamma

# Get out of sample MSE for best SVM
rad.svm.pred=predict(rad.svm, newdata=test[,-1])
mean(rad.svm.pred != test[,1])

# (viii)
library(ggplot2)

# Generate x-axis coordinates
x1.buff = (max(train[,2])-min(train[,2]))*.5
x1seq = seq(min(train[,2])-2*x1.buff, max(train[,2])+x1.buff, by=.1)

# Generate y-axis coordinates
x2.buff = (max(train[,3])-min(train[,3]))*.5
x2seq = seq(min(train[,3])-x2.buff, max(train[,3])+2*x2.buff, by=.1)

# Cartesian product and x and y-axis coordinates
grid=expand.grid(X.1=x1seq, X.2=x2seq)

# Set other predictors to 0
grid[,3:15]=0
names(grid)=paste("X.", 1:15, sep="")

# Use SVM to predict response for each point
pred.grid = predict(rad.svm, newdata=grid)

ggplot(data=train, aes(x=X.1,y=X.2)) + 
  geom_point(data=grid, aes(x=X.1,y=X.2,colour=as.factor(pred.grid)), size=.01) +
  theme_bw() + ggtitle("Radial SVM")
