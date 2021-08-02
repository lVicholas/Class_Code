# HW2
# Problem 2
train.p2=read.csv("Problem2.csv")
test.p2=read.csv("Problem2test.csv")

library(MASS)
library(ggplot2)

buff = 2
x1seq=seq(min(train.p2$X1)-buff,max(train.p2$X1)+buff,by=.1)
x2seq=seq(min(train.p2$X2)-buff,max(train.p2$X2)+buff,by=.1)
grid=expand.grid(X1=x1seq,X2=x2seq)

# mod1
mod1=glm(Y~X1+X2, family = binomial, data=train.p2)

prob.grid1=predict.glm(mod1, newdata=grid, type="response")
pred.grid1=as.numeric(prob.grid1 >= .5)

ggplot(data=train.p2, aes(x=X1,y=X2)) + 
  geom_point(aes(colour=as.factor(Y)), shape=17, size=2) + 
  geom_point(data=grid,aes(x=X1,y=X2,colour=as.factor(pred.grid1)), size=.01) +
  theme_bw() + ggtitle("Linear GLM Model")

pred.train1=predict.glm(mod1, newdata=train.p2[,-1], type="response")
p1=as.numeric(pred.train1 >= .5)
mean(p1 != train.p2[,1])

# mod2
mod2=glm(Y~X1+I(X1**2)+X2+I(X2**2), family=binomial, data=train.p2)

prob.grid2=predict.glm(mod2, newdata=grid, type="response")
pred.grid2=as.numeric(prob.grid2 >= .5)

ggplot(data=train.p2, aes(x=X1,y=X2)) + 
  geom_point(aes(colour=as.factor(Y)), shape=17, size=2) + 
  geom_point(data=grid,aes(x=X1,y=X2,colour=as.factor(pred.grid2)), size=.01) +
  theme_bw() + ggtitle("Quadratic GLM")

pred.train2=predict.glm(mod2, newdata=train.p2[,-1],type="response")
p2=as.numeric(pred.train2 >= .5)
mean(p != train.p2[,1])

# mod3
mod3=lda(Y~X1+X2, data=train.p2)
pred.grid3 = predict(mod3, newdata=grid)$class

ggplot(data=train.p2, aes(x=X1,y=X2)) + 
  geom_point(aes(colour=as.factor(Y)), shape=17, size=2) + 
  geom_point(data=grid,aes(x=X1,y=X2,colour=as.factor(pred.grid3)), size=.01) +
  theme_bw() + ggtitle("LDA")

pred.train3=predict(mod3, newdata=train.p2[,-1])
mean(pred.train3$class != train.p2[,1])


# mod4
mod4=qda(Y~X1+X2, data=train.p2)
pred.grid4 = predict(mod4, newdata=grid)$class

ggplot(data=train.p2, aes(x=X1,y=X2)) + 
  geom_point(aes(colour=as.factor(Y)), shape=17, size=2) + 
  geom_point(data=grid,aes(x=X1,y=X2,colour=as.factor(pred.grid4)), size=.01) +
  theme_bw() + ggtitle("QDA")

pred.train4=predict(mod4, data=train.p2[,-1])
mean(pred.train4$class != train.p2[,1])

####
t1=predict.glm(mod1, newdata = test.p2[,-1], type="response")
v1=as.numeric(t1 >= .5)
mean(v1 != test.p2[,1])

t2=predict.glm(mod2, newdata=test.p2[,-1], type="response")
v2=as.numeric(t2 >= .5)
mean(v2 != test.p2[,1])

v3=predict(mod3, newdata=test.p2[,-1])$class
mean(v3 != test.p2[,1])

v4=predict(mod4, newdata=test.p2[,-1])$class
mean(v4 != test.p2[,1])


# P3
train.p3 = read.csv("Problem3.csv")
test.p3 = read.csv("Problem3test.csv")

lda=lda(Y~X1+X2, data=train.p3)
qda=qda(Y~X1+X2, data=train.p3)

buff = 2
x1seq=seq(min(train.p3$X1)-buff,max(train.p3$X1)+buff,by=.1)
x2seq=seq(min(train.p3$X2)-buff,max(train.p3$X2)+buff,by=.1)
grid=expand.grid(X1=x1seq,X2=x2seq)

# LDA
pred.grid1=predict(lda, newdata=grid)$class

ggplot(data=train.p3, aes(x=X1,y=X2)) + 
  geom_point(data=train.p3,aes(x=X1,y=X2,colour=as.factor(Y)), shape=17, size=2) + 
  geom_point(data=grid,aes(x=X1,y=X2,colour=as.factor(pred.grid1)), size=.01) +
  theme_bw() + ggtitle("LDA")

lda.train=predict(lda, newdata=train.p3[,-1])
mean(lda.train$class != train.p3[,1])

# QDA
pred.grid2=predict(qda, newdata=grid)$class

ggplot(data=train.p3, aes(x=X1,y=X2)) + 
  geom_point(data=train.p3,aes(x=X1,y=X2,colour=as.factor(Y)), shape=17, size=2) + 
  geom_point(data=grid,aes(x=X1,y=X2,colour=as.factor(pred.grid2)), size=.01) +
  theme_bw() + ggtitle("QDA")

qda.train=predict(qda, newdata=train.p3[,-1])
mean(qda.train$class != train.p3[,1])

####

t3.1=predict(lda, newdata = test.p3[,-1])$class
mean(t3.1 != test.p3[,1])

t3.2=predict(qda, newdata=test.p3[,-1])$class
mean(t3.2 != test.p3[,1])

table(train.p3[,1])