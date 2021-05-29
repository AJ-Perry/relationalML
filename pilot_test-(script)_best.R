library(neuralnet)##load library---------------------------------------
library(data.table)
library(caret)
library(elmNN)
library(elmNNRcpp)
library(fbasics)
library(sigmoid)
library(keras)
library(mlbench) 
library(dplyr)
library(magrittr)
install.packages("tensorflow")
library(tensorflow)
#getwd()
##ptest <- read.csv("U_ELM_data.csv", header = T)
set.seed(101)
Data_for_UCI_named_2_s <- data.table(Data_for_UCI_named_2_)
Data_for_UCI_named_2_s %<>% mutate_if(is.factor, as.numeric)##--conversts var to num if fac---------------------------------------
ptest <- Data_for_UCI_named_2_s[sample(.N, 3000)]




View(ptest)#load data-----------------------------------------------
str(ptest)#eval data------------------------------------------------
attach(ptest)#attach data-------------------------------------------
names(ptest)#view variable names------------------------------------

#combine like columns-----------------------------------------------------
##

#min/max = normalize variable rate to "0/1"-------------------------------
ptest$TAU1 <- as.numeric(ptest$TAU1 - min(ptest$TAU1))/(max(ptest$TAU1) - min(ptest$TAU1))
ptest$TAU2 <- as.numeric(ptest$TAU2 - min(ptest$TAU2))/(max(ptest$TAU2) - min(ptest$TAU2))
ptest$TAU3 <- as.numeric(ptest$TAU3 - min(ptest$TAU3))/(max(ptest$TAU3) - min(ptest$TAU3))
ptest$TAU4 <- as.numeric(ptest$TAU4 - min(ptest$TAU4))/(max(ptest$TAU4) - min(ptest$TAU4))
ptest$P1 <- as.numeric(ptest$P1 - min(ptest$P1))/(max(ptest$P1) - min(ptest$P1))
ptest$P2 <- as.numeric(ptest$P2 - min(ptest$P2))/(max(ptest$P2) - min(ptest$P2))
ptest$P3 <- as.numeric(ptest$P3 - min(ptest$P3))/(max(ptest$P3) - min(ptest$P3))
ptest$P4 <- as.numeric(ptest$P4 - min(ptest$P4))/(max(ptest$P4) - min(ptest$P4))
ptest$G1 <- as.numeric(ptest$G1 - min(ptest$G1))/(max(ptest$G1) - min(ptest$G1))
ptest$G2 <- as.numeric(ptest$G2 - min(ptest$G2))/(max(ptest$G2) - min(ptest$G2))
ptest$G3 <- as.numeric(ptest$G3 - min(ptest$G3))/(max(ptest$G3) - min(ptest$G3))
ptest$G4 <- as.numeric(ptest$G4 - min(ptest$G4))/(max(ptest$G4) - min(ptest$G4))
#data partition = segregate 'training' vs. 'testing' data-------------------
set.seed(555)
ind <- sample(3, nrow(ptest), replace = T, prob = c(0.45, 0.35, 0.20))
training.ptest <- ptest[ind==1,]
validation.ptest <- ptest[ind==2,]
testing.ptest <- ptest[ind==3,]


#can specify which network to compute, e.g."plot(n, rep=2) based on which one returns minimum error====
library(neuralnet)
set.seed(777)
nn.ptest_trn <- neuralnet(STABILITY~TAU1+TAU2+TAU3+TAU4+P1+P2+P3+P4+G1+G2+G3+G4,
                          data =  training.ptest, 
                          hidden = c(3,3),
                          nhid ='6',
                          err.fct = 'sse',
                          linear.output = F,
                          lifesign = 'full',
                          rep = 5,
                          method = 'elm',
                          likelihood = T,
                          threshold = '0.01',
                          learningrate = '0.01',
                          act.fct  = 'sig',
                          stepmax = 100000)#100k default----------------------------------------------------------------

plot(nn.ptest_trn, rep = 'best',
     col.entry = 'yellow',
     col.out = 'blue',
     col.hidden = 'red',
     color.hidden.synapse = 'green',
     show.weights = T,
     information = T,
     information.pos = 0.1,
     radius = 0.15,
     arrow.length = 0.2,
     intercept = T,
     intercept.factor = 0.4,
     col.intercept = 'blue',
     fontsize = 16,
     dimension = 5,
     fill = 'orange2')

# Matrix
ptest.mat <- as.matrix(ptest)
dimnames(ptest.mat) <- NULL

# Partition
set.seed(1234)
ind <- sample(2, nrow(ptest.mat), replace = T, prob = c(.7, .3))
training.ptest <- ptest.mat[ind==1,1:12]
testing.ptest <- ptest.mat[ind==2, 1:12]
training.ptest.target <- ptest.mat[ind==1, 13]
testing.ptest.target  <- ptest.mat[ind==2, 13]

# Normalize
m.trn <- colMeans(training.ptest)
s.trn <- apply(training.ptest, 2, sd)
training.ptest <- scale(training.ptest, center = m.trn, scale = s.trn)
testing.ptest <- scale(testing.ptest, center = m.trn, scale = s.trn)

trn.model <- keras_model_sequential()

trn.model %>%
  layer_dense(units = 5, activation = 'relu', input_shape = c(13)) %>%
  layer_dense(units = 1)

# Compile
trn.model %>% compile(loss = 'mse',
                      optimizer = 'rmsprop',
                      metrics = 'mae')

# Fit Model
f.trn.model <- trn.model %>%
  fit(training.ptest,
      trainingtarget,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2)

# Evaluate
f.trn.model %>% evaluate(testing.ptest, testing.ptest.target)
trn.pred <- f.trn.model %>% predict(testing.ptest)
mean((testing.ptest.target-trn.pred)^2)
plot(testing.ptest.target-trn.pred)

####
#neural network/training data-------------------------------------------------------------
library(neuralnet)
set.seed(777)
nn.ptest_val <- neuralnet(STABILITY~TAU1+TAU2+TAU3+TAU4+P1+P2+P3+P4+G1+G2+G3+G4,
                          data =  validation.ptest, 
                          hidden = c(3,3),
                          err.fct = 'sse',
                          linear.output = F,
                          lifesign = 'full',
                          rep = 5,
                          likelihood = T,
                          threshold = '0.05',
                          learningrate = '0.05',
                          algorithm = 'rprop+',
                          act.fct = 'tanh',
                          stepmax = 100000)#100k default----------------------------------------------------------------


plot(nn.ptest_val, rep = 'best',
     col.entry = 'yellow',
     col.out = 'blue',
     col.hidden = 'red',
     color.hidden.synapse = 'green',
     show.weights = T,
     information = T,
     information.pos = 0.1,
     radius = 0.15,
     arrow.length = 0.12,
     intercept = T,
     intercept.factor = 0.4,
     col.intercept = 'red',
     fontsize = 16,
     dimension = 5,
     fill = 'blue2')


#neural network/training data-------------------------------------------------------------
library(neuralnet)
set.seed(777)
sigmoid <- function(x) 1/(1+exp(-x))
nn.ptest_tst <- neuralnet(STABILITY~TAU1+TAU2+TAU3+TAU4+P1+P2+P3+P4+G1+G2+G3+G4,
                          data =  testing.ptest, 
                          hidden = c(5,3),
                          err.fct = 'sse',
                          linear.output = F,
                          lifesign = 'full',
                          rep = 5,
                          likelihood = T,
                          threshold = '0.05',
                          learningrate = '0.05',
                          algorithm = 'rprop+',
                          stepmax = 100000)#100k default----------------------------------------------------------------

plot(nn.ptest_tst, rep = 'best',
     col.entry = 'yellow',
     col.out = 'blue',
     col.hidden = 'red',
     color.hidden.synapse = 'green',
     show.weights = T,
     information = T,
     information.pos = 0.1,
     radius = 0.15,
     arrow.length = 0.12,
     intercept = T,
     intercept.factor = 0.4,
     col.intercept = 'blue',
     fontsize = 16,
     dimension = 5,
     fill = 'blue4')

##xtest = testing.ptest[, -2]---------------------
ytest = testing.ptest[, 2]

ypred = neuralnet::compute(nn.ptest_tst, testing.ptest)
yhat = ypred$net.result
print(yhat)

cm = confusionMatrix(as.factor(ytest), yhat$yhat)
#prediction---------------------------------------------------------------------------------------------------------
#training data
tst_output <- compute(nn.ptest_tst, testing.ptest[,-1])#can specify which network, from 'plot' to compute, e.g."training rep=1) based on which one returns minimum error====
head(tst_output$net.result)#output eval------------------------
head(testing.ptest[1,])#training eval---------------------------

#prediction-------------------------------------------------
trn_output <- compute(nn.ptest_trn, training.ptest[,-1])
head(trn_output$net.result)
#test data
tst_output <- compute(nn.ptest_tst, testing.ptest)#can specify which network, from 'plot' to compute, e.g."training rep=2) based on which one returns minimum error====
head(nn.ptest_tst$net.result)#output eval------------------------
head(testing.ptest[1,])#training eval---------------------------

#node output calculations w/Sigmoid Activation Function-----
in6.14 <- 2.97744 + (7.20548*0.01154772) + (-2.20773*0) + (2.135*0) + (3.9579*0.04531658) + (-7.96198*0)
sigmoid <- function(ptest) 1/(1+exp(-ptest))
in6.14.1 <- 2.97744 + (-0.3131533*out6.14)
out6.14.1 <- 1/(1+exp(-in6.14.1))

#confusion matrix & misclassification error - training data-------------
output.14 <- compute(nn.ptest_tst, training.14)
p.14 <- output.14$net.result
pred.14 <- ifelse(p.14>0.5, 1, 0)
tab.14 <- table(pred.14, training.14$Deaths)
tab.14
1-sum(diag(tab.14))/sum(tab.14)

#confusion matrix & misclassification error - testing data-------------
tst_output <- compute(nn.ptest_tst, testing.ptest[,-1])
tst_output.e <- tst_output$net.result
##test.pred <- predict(nn.ptest_tst, nn.ptest_tst)
tst.pred <- ifelse(tst_output.e>0.01, 1.00, 0.9982517)
tst.pred.tabl <- table(tst.pred, testing.ptest$STABILITY)
tst.pred.tabl
1-sum(diag(tst.pred.tabl))/sum(tst.pred.tabl)

print(tab.trn.pred)
print(tst.pred.tab1)

write.csv(tst.pred.tabl)

testing.ptest$STABILITY
tst.pred

##plot(sigmoid(ptest))
sigmoid <- function(x) 1/(1+exp(-x))

# use sigmoid with default standard logistic
( b <- sigmoid(tst_output.e) )
# show shape
plot(b)
# inverse
hist( a - sigmoid(tst_output.e, inverse=TRUE) )
# with SoftMax
( c <- sigmoid(tst_output.e, SoftMax=TRUE) )
# show diffe


devtools::install_github("rstudio/keras", dependencies = TRUE)
devtools::install_github("rstudio/tensorflow", dependencies = TRUE)