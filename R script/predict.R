options(java.parameters = "-Xmx3g") # increase the java heap size
#--------------------------------------------------------------------------------------
library(quantmod)
library(tseries)
library(leaps) 
library(matrixStats)
library(forecast)
library(RWeka)
library(clusterSim)	# ... for normalization
library(ROCR)
setwd("/Users/harrythompson/Desktop")
#--------------------------------------------------------------------------------------

train.data <- read.table("train_v2.csv", header=TRUE, sep=",")
test.data <- read.table("test_v2.csv", header=TRUE, sep=",")
save(train.data, file = "train.RData")
save(train.data1, file = "train.omit.na.RData")

#--------------------------------------------------------------------------------------
# Load saved data
#--------------------------------------------------------------------------------------
load("train.RData")
load("test.RData")
train.data<-train.data[,-1] # remove the id column
test.data<-test.data[,-1] # remove the id column
rm(train.data) # remove the train data
rm(test.data) # remove the test data

# Convert the loss column to char attribute (nominal)
train.data$loss<-as.factor(train.data$loss)

#--------------------------------------------------------------------------------------
#normalizing data
#--------------------------------------------------------------------------------------
train.data1<-Normalize(loss~., data=train.data[,-1])

#--------------------------------------------------------------------------------------
# J48 tree decision algorithm
#--------------------------------------------------------------------------------------
model<-J48(loss ~.,data=train.data)
model.output<-summary(model) 
save(model, file = "modelJ48.RData")
#--------------------------------------------------------------------------------------
# Ensemble Learning models
#--------------------------------------------------------------------------------------
WPM("list-packages", "available")
# Using ensemble learning
#class <- make_Weka_classifier("weka/classifiers/functions/meta/Bagging")
#WOW(class)      

# Bagging with J48 meta learning
#-------------------------------
en.model<-Bagging(loss ~ ., data=train.data,control = Weka_control(W = list(J48, M = 2)))
en.model.output<-summary(en.model)
save(en.model, file = "enmodelJ48.RData")

# Bagging with default meta learning cor 93.1363 % MAE 0.003 RMS 0.0331
#-------------------------------
ensemble <- make_Weka_classifier("weka/classifiers/trees/REPTree")
WOW(ensemble)
en.model1<-Bagging(loss ~ ., data=train.data, control = Weka_control(W = list(ensemble)))
en.model1.output<-summary(en.model1)
save(en.model1, file = "enmodelREPTree.RData")


# Other works
#-------------------------------
naivebaye<-make_Weka_classifier("weka/classifiers/bayes/NaiveBayes")
WOW(naivebaye)
nb.model<-naivebaye(loss ~ ., data=train.data)
nb.model.output<-summary(nb.model)

# Attribute selection 
#-------------------------------
attribselec <- make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")
WOW(attribselec)

# using GainRatioAttributeEval and Ranker
#-------------------------------
GainRARanker<-Weka_control(E = list("weka/attributeSelection/GainRatioAttributeEval") ,S=list("weka/attributeSelection/Ranker"))

GainRatioAttributeEval.model<-attribselec(loss ~ ., data=train.data, control= GainRARanker)
GainRatioAttributeEval.model.output<-summary(GainRatioAttributeEval.model)

# Attribute selection using CfsSubsetEval cor 89.8441 % MAE 0.005 RMS 0.0499
#-------------------------------
CfsSubsetEval.model<-attribselec(loss ~ ., data=train.data)
CfsSubsetEval.model.output<-summary(CfsSubsetEval.model)

# IBk model k-nearest neighbors classifier cor 99.9114 % MAE 0.0001 RMS 0.0042
#-------------------------------
IBk.model<-IBk (loss ~ ., data=train.data)
IBk.model.output<-summary(IBk.model)
#save(IBk.model, file = "IBk.model.RData")

#perf <- performance(IBk.model, "tpr", "fpr")
#plot(perf)

#--------------------------------------------------------------------------------------
# Prediction using the best model
#--------------------------------------------------------------------------------------
pre.en.model<-predict(IBk.model,test.data)

#--------------------------------------------------------------------------------------
# Output result
#--------------------------------------------------------------------------------------
id<-test.id
loss<-pre.en.model
write.table(cbind(id,loss), "prediction.csv",sep=",") # store predition
