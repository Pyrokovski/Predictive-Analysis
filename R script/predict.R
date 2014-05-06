#--------------------------------------------------------------------------------------
# increase the java heap size
#--------------------------------------------------------------------------------------
options(java.parameters = "-Xmx3g") 

#--------------------------------------------------------------------------------------
library(quantmod)
library(leaps) 
library(matrixStats)
library(forecast)
library(RWeka)
library(clusterSim)	# ... for normalization
library(foreign)
setwd("~/Github/Predictive Analysis/Data")

#--------------------------------------------------------------------------------------
# Load saved data
#--------------------------------------------------------------------------------------
load("train.RData")
load("test.RData")
train.data	<-train.data[,-1] 	# remove the id column
test.id		<-test.data$id		# store id for test data
test.data	<-test.data[,-1] 	# remove the id column

# Convert the loss column to char attribute (nominal)
train.data$loss<-as.factor(train.data$loss)

#--------------------------------------------------------------------------------------
# Ensemble Learning using Bagging with J48 meta learning
#--------------------------------------------------------------------------------------
en.model<-Bagging(loss ~ ., data=train.data,control = Weka_control(W = list(J48, M = 2)))
en.model.output<-summary(en.model)
save(en.model, file = "enmodelJ48.RData")

#--------------------------------------------------------------------------------------
# Prediction using the best model
#--------------------------------------------------------------------------------------
pre.en.model<-predict(en.model,test.data)

#--------------------------------------------------------------------------------------
# Output result
#--------------------------------------------------------------------------------------
write.table(cbind(test.id,pre.en.model), "~/Github/Predictive Analysis/Output/prediction.csv",sep=",") # store predition