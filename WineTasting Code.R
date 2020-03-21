# Hadiza Mamman
# Technical Interview
# Analytical Flavor Systems

library(GGally)
library(ggplot2)
library(caret)
library(car)
library(e1071)
library(MASS)
library(moments)
library(corrplot)
library(dplyr)
library(pls)
library(mgcv)
library(nlme)
library(MuMIn)

setwd("~/Desktop/analyticalFlavorSystems/")

redWine <- read.csv('winequality-red.csv', sep = ';')
whiteWine <- read.csv('winequality-white.csv', sep = ';')
redWine['color'] <- 'red'
whiteWine['color'] <- 'white'

data <- rbind(redWine,whiteWine)

head(data)
tail(data)
names(data)
summary(data)
str(data)

summary(data$quality)
# calculate binwidth using the Freedman-Diaconis rule
bw <- 2 * IQR(data$quality) / length(data$quality)^(1/3)
qplot(quality, data=data, geom= "histogram",fill=color, binwidth=bw)
#seems to follow a normal distribution, 

summary(data$alcohol)
bw <- 2 * IQR(data$alcohol) / length(data$alcohol)^(1/3)
qplot(alcohol, data=data, geom="histogram", fill=color, binwidth=bw )
# seems to be skewed to the right, will transform data
alcoholTrans <- BoxCoxTrans(data$alcohol)
alcoholTrans
str(alcoholTrans) #lambda=-1.8
# check skewness
skewness(data$alcohol)
# the value is .565, indicating the predictor variable is moderately skewed
qplot(alcohol, data=data, fill=color, binwidth=0.02)+
        scale_x_log10()
# after the log transformation the data resembles a normal distribution
skewness(log(data$alcohol)) # the value decreased to 0.381

summary(data$density)
bw <- 2 * IQR(data$density) / length(data$density)^(1/3)
qplot(density, data=data, geom="histogram", fill=color, binwidth=bw)
# seems to follow a normal distribution

summary(data$volatile.acidity)
bw <- 2 * IQR(data$volatile.acidity) / length(data$volatile.acidity)^(1/3)
qplot(volatile.acidity, data=data, geom="histogram", fill=color, binwidth=bw)
# seems to follow a normal distribution

summary(data$sulphates)
bw <- 2 * IQR(data$sulphates) / length(data$sulphates)^(1/3)
qplot(sulphates, data=data, geom="histogram", fill=color, binwidth=bw)
# seems to be slighlty skewed
skewness(data$sulphates) #value is 1.79 indicating extreme skewness
qplot(sulphates, data=data, fill=color, binwidth=0.02)+
  scale_x_log10() #now resembles a normal distribution w/ skew value of .4

summary(data$pH)
bw <- 2 * IQR(data$pH) / length(data$pH)^(1/3)
qplot(pH, data=data, geom="histogram", fill=color, binwidth=bw)
# seems to follow a normal distribution

summary(data$total.sulfur.dioxide)
bw <- 2 * IQR(data$total.sulfur.dioxide) / length(data$total.sulfur.dioxide)^(1/3)
qplot(total.sulfur.dioxide, data=data, geom="histogram", fill=color, binwidth=bw)
# seems to follow a normal distribution, w/ significant difference between red&white wine

summary(data$free.sulfur.dioxide)
bw <- 2 * IQR(data$free.sulfur.dioxide) / length(data$free.sulfur.dioxide)^(1/3)
qplot(free.sulfur.dioxide, data=data, geom="histogram", fill=color, binwidth=bw)
# seems to be skewed to the right
skewness(data$free.sulfur.dioxide) #value is 1.22 indicating significant skewness
qplot(free.sulfur.dioxide, data=data, fill=color, binwidth=0.02)+
  scale_x_log10()
skewness(log10(data$free.sulfur.dioxide))

summary(data$chlorides)
bw <- 2 * IQR(data$chlorides) / length(data$chlorides)^(1/3)
qplot(chlorides, data=data, geom="histogram", fill=color, binwidth=bw)
# skewed to the right, will transform
chlorideTrans <- BoxCoxTrans(data$chlorides)
chlorideTrans
# check skewness
skewness(data$chlorides)
# The value is 5.397, indicating significant skewness
qplot(chlorides, data=data, fill=color, binwidth=0.002)+
  scale_x_log10()
# atfer the log transformation the data resembles a normal distribution

summary(data$residual.sugar)
bw <- 2 * IQR(data$residual.sugar) / length(data$residual.sugar)^(1/3)
qplot(data$residual.sugar, data=data, geom="histogram", fill=color, binwidth=bw)
# seems to be extremely skewed to the right
skewness(data$residual.sugar) #value is 1.43
qplot(chlorides, data=data, fill=color, binwidth=0.02)+
  scale_x_log10()

summary(data$citric.acid)
bw <- 2 * IQR(data$citric.acid) / length(data$citric.acid)^(1/3)
qplot(data$citric.acid, data=data, geom="histogram", fill=color, binwidth=bw)
# seems to follow a normal distribution

summary(data$fixed.acidity)
bw <- 2 * IQR(data$fixed.acidity) / length(data$fixed.acidity)^(1/3)
qplot(data$fixed.acidity, data=data, geom="histogram", fill=color, binwidth=bw)
# seems to be extremely skewed to the right
qplot(data$fixed.acidity, data=data, fill=color, binwidth=0.02)+
  scale_x_log10() #big difference between red&white wine

#FILTERING PREDICTORS

ggpairs(data)
# seems to be correlation between : 
# alcohol&density,fixed acidity&density,residualsugar&denisty/sulfur dioxide

nearZeroVar(data)
#indicates no problematic near zero variance predictors
cor(data$alcohol,data$density)
# -.68
cor(data$fixed.acidity,data$density)
#.459
cor(data$residual.sugar,data$density)
#.553
cor(data$residual.sugar,data$total.sulfur.dioxide)
#.495

data_set <- data[,-13]
str(data_set)
#get rid of non numeric variable (color)

#remove high correlation variable
df <- cor(data_set)
df
highCorr = findCorrelation(df,cutoff = .6)
highCorr = sort(highCorr)
adjData = data_set[,-c(highCorr)]
str(adjData)
str(data)
# density and total.sulfur.dioxide were removed from the matrix

plot(data$pH, data$quality)
#seems that the quality is highest when the ph is between 3.2-3.5
plot(data$citric.acid,data$quality)
#consistent pattern w highest quality score when citric acid is between 0.35-0.5
plot(data$volatile.acidity, data$quality)
#consistent pattern w high quality score when acidity below 0.5
plot(data$residual.sugar,data$quality)
#higher quality scores when lower than 15
plot(data$chlorides, data$quality)
#consistent pattern w higher quality score when chloride content lower than 0.2
plot(data$alcohol,data$quality)
#inconsistent pattern
plot(data$fixed.acidity,data$quality)
#higher quality scores when below 10
plot(data$sulphates,data$quality)
#higher the sulphate content, the lower the score


# DATA SPLITTING

# first, center & scale data
wineData <- scale(adjData, center = TRUE, scale = TRUE)
wineData

set.seed(1)

trainIndex <- createDataPartition(adjData$quality, p=0.8, list = FALSE)
trainSet <- wineData[trainIndex,]
testSet <- wineData[-trainIndex,]

trainSet1 <- as.data.frame(trainSet)
testSet1 <- as.data.frame(testSet)

repeatSplits <- createDataPartition(trainSet, p=0.8, list = FALSE, times =3)
str(repeatSplits)

model1 <- plsr(quality~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+pH+sulphates+alcohol,data=trainSet)
summary(model1)

model2 <- plsr(quality~fixed.acidity+volatile.acidity+citric.acid+chlorides+pH+sulphates,data=trainSet1)
summary(model2)

predictions <- predict(model1,testSet[1:6], ncomp = 1:3)
predictions

predictions1 <- predict(model2,testSet1[1:6])
predictions1

head(data)
head(predictions)
head(predictions1)

model1res <- resid(model1)
print(model1res)
#majority of the residuals are close to zero, indicating a decent model
RMSE(testSet[1:6],predictions) #value is 5.61608
RMSE(trainSet[1:6],predictions)#value is 5.114963
#RMSE values for both train and test set are close indicating a good model

plot(quality~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+pH+sulphates+alcohol,data=trainSet)
abline(model1)


