#############################################################
# Electrical Grid Stability Simulated project
#############################################################

#############################################################
# Load Required Packages and download the Electrical Grid 
# Stability Simulated project dataset from UCI Machine 
# Learning Repository
# 
# Abstract: The local stability analysis of the 4-node star system (electricity
# producer is in the center) implementing Decentral Smart Grid Control concept.
#
# Source:
# -- Creator and donor: Vadim Arzamasov (vadim.arzamasov '@' kit.edu),
# Department of computer science, Karlsruhe Institute of Technology; Karlsruhe,
# 76131; Germany -- Date: November, 2018
#
# Data Set Information:
# The analysis is performed for different sets of input values using the
# methodology similar to that described in [SchÃ¤fer, Benjamin, et al. 'Taming
# instabilities in power grid networks by decentralized control.' The European
# Physical Journal Special Topics 225.3 (2016): 569-582.]. Several input values
# are kept the same: averaging time: 2 s; coupling strength: 8 s^-2; damping:
# 0.1 s^-1
#
# Attribute Information:
# 11 predictive attributes, 1 non-predictive(p1), 2 goal fields:
# 1. tau[x]: reaction time of participant (real from the range [0.5,10]s). Tau1
# - the value for electricity producer.
# 2. p[x]: nominal power consumed(negative)/produced(positive)(real). For
# consumers from the range [-0.5,-2]s^-2; p1 = abs(p2 + p3 + p4)
# 3. g[x]: coefficient (gamma) proportional to price elasticity (real from the
# range [0.05,1]s^-1). g1 - the value for electricity producer.
# 4. stab: the maximal real part of the characteristic equation root (if
# positive - the system is linearly unstable)(real)
# 5. stabf: the stability label of the system (categorical: stable/unstable)

#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# Electrical Grid Stability Simulated dataset:
# https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv
dl <- tempfile()
download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv", dl)
egss <- read.csv2(dl, header = TRUE, sep = ",")
# Set data types to align with source
egss <- as.data.frame(egss) %>% mutate(tau1 = as.numeric(tau1),
                                     tau2 = as.numeric(tau2),
                                     tau3 = as.numeric(tau3),
                                     tau4 = as.numeric(tau4),
                                     p1 = as.numeric(p1),
                                     p2 = as.numeric(p2),
                                     p3 = as.numeric(p3),
                                     p4 = as.numeric(p4),
                                     g1 = as.numeric(g1),
                                     g2 = as.numeric(g2),
                                     g3 = as.numeric(g3),
                                     g4 = as.numeric(g4),
                                     stab = as.numeric(stab),
                                     stabf = as.character(stabf))


# Test set will be 10% of Electrical Grid Stability Simulated data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = egss$stabf, times = 1, p = 0.2, list = FALSE)
train <- egss[-test_index,]
temp <- egss[test_index,]
# Make sure stabf in test set are also in train set
test <- temp %>% 
  semi_join(train, by = "stabf")
# Add rows removed from test set back into train set
removed <- anti_join(temp, test)
train <- rbind(train, removed)
# Remove staging data used to create training and test sets
rm(dl, test_index, temp, removed)

#############################################################
# Data Exploration
#############################################################
# Data structure
str(train)
head(train)
# Check for null values
sapply(train, {function(x) any(is.na(x))})
# Data summary
summary(train)
# Plotting stablitiy
p <-ggplot(train, aes(x=stab, fill = stabf)) +
  geom_density()
p

#############################################################
# Modeling - K-nearest neighbor classifier
#############################################################
# create 10%  index of the total number of rows in dataset.
index <- sample(1:nrow(egss), 0.1 * nrow(egss)) 
# normalize function
nor <-function(x) { (x -min(x))/(max(x)-min(x))   }
#Run nomalization the 11 predictors
egss_norm <- as.data.frame(lapply(egss[,c(1,2,3,4,6,7,8,9,10,11,12)], nor))
# Summary
summary(egss_norm)
##extract training set
egss_train <- egss_norm[-index,] 
##extract testing set
egss_test <- egss_norm[index,] 
##extract stabf column of train dataset because it will be used as 'cl' argument
##in knn function.
egss_target_category <- egss[index,14] 
##extract stabf if test dataset to measure the accuracy
egss_test_category <- egss[-index,14]
# run knn function
pr <- knn(egss_train,egss_test,cl=egss_target_category,k=21)
# create confusion matrix
tab <- table(pr,egss_test_category)
# this function divides the correct predictions by total number of predictions
# that tell us how accurate the model is.
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
# Plotting results
cm <- confusionMatrix(tab)
draw_confusion_matrix <- function(cm) {
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Class1', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Class2', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Class1', cex=1.2, srt=90)
  text(140, 335, 'Class2', cex=1.2, srt=90)
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  

draw_confusion_matrix(cm)
