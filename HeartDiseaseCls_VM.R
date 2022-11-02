# Code by: VIMAL MOTHI
#-------------------------------------
# Description: According to the CDC, heart disease is one of the leading causes of death for people of most races in the US (African Americans,
# American Indians and Alaska Natives, and white people). About half of all Americans (47%) have at least 1 of 3 key risk factors for heart 
# disease: high blood pressure, high cholesterol, and smoking. Other key indicator include diabetic status, obesity (high BMI), not getting enough
# physical activity or drinking too much alcohol. Detecting and preventing the factors that have the greatest impact on heart disease is very 
# important in healthcare
#-------------------------------------
# Data set with 18 Features (Classification) - 14 Categorical and 4 Continuous
# link : https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

# Import Packages
library(missForest)
library(DataExplorer)
library(missMethods)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(Boruta)
library(naivebayes)
library(e1071)
library(rpart)
library(rpart.plot)
library(performanceEstimation)
library(caret)
library(randomForest)
library(mice)
library(ROCR)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Simulating Missing Values at Random~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

df1 = read.csv("/Users/vimalmothi/Desktop/AML assignment/R Code/heart_2020.csv", 
               stringsAsFactors = TRUE)
set.seed(123)
split_resample = sample.split(df1$HeartDisease, SplitRatio = 0.95)
df1_resample = subset(df1, split_resample == FALSE)
prop.table(table(df1_resample$HeartDisease))
prop.table(table(df1$HeartDisease))
table(df1_resample$HeartDisease)


df2 = delete_MCAR(df1_resample, c(0.05, 0.03, 0.03, 0.04, 0.06, 0.07, 0.02), 
                  c("MentalHealth", "PhysicalHealth","AgeCategory","Diabetic","Race","SleepTime","Smoking"))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DATA PRE-PROCESSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# Structure of Dataset
dim(df2)
str(df2)
plot_str(df2)
summary(df2)
head(df2)
df2 = mutate_all(df2,na_if,"")
colSums(sapply(df2,is.na)) #missing values by column
sum(is.na(df2))#Total number of missing values
md.pattern(df2)

#--------------------------------------------------------------------------------------------------------------------------
#                                                   Distribution Plots
#--------------------------------------------------------------------------------------------------------------------------

#--------------------
# Numerical variables
#--------------------

plot1 = ggplot(df2, aes(x=BMI)) + 
         geom_histogram(aes(y=..density..),color="black", fill="lightblue",binwidth = 1)+
         geom_vline(aes(xintercept=mean(BMI)),
             color="blue", linetype="dashed", size=1)

plot2 = ggplot(df2, aes(x=PhysicalHealth)) + 
         geom_histogram(color="black", fill="pink",binwidth = 1)+
         geom_vline(aes(xintercept=mean(PhysicalHealth)),
             color="blue", linetype="dashed", size=1)

plot3 = ggplot(df2, aes(x=MentalHealth)) + 
         geom_histogram(color="black", fill="red",binwidth = 1)+
         geom_vline(aes(xintercept=mean(MentalHealth)),
             color="blue", linetype="dashed", size=1)

plot4 = ggplot(df2, aes(x=SleepTime)) + 
         geom_histogram(color="black", fill="green",binwidth = 1)+
         geom_vline(aes(xintercept=mean(SleepTime)),
             color="blue", linetype="dashed", size=1)

grid.arrange(plot1, plot2, plot3, plot4)

# Density plot
plot_density(df2)

# Boxplot
require(reshape2)
ggplot(data = melt(df2), aes(x=variable, y=value)) + geom_boxplot(aes(fill=variable))

#----------------------
# Categorical Variables
#----------------------

# Barplot

barchart(df2)
barchart(df2$HeartDisease, xlab= "Count", ylab = "Heart Disease", 
         main = "Patients in dataset", col = "blue",)
table(df2$HeartDisease)
prop.table(table(df2$HeartDisease))


#--------------------------------------------------------------------------------------------------------------------------
#                                                    Imputations
#--------------------------------------------------------------------------------------------------------------------------

plot_missing(df1)
plot_missing(df2)

df3=df2 #df3 will be fully imputed dataset


# Numerical Imputations
df3$PhysicalHealth = ifelse(is.na(df3$PhysicalHealth) ,
                             ave(df3$PhysicalHealth , FUN = function(x) median(x, na.rm = TRUE)),
                             df3$PhysicalHealth)

df3$MentalHealth = ifelse(is.na(df3$MentalHealth) ,
                            ave(df3$MentalHealth , FUN = function(x) median(x, na.rm = TRUE)),
                            df3$MentalHealth)

df3$SleepTime = ifelse(is.na(df3$SleepTime) ,
                          ave(df3$SleepTime , FUN = function(x) median(x, na.rm = TRUE)),
                          df3$SleepTime)

colSums(sapply(df3,is.na)) #missing values by column


# Categorical Imputations
i1 = !sapply(df3, is.numeric)
i1

Mode <- function(x) { 
  ux <- sort(unique(x))
  ux[which.max(tabulate(match(x, ux)))] 
}

Mode(df3$Smoking)
Mode(df3$AgeCategory)
Mode(df3$Diabetic)
Mode(df3$Race)

df3[i1] = lapply(df3[i1], function(x)
  replace(x, is.na(x), Mode(x[!is.na(x)])))

colSums(sapply(df3,is.na)) #missing values by column

# Imputed Dataset
plot_missing(df3)

#--------------------------------------------------------------------------------------------------------------------------
#                                                           EDA
#--------------------------------------------------------------------------------------------------------------------------


# EDA of categorical variables with high levels
age_table = table(df3$HeartDisease,df3$AgeCategory)
barplot(age_table, 
        main = "Heart Disease by Age Category", 
        xlab = "Age Group", ylab = "Frequency", 
        col = c("green","red"),
        legend.text = rownames(age_table), 
        args.legend = list(title = "Heart Disease", x = 'topright', inset = c(-0.08,0)), 
        beside = TRUE)
par(mar=c(3,3,4,5))

GenHealth_table = table(df3$HeartDisease,df3$GenHealth)
barplot(GenHealth_table, 
        main = "Heart Disease by GenHealth Category", 
        xlab = "General Health", ylab = "Frequency", 
        col = c("green","red"),
        legend.text = rownames(GenHealth_table),
        args.legend = list(title = "Heart Disease", x = 'topright', inset = c(-0.22,0)),
        beside = TRUE)


# Correlation plot
plot_correlation(df3,'continuous', cor_args = list("use" = "pairwise.complete.obs"))


#--------------------------------------------------------------------------------------------------------------------------
#                                                      Normalization
#--------------------------------------------------------------------------------------------------------------------------

# Range of values before normalization
summary(df3%>% dplyr::select(where(is.numeric)))

# Sleeptime
df3$SleepTime = (df3$SleepTime  - min(df3$SleepTime ))/(max(df3$SleepTime ) - min(df3$SleepTime))

# BMI
df3$BMI = (df3$BMI  - min(df3$BMI ))/(max(df3$BMI ) - min(df3$BMI))

# PhysicalHealth
df3$PhysicalHealth = (df3$PhysicalHealth - min(df3$PhysicalHealth))/(max(df3$PhysicalHealth) - min(df3$PhysicalHealth))

# MentalHealth
df3$MentalHealth = (df3$MentalHealth  - min(df3$MentalHealth ))/(max(df3$MentalHealth ) - min(df3$MentalHealth))

# Range of values after normalization
summary(df3%>% dplyr::select(where(is.numeric)))

plot_density(df3)
str(df3)

#--------------------------------------------------------------------------------------------------------------------------
#                                                      ENCODING
#--------------------------------------------------------------------------------------------------------------------------


df4 = df3 #df4 will be encoded dataset

# Label Encoder (Ordinal)

LabelEncoder = function(x,order = unique(x)){
  x = as.numeric(factor(x, levels = order, exclude = NULL))
  x
}


table(df4[["AgeCategory"]], LabelEncoder(df4[["AgeCategory"]], 
                                         order = c("18-24","25-29","30-34","35-39","40-44","45-49",
                                                   "50-54","55-59","60-64","65-69","70-74","75-79","80 or older")), useNA = "ifany")
table(df4[["GenHealth"]], LabelEncoder(df4[["GenHealth"]], order = c("Excellent","Very good","Good","Fair","Poor")), useNA = "ifany")

# AgeCategory
df4$AgeCategory = LabelEncoder(df4[["AgeCategory"]],
                               order = c("18-24","25-29","30-34","35-39","40-44","45-49",
                                         "50-54","55-59","60-64","65-69","70-74","75-79","80 or older"))
range(df4$AgeCategory)

# GenHealth
df4$GenHealth = LabelEncoder(df4[["GenHealth"]],
                             order = c("Excellent","Very good","Good","Fair","Poor"))
range(df4$GenHealth)



# One-Hot Encoder (Norminal)
df4$HeartDisease = as.numeric(df4$HeartDisease)
dmy = dummyVars("~.", data = df4)
df4 = data.frame(predict(dmy, newdata = df4))
df4$HeartDisease = as.factor(df4$HeartDisease)
df4$HeartDisease = factor(df4$HeartDisease, levels = c('1',"2"), label = c("0","1"))
summary(df4)
str(df4)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FEATURE SELECTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

set.seed(124)
boruta.train = Boruta(HeartDisease~., data = df4, doTrace = 2)
print(boruta.train)

final.boruta = TentativeRoughFix(boruta.train)
print(final.boruta)

plot(final.boruta, xlab = '', xaxt = "n")
lz = lapply(1:ncol(final.boruta$ImpHistory),function(i)
  final.boruta$ImpHistory[is.finite(final.boruta$ImpHistory[,i]),i])
names(lz) = colnames(final.boruta$ImpHistory)
Labels = sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),at = 1:ncol(final.boruta$ImpHistory), cex.axis = 0.7)
par(mar = c(11, 4, 4, 3))

getSelectedAttributes(final.boruta, withTentative = F)
df_fe = attStats(final.boruta)

# Extracting confirm variable from complete dataset
boruta_rejected = subset(df_fe, subset = df_fe$decision == "Rejected")
str(boruta_rejected)
rej = t(boruta_rejected)
rej_empty = rej[-c(1:6), ]
rej_names <- colnames(rej_empty[,])
cols_rej <- c(rej_names)
df5 = df4[, !(colnames(df4) %in% cols_rej)]

str(df5)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MODEL BUILDING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

# Experiment 1  : - Test prediction without balancing dataset and with balancing dataset.
#                 - For Balanced dataset, test prediction with balancing before train-test split and after 
#                   train-test split
#                 
#                 * Only for Logistic Regression. Best appoach will be used for building other predictive models

# Experiment 2 : Which Model performs the best in its predictive ability


#--------------------------------------------------------------------------------------------------------------------------
#                                               DATASET SPLITTING
#--------------------------------------------------------------------------------------------------------------------------


# Stratified Sampling 
set.seed(124)
split = sample.split(df5$HeartDisease, SplitRatio = 0.8)
training_set= subset(df5, split == TRUE)
test_set = subset(df5,split == FALSE)
table(training_set$HeartDisease)
table(test_set$HeartDisease)

# Balancing by Oversampling training set using SMOTE
balanced_train = smote(HeartDisease~.,training_set, perc.over = 6, perc.under = 1.7)
table(balanced_train$HeartDisease)

# Proportions of labels in training (balanced) and test dataset
prop.table(table(balanced_train$HeartDisease))
prop.table(table(test_set$HeartDisease))


# Unbalanced dataset - Train : training_set  ; Test  : test_set

# Balanced dataset  - Train : balanced_train ; Test : test_set   ~~~ Only training set is balanced

#--------------------------------------------------------------------------------------------------------------------------
#                                                Logistic Regression
#--------------------------------------------------------------------------------------------------------------------------

#------------------
# Unbalanced Test
#------------------
lg_ub = glm(HeartDisease~.,training_set ,family = binomial)
summary(lg_ub)

pred_probtraining_un = predict(lg_ub, type ="response", training_set)
pred_classtraining_un = ifelse(pred_probtraining_un > 0.5, 1, 0)
confusionMatrix(data = as.factor(pred_classtraining_un), reference = training_set$HeartDisease, positive = "1")

pred_probtest_un = predict(lg_ub, type ="response", test_set)
pred_classtest_un= ifelse(pred_probtest_un > 0.5, 1, 0)
confusionMatrix(data = as.factor(pred_classtest_un), reference = test_set$HeartDisease, positive = "1")

#------------------
# Balanced Test
#------------------
lg_b = glm(HeartDisease~., balanced_train, family = binomial)
summary(lg_b)

pred_probtraining_b = predict(lg_b, type ="response", balanced_train)
pred_classtraining_b = ifelse(pred_probtraining_b > 0.5, 1, 0)
confusionMatrix(data = as.factor(pred_classtraining_b), reference = balanced_train$HeartDisease, positive = "1")

pred_probtest_b = predict(lg_b, type ="response", test_set)
pred_classtest_b= ifelse(pred_probtest_b > 0.5, 1, 0)
(confusionMatrix(data = as.factor(pred_classtest_b), reference = test_set$HeartDisease, positive = "1"))$byClass


# ROC Curve

pred_lg = prediction(pred_probtest_b,test_set$HeartDisease)
perf_lg = performance(pred_lg,"tpr","fpr")
par(mfrow=c(1,1,3))

plot(perf_lg, colorize = T)
plot(perf_lg, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))  

auc_lg = as.numeric(performance(pred_lg, "auc")@y.values)
auc_lg =  round(auc_lg, 3)
auc_lg


#--------------------------------------------------------------------------------------------------------------------------
#                                                NAIVE BAYES
#--------------------------------------------------------------------------------------------------------------------------

#----------------------------
# Basic
#----------------------------

Naive_Bayes_basic = naive_bayes(x = balanced_train[ , -1],
                                y = balanced_train$HeartDisease , laplace = 0)

pred_probtraining_nb_basic = predict(Naive_Bayes_basic, newdata = balanced_train[, -1], type ="prob")
pred_classtraining_nb_basic = predict(Naive_Bayes_basic, newdata = balanced_train[, -1], type ="class")
confusionMatrix(pred_classtraining_nb_basic, reference = balanced_train$HeartDisease, positive = "1")

pred_probtest_nb_basic = predict(Naive_Bayes_basic, newdata = test_set[, -1], type ="prob")
pred_classtest_nb_basic = predict(Naive_Bayes_basic, newdata = test_set[, -1], type ="class")
confusionMatrix(pred_classtest_nb_basic, reference = test_set$HeartDisease, positive = "1")

# ROC Curve

pred_nb_basic = prediction(as.numeric(pred_classtest_nb_basic),test_set$HeartDisease)
perf_nb_basic = performance(pred_nb_basic,"tpr","fpr")
par(mfrow=c(1,1,3))

plot(perf_nb_basic, colorize = T)
plot(perf_nb_basic, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))  

auc_nb_basic = as.numeric(performance(pred_nb_basic, "auc")@y.values)
auc_nb_basic  =  round(auc_nb_basic, 3)
auc_nb_basic 
#----------------------------
# Apply Laplace Smoothing
#----------------------------

Naive_Bayes_LP = naive_bayes(x = balanced_train[ , -1],
                                y = balanced_train$HeartDisease , laplace = 1)

pred_probtraining_nb_LP = predict(Naive_Bayes_LP, newdata = balanced_train[, -1], type ="prob")
pred_classtraining_nb_LP = predict(Naive_Bayes_LP, newdata = balanced_train[, -1], type ="class")
confusionMatrix(pred_classtraining_nb_LP, reference = balanced_train$HeartDisease, positive = "1")

pred_probtest_nb_LP = predict(Naive_Bayes_LP, newdata = test_set[, -1], type ="prob")
pred_classtest_nb_LP = predict(Naive_Bayes_LP, newdata = test_set[, -1], type ="class")
confusionMatrix(pred_classtest_nb_LP, reference = test_set$HeartDisease, positive = "1")

# ROC

pred_nb_LP = prediction(as.numeric(pred_classtest_nb_LP),test_set$HeartDisease)
perf_nb_LP = performance(pred_nb_LP,"tpr","fpr")
par(mfrow=c(1,1,3))

plot(perf_nb_LP, colorize = T)
plot(perf_nb_LP, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))  

auc_nb_lp = as.numeric(performance(pred_nb_LP, "auc")@y.values)
auc_nb_lp  =  round(auc_nb_lp, 3)
auc_nb_lp 

#----------------------------
# Apply KDE
#----------------------------

Naive_Bayes_KDE = naive_bayes(x = balanced_train[ , -1],
                              y = balanced_train$HeartDisease , laplace = 1, usekernel = TRUE,
                              usepoisson = FALSE )
 

pred_classtraining_nb_KDE = predict(Naive_Bayes_KDE, newdata = balanced_train[, -1], type ="class")
confusionMatrix(pred_classtraining_nb_KDE, reference = balanced_train$HeartDisease, positive = "1")

pred_classtest_nb_KDE = predict(Naive_Bayes_KDE, newdata = test_set[, -1], type ="class")
confusionMatrix(pred_classtest_nb_KDE, reference = test_set$HeartDisease, positive = "1")

# ROC

pred_nb_KDE = prediction(as.numeric(pred_classtest_nb_KDE),test_set$HeartDisease)
perf_nb_KDE = performance(pred_nb_KDE,"tpr","fpr")
par(mfrow=c(1,1,3))

plot(perf_nb_KDE, colorize = T)
plot(perf_nb_KDE, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))  

auc_nb_KDE = as.numeric(performance(pred_nb_KDE, "auc")@y.values)
auc_nb_KDE  =  round(auc_nb_KDE, 3)
auc_nb_KDE 
#----------------------------
# Apply POISSON
#----------------------------

Naive_Bayes_PS = naive_bayes(x = balanced_train[ , -1],
                              y = balanced_train$HeartDisease , laplace = 1, usekernel = FALSE,
                              usepoisson = TRUE )


pred_classtraining_nb_PS = predict(Naive_Bayes_PS, newdata = balanced_train[, -1], type ="class")
confusionMatrix(pred_classtraining_nb_PS, reference = balanced_train$HeartDisease, positive = "1")

pred_classtest_nb_PS = predict(Naive_Bayes_PS, newdata = test_set[, -1], type ="class")
confusionMatrix(pred_classtest_nb_PS, reference = test_set$HeartDisease, positive = "1")

# ROC

pred_nb_PS = prediction(as.numeric(pred_classtest_nb_PS),test_set$HeartDisease)
perf_nb_PS = performance(pred_nb_PS,"tpr","fpr")
par(mfrow=c(1,1,3))

plot(perf_nb_PS, colorize = T)
plot(perf_nb_PS, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))  

auc_nb_PS = as.numeric(performance(pred_nb_PS, "auc")@y.values)
auc_nb_PS  =  round(auc_nb_PS, 3)
auc_nb_PS 
#----------------------------
# Apply KDE & POISSON
#----------------------------

Naive_Bayes_KDE_PS = naive_bayes(x = balanced_train[ , -1],
                             y = balanced_train$HeartDisease , laplace = 1, usekernel = TRUE,
                             usepoisson = TRUE )


pred_classtraining_nb_KDE_PS = predict(Naive_Bayes_KDE_PS, newdata = balanced_train[, -1], type ="class")
confusionMatrix(pred_classtraining_nb_KDE_PS, reference = balanced_train$HeartDisease, positive = "1")

pred_classtest_nb__KDE_PS = predict(Naive_Bayes_KDE_PS, newdata = test_set[, -1], type ="class")
confusionMatrix(pred_classtest_nb__KDE_PS, reference = test_set$HeartDisease, positive = "1")

# ROC

pred_nb_KDE_PS = prediction(as.numeric(pred_classtest_nb__KDE_PS),test_set$HeartDisease)
perf_nb_KDE_PS = performance(pred_nb_KDE_PS,"tpr","fpr")
par(mfrow=c(1,1,3))

plot(perf_nb_KDE_PS, colorize = T)
plot(perf_nb_KDE_PS, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))  

auc_nb_KDE_PS = as.numeric(performance(pred_nb_KDE_PS, "auc")@y.values)
auc_nb_KDE_PS  =  round(auc_nb_KDE_PS, 3)
auc_nb_KDE_PS 
#----------------------------
# Tuned NB
#----------------------------

trControl_nb = trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(123)
nb_tuned = train(HeartDisease~.,balanced_train,"naive_bayes",trControl =trControl_nb )


pred_classtraining_nb_tuned = predict(nb_tuned$finalModel, newdata = balanced_train[, -1], type ="class")
confusionMatrix(pred_classtraining_nb_tuned, reference = balanced_train$HeartDisease, positive = "1")

pred_classtest_nb_tuned = predict(nb_tuned$finalModel, newdata = test_set[, -1], type ="class")
confusionMatrix(pred_classtest_nb_tuned, reference = test_set$HeartDisease, positive = "1")

# ROC

pred_nb_tuned = prediction(as.numeric(pred_classtest_nb_tuned),test_set$HeartDisease)
perf_nb_tuned = performance(pred_nb_tuned,"tpr","fpr")
par(mfrow=c(1,1,3))

plot(perf_nb_tuned, colorize = T)
plot(perf_nb_tuned, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))  

auc_nb_tuned = as.numeric(performance(pred_nb_tuned, "auc")@y.values)
auc_nb_tuned  =  round(auc_nb_tuned, 3)
auc_nb_tuned 


#--------------------------------------------------------------------------------------------------------------------------
#                                                SUPPORT VECTOR MACHINES
#--------------------------------------------------------------------------------------------------------------------------
#----------------------------
# RBF
#----------------------------

svm_rbf = svm(HeartDisease~., data = balanced_train)
summary(svm_rbf)
svm_rbf$sigma
svm_rbf$cost

pred_classtraining_svm_rbf = predict(svm_rbf, newdata = balanced_train, type ="class")
confusionMatrix(pred_classtraining_svm_rbf, reference = balanced_train$HeartDisease, positive = "1")


pred_classtest_svm_rbf = predict(svm_rbf, newdata = test_set, type ="class")
confusionMatrix(pred_classtest_svm_rbf, reference = test_set$HeartDisease, positive = "1")

# ROC

pred_svm_rbf = prediction(as.numeric(pred_classtest_svm_rbf),test_set$HeartDisease)
perf_svm_rbf = performance(pred_svm_rbf,"tpr","fpr")
par(mfrow=c(1,1,3))

plot(perf_svm_rbf, colorize = T)
plot(perf_svm_rbf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))  

auc_svm_rbf = as.numeric(performance(pred_svm_rbf, "auc")@y.values)
auc_svm_rbf  =  round(auc_svm_rbf, 3)
auc_svm_rbf 
#----------------------------
# LINEAR
#----------------------------

svm_linear = svm(HeartDisease~., data = balanced_train, kernel = "linear")
summary(svm_linear)
svm_linear$sigma
svm_linear$cost

pred_classtraining_svm_linear = predict(svm_linear, balanced_train)
confusionMatrix(pred_classtraining_svm_linear, reference = balanced_train$HeartDisease, positive = "1")

pred_classtest_svm_linear = predict(svm_linear, test_set)
confusionMatrix(pred_classtest_svm_linear, reference = test_set$HeartDisease, positive = "1")


# ROC

pred_svm_ln = prediction(as.numeric(pred_classtest_svm_linear),test_set$HeartDisease)
perf_svm_ln = performance(pred_svm_ln,"tpr","fpr")
par(mfrow=c(1,1,3))

plot(perf_svm_ln, colorize = T)
plot(perf_svm_ln, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))  

auc_svm_ln = as.numeric(performance(pred_svm_ln, "auc")@y.values)
auc_svm_ln  =  round(auc_svm_ln, 3)
auc_svm_ln 
#----------------------------
# SIGMOID
#----------------------------

svm_sigmoid = svm(HeartDisease~., data = balanced_train, kernel = "sigmoid")
summary(svm_sigmoid)
svm_sigmoid$sigma
svm_sigmoid$cost

pred_classtraining_svm_sigmoid = predict(svm_sigmoid, balanced_train)
confusionMatrix(pred_classtraining_svm_sigmoid, reference = balanced_train$HeartDisease, positive = "1")


pred_classtest_svm_sigmoid = predict(svm_sigmoid, test_set)
confusionMatrix(pred_classtest_svm_sigmoid, reference = test_set$HeartDisease, positive = "1")

# ROC

pred_svm_sg = prediction(as.numeric(pred_classtest_svm_sigmoid),test_set$HeartDisease)
perf_svm_sg = performance(pred_svm_sg,"tpr","fpr")
par(mfrow=c(1,1,3))

plot(perf_svm_sg, colorize = T)
plot(perf_svm_sg, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))  

auc_svm_sg = as.numeric(performance(pred_svm_sg, "auc")@y.values)
auc_svm_sg  =  round(auc_svm_sg, 3)
auc_svm_sg 
#----------------------------
# TUNED SVM
#----------------------------

trControl_rndm = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "random")
set.seed(123)
svm_tuned = train(HeartDisease~.,balanced_train,"svmLinear",trControl =trControl_rndm, tunelength = 15 )
svm_tuned$bestTune
summary(svm_tuned)

pred_classtraining_svm_tuned = svm_tuned %>% predict(balanced_train)
confusionMatrix(pred_classtraining_svm_tuned, reference = balanced_train$HeartDisease, positive = "1")

pred_classtest_svm_tuned = svm_tuned %>% predict(test_set)
confusionMatrix(pred_classtest_svm_tuned, reference = test_set$HeartDisease, positive = "1")


# ROC

pred_svm_tune = prediction(as.numeric(pred_classtest_svm_tuned),test_set$HeartDisease)
perf_svm_tune = performance(pred_svm_tune,"tpr","fpr")
par(mfrow=c(1,1,3))

plot(perf_svm_tune, colorize = T)
plot(perf_svm_tune, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))  

auc_svm_tune = as.numeric(performance(pred_svm_tune, "auc")@y.values)
auc_svm_tune  =  round(auc_svm_tune, 3)
auc_svm_tune 
#--------------------------------------------------------------------------------------------------------------------------
#                                                DECISION TREES
#--------------------------------------------------------------------------------------------------------------------------

#----------------------------
# GINI SPLIT
#----------------------------

DT_gini = rpart(HeartDisease~ ., data = balanced_train, method = "class")
rpart.plot(DT_gini, extra = 101, nn = TRUE)

pred_classtraining_DT_gini = predict(DT_gini, balanced_train, type = "class")
confusionMatrix(pred_classtraining_DT_gini, reference = balanced_train$HeartDisease, positive = "1")

pred_probtest_DT_gini  = predict(DT_gini, test_set)
pred_classtest_DT_gini  = predict(DT_gini, test_set, type = "class")
confusionMatrix(pred_classtest_DT_gini, reference = test_set$HeartDisease, positive = "1")

plotcp(DT_gini)

# ROC

pred_DT_gini = prediction(pred_probtest_DT_gini[,2],test_set$HeartDisease)
perf_DT_gini = performance(pred_DT_gini,"tpr","fpr")
par(mfrow=c(1,1,3))

plot(perf_DT_gini, colorize = T)
plot(perf_DT_gini, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))  

auc_DT_gini = as.numeric(performance(pred_DT_gini, "auc")@y.values)
auc_DT_gini  =  round(auc_DT_gini, 3)
auc_DT_gini 

#----------------------------
# ENTROPY SPLIT
#----------------------------


DT_ent = rpart(HeartDisease ~ ., data=balanced_train, method="class", parms=list(split="information"))
rpart.plot(DT_ent, extra = 101, nn = TRUE)


pred_classtraining_DT_ent  = predict(DT_ent, balanced_train, type = "class")
confusionMatrix(pred_classtraining_DT_ent, reference = balanced_train$HeartDisease, positive = "1")

pred_probtest_DT_ent  = predict(DT_ent, test_set)
pred_classtest_DT_ent  = predict(DT_ent, test_set, type = "class")
confusionMatrix(pred_classtest_DT_ent, reference = test_set$HeartDisease, positive = "1")

plotcp(DT_ent)

# ROC

pred_DT_Ent = prediction(pred_probtest_DT_ent[,2],test_set$HeartDisease)
perf_DT_Ent = performance(pred_DT_Ent,"tpr","fpr")
par(mfrow=c(1,1,3))

plot(perf_DT_Ent, colorize = T)
plot(perf_DT_Ent, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))  

auc_DT_Ent = as.numeric(performance(pred_DT_Ent, "auc")@y.values)
auc_DT_Ent  =  round(auc_DT_Ent, 3)
auc_DT_Ent 




#----------------------------
# TUNED DT
#----------------------------

# Train Control Default
trControl_default = trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(123)

# Find Baseline value for cp
DT_default_cp = train(HeartDisease~.,balanced_train,"rpart",trControl =trControl_default )
print(DT_default_cp)

# Find Baseline value for maxdepth
DT_default_md = train(HeartDisease~.,balanced_train,"rpart2",trControl =trControl_default )
print(DT_default_md)

# Find tuned value for cp
set.seed(123)
tuneGrid_dt = expand.grid(.cp = seq(-0.15,0.15, by = 0.01))
DT_grid = train(HeartDisease~., data = balanced_train, trControl = trControl_default, 
                tuneGrid = tuneGrid_dt, "rpart")
print(DT_grid)

# Find tuned value for maxdepth
set.seed(123)
tuneGrid_dt_md = expand.grid(maxdepth = seq(0, 10, by = 1))
DT_grid_md = train(HeartDisease~., data = balanced_train, trControl = trControl_default, 
                   tuneGrid = tuneGrid_dt_md, "rpart2")
print(DT_grid_md)


# Build tuned model
DT_tuned = rpart(HeartDisease ~ ., data=balanced_train, method="class", cp = -0.01, maxdepth = 4)

pred_classtrain_DT_tuned_training = predict(DT_tuned, balanced_train, type = "class")
confusionMatrix(pred_classtrain_DT_tuned_training, reference = balanced_train$HeartDisease, positive = "1")


pred_probtest_DT__tuned_test = predict(DT_tuned, test_set)
pred_classtest_DT__tuned_test = predict(DT_tuned, test_set, type = "class")
confusionMatrix(pred_classtest_DT__tuned_test, reference = test_set$HeartDisease, positive = "1")

# Plot CP graph
plotcp(DT_tuned)
rpart.plot(DT_tuned, extra = 101, nn = TRUE)


# ROC

pred_DT_tuned = prediction(pred_probtest_DT__tuned_test[,2],test_set$HeartDisease)
perf_DT_tuned = performance(pred_DT_tuned,"tpr","fpr")
par(mfrow=c(1,1,3))

plot(perf_DT_tuned, colorize = T)
plot(perf_DT_tuned, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1-Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))  

auc_DT_tuned = as.numeric(performance(pred_DT_tuned, "auc")@y.values)
auc_DT_tuned  =  round(auc_DT_tuned, 3)
auc_DT_tuned

#-------------------------------------------------------- END OF PROGRAM -----------------------------------------------------------------------
# Discussion included in word doc write up
