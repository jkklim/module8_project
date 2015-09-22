# Prediction Assignment Writeup
LIM KAH KHENG  
22 September 2015  

# Background


Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 


```r
suppressWarnings(suppressMessages(library(Hmisc)))
suppressWarnings(suppressMessages(library(caret)))
suppressWarnings(suppressMessages(library(randomForest)))
suppressWarnings(suppressMessages(library(foreach)))
suppressWarnings(suppressMessages(library(doParallel)))

set.seed(1234)

setwd("e:/module8/project")

# read files
# values contained a "#DIV/0!" will be replaced with an NA value.

trainingdata <- read.csv("pml-training.csv", na.strings=c("#DIV/0!") )
testdata <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!") )
```


# PreProcessing

Columns 8 and above to cast to numeric


```r
options(warn=-1)
for(i in c(8:ncol(trainingdata)-1)) {trainingdata[,i] = as.numeric(as.character(trainingdata[,i]))}
for(i in c(8:ncol(testdata)-1)) {testdata[,i] = as.numeric(as.character(testdata[,i]))}
```


Choose feature set that only included complete columns. Remove user name, timestamps and windows. 

```r
feature_set <- colnames(trainingdata[colSums(is.na(trainingdata)) == 0])[-(1:7)]
model_data <- trainingdata[feature_set]
feature_set
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```

# Build model

```r
index <- createDataPartition(y=model_data$classe, p=0.75, list=FALSE )
training <- model_data[index,]
testing <- model_data[-index,]
```

# Random Forest with Parallel Processing

We can now train a classifier with the training data. To do that we will use parallelise the processing with the foreach and doParallel package : we call registerDoParallel to instantiate the configuration. So we ask to process 6 random forest with 150 trees.


```r
registerDoParallel()
x <- training[-ncol(training)]
y <- training$classe

rf <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {randomForest(x, y, ntree=ntree)}
```


# list the variables importance


```r
imp <- varImp(rf)
imp$Variable <- row.names(imp)
imp[order(imp$Overall, decreasing = T),]
```

```
##                        Overall             Variable
## roll_belt            954.33141            roll_belt
## yaw_belt             669.04676             yaw_belt
## pitch_forearm        567.83219        pitch_forearm
## magnet_dumbbell_z    566.13683    magnet_dumbbell_z
## pitch_belt           524.68309           pitch_belt
## magnet_dumbbell_y    499.01859    magnet_dumbbell_y
## roll_forearm         453.92000         roll_forearm
## magnet_dumbbell_x    365.54324    magnet_dumbbell_x
## roll_dumbbell        313.25752        roll_dumbbell
## magnet_belt_z        308.45348        magnet_belt_z
## accel_dumbbell_y     303.54616     accel_dumbbell_y
## accel_belt_z         298.14903         accel_belt_z
## magnet_belt_y        285.14191        magnet_belt_y
## accel_forearm_x      244.77716      accel_forearm_x
## roll_arm             241.89871             roll_arm
## accel_dumbbell_z     238.69086     accel_dumbbell_z
## gyros_belt_z         226.24682         gyros_belt_z
## magnet_forearm_z     212.99347     magnet_forearm_z
## yaw_dumbbell         199.48258         yaw_dumbbell
## magnet_arm_x         197.12378         magnet_arm_x
## total_accel_dumbbell 195.11175 total_accel_dumbbell
## accel_dumbbell_x     191.80179     accel_dumbbell_x
## gyros_dumbbell_y     189.07906     gyros_dumbbell_y
## magnet_belt_x        187.86945        magnet_belt_x
## yaw_arm              186.73611              yaw_arm
## accel_forearm_z      183.10723      accel_forearm_z
## accel_arm_x          176.00622          accel_arm_x
## magnet_arm_y         173.64190         magnet_arm_y
## magnet_forearm_x     170.11418     magnet_forearm_x
## magnet_forearm_y     164.83429     magnet_forearm_y
## total_accel_belt     158.63823     total_accel_belt
## magnet_arm_z         144.37123         magnet_arm_z
## pitch_dumbbell       133.38427       pitch_dumbbell
## pitch_arm            133.12413            pitch_arm
## yaw_forearm          126.97380          yaw_forearm
## accel_arm_y          118.10747          accel_arm_y
## accel_forearm_y      106.21114      accel_forearm_y
## gyros_arm_y          101.89255          gyros_arm_y
## gyros_dumbbell_x     100.23037     gyros_dumbbell_x
## gyros_arm_x           99.41206          gyros_arm_x
## gyros_forearm_y       97.53514      gyros_forearm_y
## accel_arm_z           97.44525          accel_arm_z
## accel_belt_y          92.74393         accel_belt_y
## accel_belt_x          84.74569         accel_belt_x
## total_accel_forearm   84.59747  total_accel_forearm
## gyros_belt_y          82.75063         gyros_belt_y
## total_accel_arm       79.32839      total_accel_arm
## gyros_belt_x          73.49829         gyros_belt_x
## gyros_forearm_z       66.74291      gyros_forearm_z
## gyros_dumbbell_z      65.15999     gyros_dumbbell_z
## gyros_forearm_x       56.25979      gyros_forearm_x
## gyros_arm_z           43.48308          gyros_arm_z
```

# Prediction with confusion Matrix


```r
predictions1 <- predict(rf, newdata=training)
confusionMatrix(predictions1,training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
predictions2 <- predict(rf, newdata=testing)
confusionMatrix(predictions2,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    2    0    0    0
##          B    0  945    8    0    0
##          C    0    2  845    5    0
##          D    0    0    2  799    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9961         
##                  95% CI : (0.994, 0.9977)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9951         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9958   0.9883   0.9938   1.0000
## Specificity            0.9994   0.9980   0.9983   0.9995   1.0000
## Pos Pred Value         0.9986   0.9916   0.9918   0.9975   1.0000
## Neg Pred Value         1.0000   0.9990   0.9975   0.9988   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1927   0.1723   0.1629   0.1837
## Detection Prevalence   0.2849   0.1943   0.1737   0.1633   0.1837
## Balanced Accuracy      0.9997   0.9969   0.9933   0.9966   1.0000
```


# Prepare submission


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

x <- testdata
x <- x[feature_set[feature_set!='classe']]
answers <- predict(rf, newdata=x)

answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
pml_write_files(answers)
```










