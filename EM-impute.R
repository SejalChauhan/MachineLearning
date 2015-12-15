library(Amelia)
library(foreign)
library(imputeR)

data<-read.arff("/Users/sejalc/PycharmProjects/MachineLearning/impute-exo/planetdata2.arff")

trainX <- data[, names(data) != "class"]
a.out <- amelia(data, m=5, empri = 193)
write.amelia(obj=a.out, file.stem = "/Users/sejalc/PycharmProjects/MachineLearning/impute-exo/EM-Amelia-class.txt")