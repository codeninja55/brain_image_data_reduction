library("R.matlab")
data <- readMat("input.mat")

examples <- data$examples
labels <- data$labels

# split train and test data
train_indices <- sort(sample(nrow(examples), nrow(examples)*.5))

trainExamples <- examples[train_indices, ]
trainLabels <- labels[train_indices, ]
testExamples <- examples[-train_indices, ]
testLabels <- labels[-train_indices, ]

pca <- prcomp(trainExamples, center = TRUE, scale. = TRUE)
summary(pca)

cumpro <- cumsum(pca$sdev^2 / sum(pca$sdev^2))
plot(cumpro, xlab = "PC #", ylab = "Amount of explained variance", main = "Cumulative variance plot")
abline(v = 24, col="blue", lty=5)
abline(h = 1, col="blue", lty=5)
legend("topleft",
       col=c("blue"), lty=5, cex=0.6)

trainExamples <- pca$x[,-25]

testExamples <- predict(pca, newdata = testExamples)
testExamples <- testExamples[, -25]


writeMat("output.mat", trainExamples = trainExamples, trainLabels = trainLabels, 
         testExamples = testExamples, testLabels = testLabels)

