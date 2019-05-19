
addpath('fmri', 'fmri/Netlab')


% load the data
%examples = csvread('data/encoded_fmri_X.csv')
examples = csvread('data/autoassociated_fmri_X.csv')
%labels = csvread('data/encoded_fmri_y.csv')
labels = csvread('data/autoassociated_fmri_y.csv')

% split the data in half
ntotal   = size(examples,1);
oindices = 1:2:(ntotal-1);
eindices = 2:2:ntotal;
trainExamples = examples(oindices,:);
trainLabels   = labels(oindices,1);
testExamples  = examples(eindices,:);
testLabels    = labels(eindices,1);

% train a classifier
[classifier] = trainClassifier(trainExamples,trainLabels,'nbayes');

% apply a classifier
[predictions] = applyClassifier(testExamples,classifier);

% use the predictions to compute a success metric
% (accuracy and average rank now, precision/recall, forced
% proportions,etc in the future)
% result is a cell array to pack the result in (a single number for
% the two metrics implemented)
% trace contains the extra information we talked about producing with
% rankings of labels - please see the out section of the comments in
% summarizePredictions
[result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'accuracy',testLabels);
result{1}