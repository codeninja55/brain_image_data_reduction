% example_animation
%
% working example illustrating use of fmri_core software to load
% IDM data, transform it into examples, do feature selection, train NBayes classifier,
% apply it to test data, and see the result.
%
% History:
%  - Oct 07,2005 Wei - rewrite

% load the data.  returns cell arrays of IDMs
load data-starplus-04847-v7.mat;
[info,ndata,nmeta,activeVoxels] = transformIDM_selectActiveVoxact(info, data, meta, 5);


% transform one IDM them into examples
[examples,labels,expInfo] = idmToExamples_fixation(info,ndata,nmeta,'full');

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

