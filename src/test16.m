load data-starplus-04847-v7.mat;

% create training examples consisting of 8 second (16 image) windows of data,
% labeled by whether the subject is viewing a picture or a sentence.  Label 1
% means they are reading a sentence, label=2 means they are viewing a picture.
% "examples" is a NxM array, where N is the number of training examples, and M is
% the number of features per training example (in this case, one feature per voxel
% at each of 16 time points).

    % collect the non-noise and non-fixation trials
    trials=find([info.cond]>1); 
    [info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);
    % seperate P1st and S1st trials
    [infoP1,dataP1,metaP1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='P'));
    [infoS1,dataS1,metaS1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='S'));
 
    % seperate reading P vs S
    [infoP2,dataP2,metaP2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[1:16]);
    [infoP3,dataP3,metaP3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[17:32]);
    [infoS2,dataS2,metaS2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[17:32]);
    [infoS3,dataS3,metaS3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[1:16]);

    % convert to examples
    [examplesP2,labelsP2,exInfoP2]=idmToExamples_condLabel(infoP2,dataP2,metaP2);
    [examplesP3,labelsP3,exInfoP3]=idmToExamples_condLabel(infoP3,dataP3,metaP3);
    [examplesS2,labelsS2,exInfoS2]=idmToExamples_condLabel(infoS2,dataS2,metaS2);
    [examplesS3,labelsS3,exInfoS3]=idmToExamples_condLabel(infoS3,dataS3,metaS3);

    % combine examples and create labels.  Label 'picture' 1, label 'sentence' 2.
    examplesP=[examplesP2;examplesP3];
    examplesS=[examplesS2;examplesS3];
    labelsP=ones(size(examplesP,1),1);
    labelsS=ones(size(examplesS,1),1)+1;
    examples=[examplesP;examplesS];
    labels=[labelsP;labelsS];
    
%     % split the data 
%     ntotal   = size(examples,1);
% 
%     % 50/50 split
%     P = 0.50;
%     idx = randperm(ntotal);
% 
%     oindices = idx(1:round(P*ntotal));
%     eindices = idx(round(P*ntotal)+1:end);
%     trainExamples = examples(oindices,:);
%     trainLabels   = labels(oindices,1);
%     testExamples  = examples(eindices,:);
%     testLabels    = labels(eindices,1);

examples = normalize(examples);
% save("input.mat", "examples", "labels", "-v6");
c = cvpartition(size(examples, 1), 'LeaveOut');
testAccuracy = [];

% leave one out CV
for i = 1:c.NumTestSets
   trainIndices = training(c, i);
   testIndices = test(c, i);
   trainExamples = examples(trainIndices, :);
   trainLabels = labels(trainIndices, 1);
   testExamples = examples(testIndices, :);
   testLabels = labels(testIndices, 1);
   
   % apply PCA
%    [coeff,scoreTrain,~,~,explained,mu] = pca(trainExamples);
%    sum_explained = 0;
%    idx = 0;
%    while sum_explained < 95
%        idx = idx + 1;
%        sum_explained = sum_explained + explained(idx);
%    end
%    fprintf('Number of principal components to take: %d\n', idx);
%    trainExamples = scoreTrain(:, 1:idx);
%    testExamples = (testExamples - mu) * coeff(:, 1:idx);
   
   % train model and predict
   [classifier] = trainClassifier(trainExamples,trainLabels,'nbayes');
   [predictions] = applyClassifier(testExamples,classifier, 'nbayes');
   [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'accuracy',testLabels);
   testAccuracy(i) = result{1};
end

disp(mean(testAccuracy));
 
 
 