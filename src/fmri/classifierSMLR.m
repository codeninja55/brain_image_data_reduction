% classifierSMLR
%
% Train a SMLR classifier
% 
% In:
% - trainSet       a matrix real values with dimension of
% (#trainexamples*#features)
% - trainLabels    a column vector of integers with length of
% (#trainexamples) 
% - classifierParameters    a cell, which contains three elements: 
%    first- initial stepsize 
%    second - stopCriterion
%    third - penalty coefficient. 
%   If this cell is empty, the defaults are:
%      stepsize      = 0.01;
%      stopCriterion = 0.001;
%      penalty       = 10;
%
% Out:
% - classifier - contains learnt classifier
%
% Dep:
%
% History: 
%  - Oct 07,2005 Wei - redocument
%  - Mar 20, 2005  Wei - created  
%    optimization was adapted from Francisco Pereira.
%
% Known bugs:
%
% Ex:
% - [models]=classifierSMLR(examples,labels,{0.01,0.001,10})
%
% Reference: 
% - Machine learning by Tom Mitchell
%   This funciton uses the steepest descent with the automated stopping
%   rule as the optimization method.  

function [models] = classifierSMLR( varargin )
  
  l = length(varargin);
  if l < 3
    fprintf('syntax: classifierSMLR(trainSet,trainLabels, parameters)\n');
    return;
  elseif l > 3
    fprintf('syntax: classifierSMLR(trainSet, trainLabels, parameters)');
    return;
  end
  
  
  trainSet    = varargin{1};
  trainLabels = varargin{2};
  classifierParameters = varargin{3};
  
  [trainSet, mean_trainLabels] = correlation_train(trainSet, trainLabels);
  [models] = classifierSMLR_core(trainSet, trainLabels, classifierParameters);
  [models] = changeWeight(models, mean_trainLabels);
  
  
function [new_d, mean_d] = correlation_train(data,labels)
    conds=sort(unique(labels));
    numConds=length(conds);
    [numRows, numCols]=size(data);
    mean_d=zeros(numConds,numCols);
    for j=1:numRows
        k=find(conds==labels(j));
        mean_d(k,:)=mean_d(k,:)+data(j,:)/sum(labels==labels(j));
    end
    
    new_d = zeros(numRows, numCols);
    for j=1:numRows
        k=find(conds==labels(j));
        new_d(j,:) = (data(j,:)-mean(data(j,:))).*(mean_d(k,:)-mean(mean_d(k,:)))/(std(data(j,:))*std(mean_d(k,:)));
    end
        
function [models_new] = changeWeight(models, mean_d)
    
    lastpos           = size(models,1);
    weights=models{lastpos-2};
    [nFeatures, nClasses]=size(weights);
    
    mean_d=[ones(size(mean_d,1),1) mean_d];
    c=mean_d';
    
    c=c(:,1:nClasses);
    weights_new=zeros(nFeatures,nClasses);
    for j=1:nClasses
        weights_new(:,j)=(c(:,j)-mean(c(:,j))).*weights(:,j)/std(c(:,j));
    end
    models_new = models;
    models_new{lastpos-2}=weights_new;
    models_new{lastpos-1}.uncombinedWeights=weights;


