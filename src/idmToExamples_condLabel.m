% idmToExamples_condLabel
%
% Convert info, data, meta structure to example, lables structure, which
% can be used in training and applying a classifier.
%
% Input:
%  - info, data,meta
%
% Output:
%   examples -- a matrix of real values with dimension of (# of instance)*(# of features)
%   labels -- a column vector of integers with length of (# of instance),
%   whose integers are the condition number in each trial.
%   expInfo -- a structure, which contains the extra information.
%
% Dependencies:
%
% Notes:
%  - produced data and labels contain the trials with condition number 0.
% History:
%  - Oct 07,2005 Wei - redocument
%  - Sep 12, 2005 - Wei, not ignore 0 conditions
%  - Mar 02, 2005 - Wei created
%
% Example:
%  - [examples,labels,expInfo]=idmToExamples_condLabel(info,data,meta);
%


function [examples,labels,expInfo] = idmToExamples_condLabel( varargin )
  
  l = length(varargin);
  if l < 3
    fprintf(1,'syntax: idmToExamples_condLabel(info,data,meta)');
    examples = []; labels = [];
    return;
  else
    info = varargin{1};
    data = varargin{2};
    meta = varargin{3};
  end
  
  % gather information about number,type and length of trials
  ntrials=length(info);
  nvoxels=size(data{1},2);
  trials=[info.cond];
  uniqueConds=sort(unique(trials));
  nconds=length(uniqueConds);
  trialLenCond=zeros(ntrials,nconds)+Inf;
  ntrialsCond=zeros(1,nconds);
  for c=1:ntrials
      tmp=find(uniqueConds==trials(c));
      trialLenCond(c,tmp)=size(data{c},1);
      ntrialsCond(tmp)=ntrialsCond(tmp)+1;
  end
  minTrialLenCond=min(trialLenCond,[],1);

  minTrialLen = min(minTrialLenCond);
  nfeatures = minTrialLen * nvoxels;
  nexamples = ntrials;  

  examples  = zeros(nexamples,nfeatures);
  labels    = trials';

  expInfo.experiment   = '';
  expInfo.dataType     = '';
  expInfo.meta         = meta;
  expInfo.featureToIDM = zeros(nfeatures,2);
  ftoidm1=repmat((1:nvoxels)',1,minTrialLen);
  ftoidm2=[];
  for m=1:minTrialLen
      ftoidm2(:,m)=ones(nvoxels,1)*m;
  end
  expInfo.featureToIDM(:,1)=reshape(ftoidm1,nfeatures,1);
  expInfo.featureToIDM(:,2)=reshape(ftoidm2,nfeatures,1);
  
  % 
  % Finally, create the examples
  %
  
  for j=1:ntrials
      tmpdata=data{j}(1:minTrialLen,:);
      tmpdata=reshape(tmpdata',nvoxels*minTrialLen,1);
      examples(j,:)=tmpdata';
  end
  
function test()
    info(1).cond=4;info(2).cond=2;info(3).cond=3;info(4).cond=4;info(5).cond=0;
    meta=[];
    data{1}=[1 2;3 4;5 6];
    data{2}=[0 0 ;data{1}+1.5];
    data{3}=[6 7; 8 9];
    data{4}=[4 5;6 7];
    data{5}=[1 2;3 4;5 6];
    [e,l,inf]=idmToExamples_condLabel(info,data,meta);
    
    %l=[4 2 3 4 0]';
    %d=[1 2 3 4;
    %   0 0 2.5 3.5
    %   6 7 8 9
    %   4 5 6 7
    %   1 2 3 4];
    
    