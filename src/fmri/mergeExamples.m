% mergeExamples
%
% Merge two example sets
%
% Input:
%  - examples, labels, and expInfo for two dataset 
%
% Output:
%  - the joint examples, labels, and expInfo
%
% Notes:
%  - The experiment types must match
%  - The numbers of features must be same for each example set.
%
% Examples:
%  - [examples, labels, expInfo] = mergeExamples(examples1, labels1, expInfo1, examples2, labels2, expInfo2)
%
% History:
%  - Oct 07,2005 Wei - redocument
%  - 21 Oct 2002, Xuerui - created
%
%

function [examples, labels, expInfo] = mergeExamples(varargin)

l = length(varargin);
if l < 6
    fprintf(1,'syntax: mergeExamples(examples1, labels1, expInfo1, examples2, labels2, expInfo2)');
    examples = []; labels = [];
    return;
else
    examples1 = varargin{1};
    labels1 = varargin{2};
    expInfo1 = varargin{3};
    examples2 = varargin{4};
    labels2 = varargin{5};
    expInfo2 = varargin{6};
end

expInfo = expInfo1;
examples = [examples1' examples2']';
labels = [labels1' labels2']';
expInfo.meta.ntrials = expInfo1.meta.ntrials + expInfo2.meta.ntrials;