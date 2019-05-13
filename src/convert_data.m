%%  
% load the data (assume you're using version 7 of Matlab.  If you're using
% version 6, load the -v6.mat file instead)
load('data\data-starplus-04847-v7.mat')

% add the functions as paths to MATLAB
addpath('fmri', 'fmri/Netlab')

% examine the variables info, data, and meta, and read their description in
% section 2 below.
meta
info
data

% select non-noisey trials
[i,d,m] = transformIDM_selectTrials(info,data,meta,find([info.cond]~=0)); 

% create training data
[examples,labels,expInfo] = idmToExamples_condLabel(i,d,m);
save("input.mat", "examples", "labels", "-v6");
fprintf('Data saved as input.mat\n')