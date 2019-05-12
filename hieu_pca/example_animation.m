% example_animation
%
% working example illustrating animation and other data visualization
%
% History:
%  - Oct 07,2005 Wei - rewrite


% load the data.  returns cell arrays of IDMs
load data-starplus-04847-v7.mat;

% plot an image showing the brain activity for the 7th snapshot in trial 3,
% displaying the z=6 slice
n=7; trialNum=3; z=6; 
plotSnapshot(info,data,meta,trialNum,z,n,0,0);

% plot the entire time series for trial 3, for each voxel in each brain slice
plotTrial(info,data,meta,3)

% watch a movie of the brain activity for brain slice z=6 of trial 3
M=animateTrial(info,data,meta,trialNum,z);

% watch it again
movie(M);

% watch a movie of the brain activity for six z slices, begining with z=6, of
% trial 3
M=animate6Trial(info,data,meta,trialNum,z);


% watch a movie of the brain activity for all 16 z slices of
% trial 3
M=animate16Trial(info,data,meta,3);

% erase the big movie data structure
clear M;
