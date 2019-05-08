% localInformation
%
% provide the detail information about the dataset. But this function for
% the machine learning course project will not do the same thing as it
% supposed to do. Because we knew what data we provide, here only produced
% the necessary output for the functions who called this one.
%
% History:
% - Oct 10, 2005 - Wei rewrote
% - 17 Mar 2005 - fpereira - adapted from old mri_reference.m

function [information] = localInformation(varargin)
  
  l = length(varargin);
  if l < 1; fprintf('syntax: localInformation(<study name>)\n'); return;
  else
    study = varargin{1};
  end

  information.host = '';
  information.dataplace  = '';
  information.codeplace  = '';
  information.subjects   = '';
  information.rois       = '';
  information.dimensions = [64 64 8];
  information.nconds     = '';
