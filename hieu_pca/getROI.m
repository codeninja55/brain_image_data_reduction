% getROI(meta,roiNameString)
% 
% Return the struct from meta.rois that corresponds to 'roiNameString'
%
% Input: 
%  - meta
%  - roiNameString, a string
% 
% Output:
%  - roi, a structure which contains the information about this roi
%
% Example: 
%  - roi=getROI(meta,'LT');
%
% History:  
%  - Oct 07,2005 Wei - redocument
%  - 2/19/03, TMM created
%

function [roi] = getROI(meta, roiNameStr)
  roi=0;
  for i=1:length(meta.rois)
    if strcmpi(roiNameStr,meta.rois(i).name)
      roi=meta.rois(i);
      return
    end
  end
  
      
  
