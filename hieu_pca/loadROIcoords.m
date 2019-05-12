% loadROIcoords
%
% load ROI coordinate information from subject data directory, adding it
% to the meta of the info/data/meta.
%
% creates meta.rois as a cell array where each cell has the properties
% 'name' and 'coords'.  'name' is the string name of the subdirectory
% corresponding to the ROI.  'coords' is a [numVoxels x 3] array giving
% the x,y,z coordinate for each voxel in the ROI.  
%
% Input:
%  - info,data,meta
%
% Output:
%  - info,data,meta, the new meta contains the new information.
%
% Example:
% [i,d,m]=loadROIcoords(i,d,m,'/shared/fmri/data/data-brainlex/08057/mroi/');
%
% History
%  - Oct 07,2005 Wei - redocument
%  - 9/13/02 Tom - Created.
%  - 11/21/02 Tom - made directory paths work under Windows and Linux

function [i,d,m] = loadROIcoords(i,d,m)
  
  information = mri_reference(m.study);
  ROIrootdir=fullfile(information.dataplace, m.study, m.subject,'mroi');
  dirs=dir(ROIrootdir);
  roinum=0;
  for j=1:1:length(dirs)
    % ignore directories beginning with '.', as well as the 'allbrain' ROI
    if (dirs(j).name(1)~='.') & ~strcmpi(dirs(j).name,'allbrain') & dirs(j).isdir 
      roinum=roinum+1;
      m.rois(roinum).name=dirs(j).name;
      ROIdir=fullfile(ROIrootdir,dirs(j).name,'allvoxels');
      %% IMPORTANT!!! Francisco says that each voxel coordinate in the
      % original data begins with 0, and he adds 1 to each coordinate so
      % that Matlab can instead have them all begin with the value 1.
      % Hence, we do the same here:
      m.rois(roinum).coords=load(ROIdir, '-ascii')+1;
    end
  end
  
  
