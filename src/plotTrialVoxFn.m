% plotTrialVoxFn(voxnum,x,y,z)
% when user clicks on a voxel displayed by plotTrial, this function is
% called with the voxel column number and its x,y,z coordinates.  At
% present, prints these coordinates to the user terminal.
%
% this function is invoked by plotTrial via the ButtonDownFcn associated
% with each subplot in the figure.
%
% another very useful-looking figure property is KeyPressFcn, which allows
% user to invoke command by pressing key, and the KeyPressFcn can see
% which key by querying the  CurrentCharacter property of the figure
%
% History
% 9/13/02 Tom - created.
%

function [voxnum,x,y,z] = plotTrialVoxFn(voxnum,x,y,z)

  % this print statement causes trouble... apparently it can't find 'm'
%  global m
%  fprintf('voxel %d.  x=%d, y=%d, z=%d.  ROI=%s\n',voxnum,x,y,z,...
%           char(m.colToROI(voxnum)));
  fprintf('voxel %d.  x=%d, y=%d, z=%d.\n',voxnum,x,y,z)

  % toggle width of voxel plot axes to let user know which one they clicked
  if get(gca,'linewidth')==0.5;
     set(gca,'linewidth',2.5);
  else
    set(gca,'linewidth',0.5);
  end

  % here's how to toggle the background color
%  if get(gca,'color')==[1 1 1] ;
%    set(gca,'color','cyan'); 
%  else
%    set(gca,'color','white'); 
%  end
