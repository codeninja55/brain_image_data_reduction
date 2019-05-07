% plotVoxelColor
%
%  Plot the time course for the selected voxel, concatenating all
%  trials.  Plot each condition in a different color (for up to five
%  conditions).
%
%  Color key: 
%   Condition 0: red
%   Condition 1: green
%   Condition 2: blue
%   Condition 3: magenta
%   Condition 4: cyan
%   Condition 5+: black
%
% plotVoxel(info,data,meta,x,y,z,<titlestring>)
%   x,y,z : coordinates of the voxel to be plotted
%   titlestring : optional text string to be printed with plots.
%
% or
%
% plotVoxel(info,data,meta,<column number>,<titlestring>)
%
% Input:
%  - info,data,meta
%  - x,y,z, coordinate of the interesting voxel
%  - title, a string
%  or
%  - info,data,meta
%  - n, the nth voxels
%  - title, a string
%
% Output:
%  - none.
%
% Example:
%  - plotVoxelColor(i,d,m,46,54,10)  % plots time course for voxel <46,54,10>
%  or 
%  - plotVoxelColor(i,d,m,34);
%  
% History: 
%  - Oct 07,2005 Wei - redocument
%  - 8/16/02 tom - created. (based on plotTrial)
%  - 10/16/02 fp - adapted to accept a column number as an argument
%  - 8/27/04 jp - adapted to vary color based on condition

function [] = plotVoxelColor( varargin )

  % process arguments
  l = length(varargin);
  if l < 4
    fprintf('syntax:\nplotVoxel(<info>,<data>,<meta>,<x>,<y>,<z>,[titlestring])\nor\nplotVoxel(<info>,<data>,<meta>,<voxel column>,[titlestring])\n'); 
    return;
  end

  info  = varargin{1};
  data  = varargin{2};
  meta  = varargin{3};
  titlestring   = '';
  
  if l > 5  
    x = varargin{4};
    y = varargin{5};
    z = varargin{6};  
    if l > 6
      titlestring = varargin{7};
    end
  else
    column = varargin{4};
    if l > 4
      titlestring = varargin{5};
    end
    
    x = meta.colToCoord(column,1);
    y = meta.colToCoord(column,2);
    z = meta.colToCoord(column,3);
  end
    
  % delete graphics in the current figure window
  clf reset;
  hold on;
  
  vtc = getVoxelTimecourse(info,data,meta,x,y,z);
  tcount=1;

  %iterate through trials, plotting data in color
  for idx=1:meta.ntrials
    tnew=tcount+info(idx).len-1;
    if(tnew>=length(vtc))
        tnew=length(vtc);
    end
        
    switch info(idx).cond
        case 0
            plot(tcount:tnew,vtc(tcount:tnew),'r');
        case 1
            plot(tcount:tnew,vtc(tcount:tnew),'g');
        case 2
            plot(tcount:tnew,vtc(tcount:tnew),'b');
        case 3
            plot(tcount:tnew,vtc(tcount:tnew),'m');
        case 4
            plot(tcount:tnew,vtc(tcount:tnew),'c');
        otherwise
            plot(tcount:tnew,vtc(tcount:tnew),'k');
    end
    hold on;
    tcount = tnew;
  end
  hold on;
  tstr=sprintf('%s %s subject%s, region %s,', titlestring,meta.study,meta.subject,meta.roi);
  tstr=sprintf('%s\n <x,y,z>=<%d,%d,%d> ', tstr,x,y,z);
  conditionstr= sprintf(' %0.5g', [info.cond]);
  tstr=sprintf('%s condition sequence = %s', tstr, conditionstr);
  title(tstr);
  
  

