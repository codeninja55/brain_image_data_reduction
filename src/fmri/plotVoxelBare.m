%  Simplified version of PlotVoxel that does not plot condition numbers
%  and does not construct automatic title strings. 
%
% plotVoxelBare(info,data,meta,x,y,z,<titlestring>)
%   x,y,z : coordinates of the voxel to be plotted
%   titlestring : optional text string to be printed with plots. No text
%   will be added to this titlestring
%
% or
%
% plotVoxel(info,data,meta,<column number>,<titlestring>)
%
% Example:
%   plotVoxel(i,d,m,46,54,10)  % plots time course for voxel <46,54,10>
%  
% History: 
% 2/11/03  tom - created. (based on plotVoxel)

function [gca] = plotVoxelBare( varargin )

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
  
  vtc = getVoxelTimecourse(info,data,meta,x,y,z);

  hp=plot(vtc);
  
  title(titlestring);
  % place title at bottom center inside the plot (gca gives plot handle)
  xcenter=mean(get(gca,'XLim'));
  yrange=get(gca,'YLim');
  ybottom=yrange(1);
  set(get(gca,'Title'),'Position', [xcenter, ybottom, 1.0])
  

