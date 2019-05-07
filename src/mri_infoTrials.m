% mri_infoTrials
%
% Gather information about number,type and length of trials
% Returns information about the IDM:
%  - minTrialLenCond - minimum trial length in each condition (indexed by condition #)
%  - ntrialsCond     - # of trials per condition (indexed by condition #)
%  - trialBegin/trialEnd - which portion of trial gets turned into examples
%
% WARNING: in construction - need to pack the return of this
% function, in a fashion similar to what occurs in mri_reference,
% which returns a struct
%
% Notes:
%
% Example:
%  - [ntrials,nvoxels,nconds,minTrialLenCond,ntrialsCond,trialBegin,...
%     trialEnd] = mri_infoTrials(info,data,meta);
%
% History:
%  - Oct 07,2005 Wei - redocument
%  - 22 Sep 02 - fp - in construction

function [ntrials,nvoxels,nconds,minTrialLenCond,ntrialsCond,trialBegin,trialEnd] = mri_infoTrials( varargin )

  l = length(varargin);
  info = varargin{1};
  data = varargin{2};
  meta = varargin{3};
  if l > 3
    study = varargin{4};
  else 
    study = meta.study;
  end
  
  ntrials = max(size(data));
  nvoxels = size(data{1},2);

  % figure out minimum trial lengths and # trials per condition
  for nt = 1:1:ntrials
    cond = info(nt).cond;

    if cond > 0
      switch study
       case {'data-syntamb2'}
	% WARNING - this is here until I get the pipeline to
	% discard conditions 6 and 7
	if cond < 6
	  minTrialLenCond(cond) = 1000;
	end	
       otherwise
	minTrialLenCond(cond) = 1000;
      end
    end
  end
  
  nconds      = length(minTrialLenCond);
  ntrialsCond = zeros(nconds,1);
  
  for nt = 1:1:ntrials
    len  = info(nt).len;
    cond = info(nt).cond;

    if cond > 0 & cond <= nconds
      ntrialsCond(cond) = ntrialsCond(cond) + 1;
      if len < minTrialLenCond(cond)
	minTrialLenCond(cond) = len;
      end    
    end
  end  

  % smallest trial in any condition 
  switch study
   case {'data-starplus-sp','data-starplus-ps'}
    trialBegin = ones(nconds,1) + 0;
    trialEnd   = min(minTrialLenCond',ones(nconds,1) + 36);
   case {'data-categories'}
    trialBegin = ones(nconds,1) + 6; % skip first 6 images
    trialEnd   = min(minTrialLenCond',ones(nconds,1) + 37);  
   case {'data-syntamb2'}
    trialBegin = ones(nconds,1) + 0; %
    trialEnd   = min(minTrialLenCond',ones(nconds,1) + 22);        
   otherwise
%    fprintf('WARNING: no support for trial Begin/End for the %s study\n',study);
    trialBegin = 1;
    trialEnd   = 1;
  end 
