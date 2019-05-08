% getConditionTimecourse(info,data,meta)
%
%  Returns a column vector containing the condition numbers at each
%  time instant throughout the IDM
%
% Input: 
%  - info, data, meta
%
% Output:
%  - conditionTimecourse, a column vector integers with length of total
%  numbers of snapshots.
%  
% History: 
%  - Oct 07,2005 Wei - redocument
%  - 8/16/02 Tom - created.  

function [ctc] = getConditionTimecourse(info,data,meta)
  tcLength = sum([info.len]);
  ctc=zeros(tcLength,1);
  beginPos=1;
  for i=1:1:meta.ntrials
    cond=info(i).cond;
    endPos=beginPos-1+info(i).len;
    ctc(beginPos:endPos)=cond;
    beginPos=endPos+1;
  end
