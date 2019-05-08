% computeCorrelation
%
% Computes correlation between two matrices, column by column
%
% Input:
%  - two matrices with the same # of columns
%
% Outputs:
%  - a vector with the # of columns, containing the correlation for each
%
% Examples:
%  - columnCorr=computeCorrelation(example1,example2)
%
% History:
%  - Oct 07,2005 Wei - redocument
%  - created - no clear
%

function [columnCorrelation] = computeCorrelation(A,B)

ncolumns = size(A,2);
if size(B) ~= size(A);
  fprintf('syntax: computeCorrelation(A,B)\n');
  fprintf('- A,B are matrices with the same # of columns\n');
  return;
end

AB   = A.*B;
muA  = mean(A,1);
sA   = std(A,1,1);
muB  = mean(B,1);
sB   = std(B,1,1);
muAB = mean(AB,1);

columnCorrelation = (muAB - muA.*muB) ./ (sA.*sB);

%columnSlow = zeros(1,ncolumns);
%for c = 1:ncolumns
%  m = corrcoef(A(:,c),B(:,c));
%  columnSlow(c) = m(1,2);
%end
  
