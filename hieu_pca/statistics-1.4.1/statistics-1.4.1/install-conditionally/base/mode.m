## Copyright (C) 2007-2017 David Bateman
## Copyright (C) 2017 Nicholas R. Jankowski
##
## This program is free software: you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation, either version 3 of the
## License, or (at your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {} {} mode (@var{x})
## @deftypefnx {} {} mode (@var{x}, @var{dim})
## @deftypefnx {} {[@var{m}, @var{f}, @var{c}] =} mode (@dots{})
## Compute the most frequently occurring value in a dataset (mode).
##
## @code{mode} determines the frequency of values along the first non-singleton
## dimension and returns the value with the highest frequency.  If two, or
## more, values have the same frequency @code{mode} returns the smallest.
##
## If the optional argument @var{dim} is given, operate along this dimension.
##
## The return variable @var{f} is the number of occurrences of the mode in
## the dataset.
##
## The cell array @var{c} contains all of the elements with the maximum
## frequency.
## @seealso{mean, median}
## @end deftypefn

function [m, f, c] = mode (x, dim)

  if (nargin < 1 || nargin > 2)
    print_usage ();
  endif

  if (! (isnumeric (x) || islogical (x)))
    error ("mode: X must be a numeric vector or matrix");
  endif

  nd = ndims (x);
  sz = size (x);
  if (nargin < 2)
    ## Find the first non-singleton dimension.
    (dim = find (sz > 1, 1)) || (dim = 1);
  else
    if (! (isscalar (dim) && dim == fix (dim) && dim > 0))
      error ("mode: DIM must be an integer and a valid dimension");
    endif
  endif

  if (isempty (x))
    %% codepath for Matlab compatibility. empty x produces NaN output, but 
    %% m f and c shape depends on size of x.  
    if ((nargin == 1) && (nd == 2) && (sz == [0 0]))
      f = 0; %seems to always be a double even if x is single
      if (isa (x, "single"))
        m = single (NaN);
        c = {(single (NaN (0, 1)))};
      else
        m = NaN;
        c = {(NaN (0, 1))}; 
      endif
    else
       if (nargin == 2)
         sz(dim) = 1;
       else
         sz (find ((sz ~= 1), 1)) = 1;
       endif
      f = zeros (sz); %seems to always be a double even if x is single
      if (isa (x, "single"))
        m = single (NaN (sz));
        c = cell (sz);
        c(:) = {(single (NaN (1, 0)))};
      else
        m = NaN (sz);
        c = cell (sz);
        c(:) = {(NaN (1, 0))};
      endif
    endif
  
    return

  endif

  if (dim > nd)
    ## Special case of mode over non-existent dimension.
    m = x;
    f = ones (size (x));
    c = num2cell (x);
    return;
  endif

  sz2 = sz;
  sz2(dim) = 1;
  sz3 = ones (1, nd);
  sz3(dim) = sz(dim);

  if (issparse (x))
    t2 = sparse (sz(1), sz(2));
  else
    t2 = zeros (sz);
  endif

  if (dim != 1)
    perm = [dim, 1:dim-1, dim+1:nd];
    t2 = permute (t2, perm);
  endif

  xs = sort (x, dim);
  t = cat (dim, true (sz2), diff (xs, 1, dim) != 0);

  if (dim != 1)
    t2(permute (t != 0, perm)) = diff ([find(permute (t, perm))(:); prod(sz)+1]);
    f = max (ipermute (t2, perm), [], dim);
    xs = permute (xs, perm);
  else
    t2(t) = diff ([find(t)(:); prod(sz)+1]);
    f = max (t2, [], dim);
  endif

  c = cell (sz2);
  if (issparse (x))
    m = sparse (sz2(1), sz2(2));
  else
    m = zeros (sz2, class (x));
  endif
  for i = 1 : prod (sz2)
    c{i} = xs(t2(:, i) == f(i), i);
    m(i) = c{i}(1);
  endfor

endfunction


%!test
%! [m, f, c] = mode (toeplitz (1:5));
%! assert (m, [1, 2, 2, 2, 1]);
%! assert (f, [1, 2, 2, 2, 1]);
%! assert (c, {[1; 2; 3; 4; 5], [2], [2; 3], [2], [1; 2; 3; 4; 5]});
%!test
%! [m, f, c] = mode (toeplitz (1:5), 2);
%! assert (m, [1; 2; 2; 2; 1]);
%! assert (f, [1; 2; 2; 2; 1]);
%! assert (c, {[1; 2; 3; 4; 5]; [2]; [2; 3]; [2]; [1; 2; 3; 4; 5]});
%!test
%! a = sprandn (32, 32, 0.05);
%! sp0 = sparse (0);
%! [m, f, c] = mode (a);
%! [m2, f2, c2] = mode (full (a));
%! assert (m, sparse (m2));
%! assert (f, sparse (f2));
%! c_exp(1:length (a)) = { sp0 };
%! assert (c , c_exp);
%! assert (c2, c_exp);

%!assert (mode ([2, 3, 1, 2, 3, 4] ,1), [2, 3, 1, 2, 3, 4])
%!assert (mode ([2, 3, 1, 2, 3, 4], 2), 2)
%!assert (mode ([2, 3, 1, 2, 3, 4]), 2)
%!assert (mode (single ([2, 3, 1, 2, 3, 4])), single (2))
%!assert (mode (int8 ([2, 3, 1, 2, 3, 4])), int8 (2))

%!assert (mode ([2; 3; 1; 2; 3; 4], 1), 2)
%!assert (mode ([2; 3; 1; 2; 3; 4], 2), [2; 3; 1; 2; 3; 4])
%!assert (mode ([2; 3; 1; 2; 3; 4]), 2)

%!test
%! x = magic (3);
%! [m, f, c] = mode (x, 3);
%! assert (m, x);
%! assert (f, ones (3, 3));
%! assert (c, num2cell (x));

%!shared x
%! x(:,:,1) = toeplitz (1:3);
%! x(:,:,2) = circshift (toeplitz (1:3), 1);
%! x(:,:,3) = circshift (toeplitz (1:3), 2);
%!test
%! [m, f, c] = mode (x, 1);
%! assert (reshape (m, [3, 3]), [1 1 1; 2 2 2; 1 1 1]);
%! assert (reshape (f, [3, 3]), [1 1 1; 2 2 2; 1 1 1]);
%! c = reshape (c, [3, 3]);
%! assert (c{1}, [1; 2; 3]);
%! assert (c{2}, 2);
%! assert (c{3}, [1; 2; 3]);
%!test
%! [m, f, c] = mode (x, 2);
%! assert (reshape (m, [3, 3]), [1 1 2; 2 1 1; 1 2 1]);
%! assert (reshape (f, [3, 3]), [1 1 2; 2 1 1; 1 2 1]);
%! c = reshape (c, [3, 3]);
%! assert (c{1}, [1; 2; 3]);
%! assert (c{2}, 2);
%! assert (c{3}, [1; 2; 3]);
%!test
%! [m, f, c] = mode (x, 3);
%! assert (reshape (m, [3, 3]), [1 2 1; 1 2 1; 1 2 1]);
%! assert (reshape (f, [3, 3]), [1 2 1; 1 2 1; 1 2 1]);
%! c = reshape (c, [3, 3]);
%! assert (c{1}, [1; 2; 3]);
%! assert (c{2}, [1; 2; 3]);
%! assert (c{3}, [1; 2; 3]);

## Test input validation
%!error mode ()
%!error mode (1, 2, 3)
%!error <X must be a numeric> mode ({1 2 3})
%!error <DIM must be an integer> mode (1, ones (2,2))
%!error <DIM must be an integer> mode (1, 1.5)
%!error <DIM must be .* a valid dimension> mode (1, 0)

## Tests for empty input Matlab compatibility (bug #48690)
%!test
%! [m, f, c] = mode ([]);
%! assert (m, NaN);
%! assert (f, 0);
%! assert (c, {(NaN (0, 1))});
%!test
%! [m, f, c] = mode (single ([]));
%! assert (m, single (NaN));
%! assert (f, double (0));
%! assert (c, {(single (NaN (0, 1)))});
%!test
%! [m, f, c] = mode (ones (0, 1));
%! assert (m, NaN);
%! assert (f, 0);
%! assert (c, {(NaN (1, 0))});
%!test
%! [m, f, c] = mode (ones (1, 0));
%! assert (m, NaN);
%! assert (f, 0);
%! assert (c, {(NaN (1, 0))});
%!test
%! [m, f, c] = mode (single (ones (1, 0)));
%! assert (m, single (NaN));
%! assert (f, double (0));
%! assert (c, {(single (NaN (1, 0)))});
%!test
%! [m, f, c] = mode (ones (2, 0));
%! assert (m, NaN (1, 0));
%! assert (f, zeros (1, 0));
%! assert (c, cell (1, 0));
%!test
%! [m, f, c] = mode (ones (0, 2));
%! assert (m, NaN (1, 2));
%! assert (f, zeros (1, 2));
%! assert (c, {(NaN (1, 0)),(NaN (1, 0))});
%!test
%! [m, f, c] = mode (single (ones (2, 0)));
%! assert (m, single (NaN (1, 0)));
%! assert (f, zeros (1, 0));
%! assert (c, cell (1, 0));
%!test
%! [m, f, c] = mode (ones (0, 0, 1, 0));
%! assert (m, NaN (1, 0, 1, 0));
%! assert (f, zeros (1, 0, 1, 0));
%! assert (c, cell (1, 0, 1, 0));
%!test
%! [m, f, c] = mode (ones (1, 1, 1, 0));
%! assert (m, NaN (1, 1));
%! assert (f, zeros (1, 1));
%! assert (c, {(NaN (1, 0))});

%!assert (mode (ones (0, 0, 0, 0)), NaN (1, 0, 0, 0))
%!assert (mode (ones (0, 0, 0, 1)), NaN (1, 0, 0, 1))
%!assert (mode (ones (0, 0, 0, 2)), NaN (1, 0, 0, 2))
%!assert (mode (ones (0, 0, 1, 0)), NaN (1, 0, 1, 0))
%!assert (mode (ones (0, 0, 1, 1)), NaN (1, 1, 1, 1))
%!assert (mode (ones (0, 0, 1, 2)), NaN (1, 0, 1, 2))
%!assert (mode (ones (0, 0, 2, 0)), NaN (1, 0, 2, 0))
%!assert (mode (ones (0, 0, 2, 1)), NaN (1, 0, 2, 1))
%!assert (mode (ones (0, 0, 2, 2)), NaN (1, 0, 2, 2))
%!assert (mode (ones (0, 1, 0, 0)), NaN (1, 1, 0, 0))
%!assert (mode (ones (0, 1, 0, 1)), NaN (1, 1, 0, 1))
%!assert (mode (ones (0, 1, 0, 2)), NaN (1, 1, 0, 2))
%!assert (mode (ones (0, 1, 1, 0)), NaN (1, 1, 1, 0))
%!assert (mode (ones (0, 1, 1, 1)), NaN (1, 1, 1, 1))
%!assert (mode (ones (0, 1, 1, 2)), NaN (1, 1, 1, 2))
%!assert (mode (ones (0, 1, 2, 0)), NaN (1, 1, 2, 0))
%!assert (mode (ones (0, 1, 2, 1)), NaN (1, 1, 2, 1))
%!assert (mode (ones (0, 1, 2, 2)), NaN (1, 1, 2, 2))
%!assert (mode (ones (0, 2, 0, 0)), NaN (1, 2, 0, 0))
%!assert (mode (ones (0, 2, 0, 1)), NaN (1, 2, 0, 1))
%!assert (mode (ones (0, 2, 0, 2)), NaN (1, 2, 0, 2))
%!assert (mode (ones (0, 2, 1, 0)), NaN (1, 2, 1, 0))
%!assert (mode (ones (0, 2, 1, 1)), NaN (1, 2, 1, 1))
%!assert (mode (ones (0, 2, 1, 2)), NaN (1, 2, 1, 2))
%!assert (mode (ones (0, 2, 2, 0)), NaN (1, 2, 2, 0))
%!assert (mode (ones (0, 2, 2, 1)), NaN (1, 2, 2, 1))
%!assert (mode (ones (0, 2, 2, 2)), NaN (1, 2, 2, 2))
%!assert (mode (ones (1, 0, 0, 0)), NaN (1, 1, 0, 0))
%!assert (mode (ones (1, 0, 0, 1)), NaN (1, 1, 0, 1))
%!assert (mode (ones (1, 0, 0, 2)), NaN (1, 1, 0, 2))
%!assert (mode (ones (1, 0, 1, 0)), NaN (1, 1, 1, 0))
%!assert (mode (ones (1, 0, 1, 1)), NaN (1, 1, 1, 1))
%!assert (mode (ones (1, 0, 1, 2)), NaN (1, 1, 1, 2))
%!assert (mode (ones (1, 0, 2, 0)), NaN (1, 1, 2, 0))
%!assert (mode (ones (1, 0, 2, 1)), NaN (1, 1, 2, 1))
%!assert (mode (ones (1, 0, 2, 2)), NaN (1, 1, 2, 2))
%!assert (mode (ones (1, 1, 0, 0)), NaN (1, 1, 1, 0))
%!assert (mode (ones (1, 1, 0, 1)), NaN (1, 1, 1, 1))
%!assert (mode (ones (1, 1, 0, 2)), NaN (1, 1, 1, 2))
%!assert (mode (ones (1, 1, 1, 0)), NaN (1, 1, 1, 1))
%!assert (mode (ones (1, 1, 2, 0)), NaN (1, 1, 1, 0))
%!assert (mode (ones (1, 2, 0, 0)), NaN (1, 1, 0, 0))
%!assert (mode (ones (1, 2, 0, 1)), NaN (1, 1, 0, 1))
%!assert (mode (ones (1, 2, 0, 2)), NaN (1, 1, 0, 2))
%!assert (mode (ones (1, 2, 1, 0)), NaN (1, 1, 1, 0))
%!assert (mode (ones (1, 2, 2, 0)), NaN (1, 1, 2, 0))
%!assert (mode (ones (2, 0, 0, 0)), NaN (1, 0, 0, 0))
%!assert (mode (ones (2, 0, 0, 1)), NaN (1, 0, 0, 1))
%!assert (mode (ones (2, 0, 0, 2)), NaN (1, 0, 0, 2))
%!assert (mode (ones (2, 0, 1, 0)), NaN (1, 0, 1, 0))
%!assert (mode (ones (2, 0, 1, 1)), NaN (1, 0, 1, 1))
%!assert (mode (ones (2, 0, 1, 2)), NaN (1, 0, 1, 2))
%!assert (mode (ones (2, 0, 2, 0)), NaN (1, 0, 2, 0))
%!assert (mode (ones (2, 0, 2, 1)), NaN (1, 0, 2, 1))
%!assert (mode (ones (2, 0, 2, 2)), NaN (1, 0, 2, 2))
%!assert (mode (ones (2, 1, 0, 0)), NaN (1, 1, 0, 0))
%!assert (mode (ones (2, 1, 0, 1)), NaN (1, 1, 0, 1))
%!assert (mode (ones (2, 1, 0, 2)), NaN (1, 1, 0, 2))
%!assert (mode (ones (2, 1, 1, 0)), NaN (1, 1, 1, 0))
%!assert (mode (ones (2, 1, 2, 0)), NaN (1, 1, 2, 0))
%!assert (mode (ones (2, 2, 0, 0)), NaN (1, 2, 0, 0))
%!assert (mode (ones (2, 2, 0, 1)), NaN (1, 2, 0, 1))
%!assert (mode (ones (2, 2, 0, 2)), NaN (1, 2, 0, 2))
%!assert (mode (ones (2, 2, 1, 0)), NaN (1, 2, 1, 0))
%!assert (mode (ones (2, 2, 2, 0)), NaN (1, 2, 2, 0))
%!assert (mode (ones (1, 1, 0, 0, 0)), NaN (1, 1, 1, 0, 0))
%!assert (mode (ones (1, 1, 1, 1, 0)), NaN (1, 1, 1, 1, 1))
%!assert (mode (ones (2, 1, 1, 1, 0)), NaN (1, 1, 1, 1, 0))
%!assert (mode (ones (1, 2, 1, 1, 0)), NaN (1, 1, 1, 1, 0))

## Tests for empty input with DIM call
%!test
%! [m, f, c] = mode ([], 1);
%! assert (m, NaN (1,0));
%! assert (f, zeros (1,0));
%! assert (c, cell (1,0));

%!test
%! [m, f, c] = mode ([], 2);
%! assert (m, NaN (0,1));
%! assert (f, zeros (0,1));
%! assert (c, cell (0,1));

%!test
%! [m, f, c] = mode ([], 3);
%! assert (m, []);
%! assert (f, []);
%! assert (c, {});

%!test
%! [m, f, c] = mode (single([]), 1);
%! assert (m, single (NaN (1,0)));
%! assert (f, double (zeros (1,0)));
%! assert (c, cell (1,0));

%!test
%! [m, f, c] = mode (single([]), 2);
%! assert (m, single (NaN (0,1)));
%! assert (f, double (zeros (0,1)));
%! assert (c, cell (0,1));

%!test
%! [m, f, c] = mode (single([]), 3);
%! assert (m, single([]));
%! assert (f, []);
%! assert (c, {});

%!test
%! [m, f, c] = mode (ones(1,0), 1);
%! assert (m, NaN (1,0));
%! assert (f, zeros (1,0));
%! assert (c, cell (1,0));

%!test
%! [m, f, c] = mode (ones(1,0), 2);
%! assert (m, NaN);
%! assert (f, 0);
%! assert (c, {(NaN (1,0))});

%!test
%! [m, f, c] = mode (ones(1,0), 3);
%! assert (m, NaN (1,0));
%! assert (f, zeros (1,0));
%! assert (c, cell (1,0));

%!test
%! [m, f, c] = mode (ones(0,1), 1);
%! assert (m, NaN);
%! assert (f, 0);
%! assert (c, {(NaN (1,0))});

%!test
%! [m, f, c] = mode (ones(0,1), 2);
%! assert (m, NaN (0,1));
%! assert (f, zeros (0,1));
%! assert (c, cell (0,1));

%!test
%! [m, f, c] = mode (ones(0,1), 3);
%! assert (m, NaN (0,1));
%! assert (f, zeros (0,1));
%! assert (c, cell (0,1));

%!test
%! [m, f, c] = mode (ones(0,0,1), 1);
%! assert (m, NaN (1,0));
%! assert (f, zeros (1,0));
%! assert (c, cell (1,0));

%!test
%! [m, f, c] = mode (ones(0,0,1), 2);
%! assert (m, NaN (0,1));
%! assert (f, zeros (0,1));
%! assert (c, cell (0,1));

%!test
%! [m, f, c] = mode (ones(0,0,1), 3);
%! assert (m, []);
%! assert (f, []);
%! assert (c, {});

%!test
%! [m, f, c] = mode (ones(1,0,1), 1);
%! assert (m, NaN (1,0));
%! assert (f, zeros (1,0));
%! assert (c, cell (1,0));

%!test
%! [m, f, c] = mode (ones(1,0,1), 2);
%! assert (m, NaN);
%! assert (f, 0);
%! assert (c, {(NaN (1,0))});

%!test
%! [m, f, c] = mode (ones(1,0,1), 3);
%! assert (m, NaN (1,0));
%! assert (f, zeros (1,0));
%! assert (c, cell (1,0));
