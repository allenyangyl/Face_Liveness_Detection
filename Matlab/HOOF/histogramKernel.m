function [ output ] = histogramKernel( hist1, hist2 )
%HISTOGRAMKERNEL find the 'histogram kernel' between time series of
%histograms
%  This test code implements a so-called histogram intersection kernel for
%  histogram time series data by extending the original histogram kernel.
%
% Parameters:
%       hist1:  first normalized to 1 histogram of size d1 bins x N1 samples
%       hist2:  second normalized to 1 histogram of size d2 bins x N2 samples
% Outputs:
%       output: the value of the histogram kernel between two histogram
%       time series
%
% (c) Rizwan Chaudhry - JHU Vision Lab

[d1, N1] = size(hist1);
[d2, N2] = size(hist2);

if d1 ~= d2
    error('Number of bins in the histograms must be the same');
end

% TODO: check for normalization

N = min(N1,N2);

output = sum(sum(min(hist1(:,1:N), hist2(:,1:N)),1))/N;