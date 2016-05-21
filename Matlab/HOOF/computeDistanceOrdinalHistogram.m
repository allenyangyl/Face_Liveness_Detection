function dist = computeDistanceOrdinalHistogram(hist1, hist2)
% dist = computeDistanceOrdinalHistogram(hist1, hist2)
%
% Computes the distance between two ordinal histograms. Ordinal histograms are histograms formed from 
% ordinal data such as linear intervals etc
%
% Implementation of the MDPA (Minimum Distance of Pair Assignments) algorithm in
% S. -H. Cha, S.N. Srihari, On measuring the distance between histograms, Pattern Recognition, 35 (2002) 1355-1370
%
% (c) Rizwan Chaudhry - JHU Vision Lab

% Assuming both histograms have the same number of bins

histDiff = hist1-hist2;
prefixsum = cumsum(histDiff);
h_dist = sum(abs(prefixsum));

dist = h_dist;