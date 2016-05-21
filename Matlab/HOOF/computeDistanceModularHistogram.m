function dist = computeDistanceModularHistogram(hist1, hist2)

% dist = computeDistanceModularHistogram(hist1, hist2)
%
% Computes the distance between two modular histograms. Modular histograms are histograms formed from 
% modular data such as angles etc
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

dist_increased = 0;

while dist_increased == 0
    d = min(prefixsum(prefixsum > 0));
    if isempty(d) == 1
        break;
    end
    tempprefixsum = prefixsum - repmat(d,size(prefixsum));
    h_dist2 = sum(abs(tempprefixsum));
    if h_dist2 < h_dist
        h_dist = h_dist2;
        prefixsum = tempprefixsum;
    else
        dist_increased = 1;
    end
end
dist_increased = 0;
while dist_increased == 0
    d = max(prefixsum(prefixsum < 0));
    if isempty(d) == 1
        break;
    end
    tempprefixsum = prefixsum - repmat(d,size(prefixsum));
    h_dist2 = sum(abs(tempprefixsum));
    if h_dist2 < h_dist
        h_dist = h_dist2;
        prefixsum = tempprefixsum;
    else
        dist_increased = 1;
    end
end

dist = h_dist;