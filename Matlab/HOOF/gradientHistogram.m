function ohog = gradientHistogram(Fx,Fy,binSize)
% Compute HOOF feature
% INPUTS
%   Fx      - X-flow 
%   Fy	    - Y-flow
%   binSize - number of bins used
%
% OUTPUTS
%   ohog    - output histogram of oriented optical flow
%
% EXAMPLE
%

%% Written by  : Rizwan Chaudhry and Avinash Ravichandran
%% $DATE       : 28-Aug-2008 11:00:58 $
%% $Revision   : 1.00 $
%% Matlab      : 7.4.0.287 (R2007a)
%% FILENAME    : gradientHistogram.m
%
% (c) Rizwan Chaudhry, Avinash Ravichandran - JHU Vision Lab

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%gradientHistogram.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


magnitudeImage   = (Fx.^2 + Fy.^2 ).^0.5;
orientationImage =  atan2(Fy,Fx);

greaterPiBy2Index = orientationImage > pi/2;
smallerMinusPiBy2Index = orientationImage < -pi/2;
remainingIndex = orientationImage <=pi/2 & orientationImage >= -pi/2;

greaterPiBy2Mat = greaterPiBy2Index.*orientationImage;
smallerMinusPiBy2Mat = smallerMinusPiBy2Index.*orientationImage;
remainingMat = remainingIndex.*orientationImage;

piMat = pi*ones(size(orientationImage));

convertGreaterPiBy2Mat = greaterPiBy2Index.*piMat - greaterPiBy2Mat;
convertSmallerMinusPiBy2Mat = smallerMinusPiBy2Index.*(-piMat) - smallerMinusPiBy2Mat;

newOrientationImage = convertGreaterPiBy2Mat + remainingMat + convertSmallerMinusPiBy2Mat;


% [hog,idx] =
% histc(reshape(orientationImage,1,[]),linspace(-pi,pi,binSize+1) );
[hog,idx] = histc(reshape(newOrientationImage,1,[]),linspace(-pi/2,pi/2,binSize+1) );
values = reshape(magnitudeImage,1,[]);

for k=1:binSize
    bin(k) = sum(values(find(idx==k)));
end

ohog = bin/sum(bin);

ohog = ohog';
