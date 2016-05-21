%% High Frequency Descriptor
function HFD = HFD(Image, ThresholdFrequency, ImageSize, HighFrequencyRadius)
%% default parameter
if nargin < 4
    if nargin < 3
        if nargin < 2
            ThresholdFrequency = 100;
        end
        ImageSize = 64;
    end
    HighFrequencyRadius = ImageSize / 3;
end
%% fourier spectra
ClientTrainFFT = abs(fftshift(fft2(Image)));
%% energy of high frequencies
HighFrequencyEnergy = 0;
for x = 1 : ImageSize
    for y = 1 : ImageSize
        if ((x-(1+ImageSize)/2)^2+(y-(1+ImageSize)/2)^2)>HighFrequencyRadius^2 && ClientTrainFFT(x,y)>=ThresholdFrequency
            HighFrequencyEnergy = HighFrequencyEnergy + ClientTrainFFT(x,y);
        end
    end
end
%% energy of all frequencies
TotalEnergy = sum(ClientTrainFFT(:)) - sum(Image(:));
%% high frequency descriptor
HFD = HighFrequencyEnergy / TotalEnergy;
end