function Y = DoG(X, sigma1, sigma2)
%% default parameter
if nargin == 1
    sigma1 = 0.5;
    sigma2 = 1;
end
%% Init. operations
F1 = fspecial('gaussian', 2*ceil(3*sigma1)+1, sigma1);
F2 = fspecial('gaussian', 2*ceil(3*sigma2)+1, sigma2);

%% Filtering
XF1 = imfilter(X, F1, 'replicate', 'same');
XF2 = imfilter(X, F2, 'replicate', 'same');
DXF = XF1 - XF2;

%% Fourier Spectra
Y = abs(fftshift(fft2(DXF)));
Y = Y(:);
end