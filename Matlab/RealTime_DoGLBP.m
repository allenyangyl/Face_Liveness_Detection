clear all, close all, clc

%% camera
adaptorname = 'winvideo';
info = imaqhwinfo(adaptorname);
deviceID = info.DeviceIDs{1};
format = info.DeviceInfo(deviceID).SupportedFormats{2};
video = videoinput(adaptorname, deviceID);
%set(video, 'ReturnedColorSpace', 'rgb');
vidRes = get(video, 'VideoResolution');
nBands = get(video, 'NumberOfBands'); 
hImage = image( zeros(vidRes(2), vidRes(1), nBands) ); 
preview(video, hImage); 

%% SVM classifier
addpath('./libsvm-3.19/matlab');
addpath('./lbp-0.3.3');
load('DoG&LBP_SVM_All.mat');
Map_u2_16 = getmapping(16, 'u2');
Map_u2_8 = getmapping(8, 'u2');

%% test
detector = vision.CascadeObjectDetector('MinSize', [64,64]);
bias = 0.5;
threshold = 5;
max_frame = 1000;
i = 0;
flag = zeros(1, max_frame);
tic;
while(1)    
    i = i + 1;
    data = getsnapshot(video);
    box = step(detector, data);
    if size(box, 1) >= 1
        imshow(data);
        frame_now = rgb2gray(imresize(data(box(1,2):box(1,2)+box(1,4), box(1,1):box(1,1)+box(1,3), :), [64,64]));
        rectangle('Position',box(1,:),'LineWidth',4,'EdgeColor','r');
        TestFeature = [LBP_feature(frame_now, Map_u2_16, Map_u2_8), DoG(frame_now, 0.5, 1)'];
        TestFeature = (TestFeature-MinMax(1,:))./(MinMax(2,:)-MinMax(1,:));
        [TestLabel, TestAccuracy, TestValue] = svmpredict(1, TestFeature, model, '-q');
        flag = [flag(2:size(flag,2)), TestValue+bias];
        disp(['frame:',num2str(i),'; time:',num2str(toc),'; flag:',num2str(TestValue+bias)]);
        text(20, 20, {['Frame: ',num2str(i)];['Time: ',num2str(toc),'s'];['Flag: ',num2str(TestValue+bias)];['TotalFlag: ',num2str(sum(flag))]}, 'Color', 'r', 'FontSize', 12, 'VerticalAlignment', 'top');
    end
    if abs(sum(flag)) > threshold || i > max_frame;
        break;
    end
end
if sum(flag)>0
    disp('real face');
    text(box(1,1)+box(1,3)/2, box(1,2), 'Real Face', 'Color', 'r', 'FontSize', 30, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
else
    disp('fake face');
    text(box(1,1)+box(1,3)/2, box(1,2), 'Fake Face', 'Color', 'r', 'FontSize', 30, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end