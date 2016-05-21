clear all, close all, clc

%% camera
adaptorname = 'winvideo';
info = imaqhwinfo(adaptorname);
deviceID = info.DeviceIDs{1};
format = info.DeviceInfo(deviceID).SupportedFormats{5};
video = videoinput(adaptorname, deviceID, format);
vidRes = get(video, 'VideoResolution');
nBands = get(video, 'NumberOfBands'); 
hImage = image( zeros(vidRes(2), vidRes(1), nBands) ); 
preview(video, hImage); 

%% SVM classifier
addpath('./libsvm-3.19/matlab');
addpath('./HOOF');
load('OF_SVM_All.mat');

%% test
opticalFlow = vision.OpticalFlow('ReferenceFrameSource', 'Input Port', 'OutputValue', 'Horizontal and vertical components in complex form', 'Method', 'Lucas-Kanade');
detector = vision.CascadeObjectDetector('MinSize', [64,64]);
bins = 9;
blocks = 10;
bias = 0;
threshold = 5;
max_frame = 1000;
i = 0;
flag = zeros(1, 50);
tic;
while(1)
    i = i + 1;
    frame_now = rgb2gray(getsnapshot(video));    
    box = step(detector, frame_now);
    if size(box, 1) == 1 && i > 1
        OF = step(opticalFlow, double(frame_now), double(frame_pre));
        HOOF_Feature = HOOF(OF(box(2):box(2)+box(4), box(1):box(1)+box(3), :), bins, blocks);
        [TestLabel, TestAccuracy, TestValue] = svmpredict(1, HOOF_Feature, model, '-q');
        flag = [flag(2:size(flag,2)), TestValue+bias];
        disp(['frame:',num2str(i),'; time:',num2str(toc),'; flag:',num2str(TestValue+bias)]);
        rectangle('Position',box(1,:),'LineWidth',4,'EdgeColor','r');
        text(20, 20, {['Frame: ',num2str(i)];['Time: ',num2str(toc),'s'];['Flag: ',num2str(TestValue+bias)];['TotalFlag: ',num2str(sum(flag))]}, 'Color', 'r', 'FontSize', 12, 'VerticalAlignment', 'top');
    end
    frame_pre = frame_now;    
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