% clear all, close all, clc
% 
% %% read client and imposter testing data name
% addpath('./libsvm-3.19/matlab');
% addpath('./HOOF');
% % load OF_SVM_CASIA.mat
% % load OF_SVM_PRINT_ATTACK.mat
% load OF_SVM_All.mat
% 
% %% extract feature
% detector = vision.CascadeObjectDetector('MinSize', [100,100]);
% flagSet = cell(30, 8);
% bins = 9;
% blocks = 10;
% for IdxSubject = 1 : 30
%     for IdxData = 1 : 8
%         if IdxData <= 8
%             Name = ['./CASIA/test_release/' num2str(IdxSubject) '/' num2str(IdxData) '.avi'];
%         else
%             Name = ['./CASIA/test_release/' num2str(IdxSubject) '/HR_' num2str(IdxData-8) '.avi'];
%         end
%         Mov = VideoReader(Name);
%         NumFrame = Mov.NumberOfFrames;
%         opticalFlow = vision.OpticalFlow('ReferenceFrameSource', 'Input Port', 'OutputValue', 'Horizontal and vertical components in complex form', 'Method', 'Lucas-Kanade');
%         flag = [];
%         frame_now = rgb2gray(read(Mov, 1));
%         for IdxFrame = 2 : NumFrame
%             frame_pre = frame_now;
%             frame_now = rgb2gray(read(Mov, IdxFrame));
%             box = step(detector, frame_now);
%             if size(box, 1) == 1
%                 OF = step(opticalFlow, double(frame_now), double(frame_pre));
%                 Feature = [];
%                 for iBlock = 1 : blocks
%                     for jBlock = 1 : blocks
%                         Feature = [Feature, gradientHistogram(...
%                             real(OF(round(box(2)+box(4)*(iBlock-1)/(blocks+1)):round(box(2)+box(4)*(iBlock+1)/(blocks+1)), round(box(1)+box(3)*(jBlock-1)/(blocks+1)):round(box(1)+box(3)*(jBlock+1)/(blocks+1)))), ...
%                             imag(OF(round(box(2)+box(4)*(iBlock-1)/(blocks+1)):round(box(2)+box(4)*(iBlock+1)/(blocks+1)), round(box(1)+box(3)*(jBlock-1)/(blocks+1)):round(box(1)+box(3)*(jBlock+1)/(blocks+1)))), ...
%                             bins)'];
%                     end
%                 end
%                 Feature(isnan(Feature)) = 0;
%                 TestFeature = Feature;
%                 [TestLabel, TestAccuracy, TestValue] = svmpredict(1, TestFeature, model, '-q');
%                 flag = [flag, TestValue];
%             end
%         end
%         flagSet{IdxSubject, IdxData} = flag;
%         disp([num2str(IdxSubject) ', ' num2str(IdxData) ', ' num2str(mean(flag))])
%         clear Mov;
%     end
% end
% % save OF_SVM_CASIA_testfeature.mat flagSet

%% testing
load OF_SVM_CASIA_testfeature.mat flagSet
DataCorrectRate = zeros(3, 1001);
CorrectRate = zeros(3, 1001);
for Bias = -5:0.01:5;
    CorrectNoReal = 0;
    CorrectNoFake = 0;
    TotalNoReal = 0;
    TotalNoFake = 0;
    DataCorrectNoReal = 0;
    DataCorrectNoFake = 0;
    for IdxSubject = 1 : 30
        for IdxData = 1 : 8
            if IdxData <= 2
                CorrectNoReal = CorrectNoReal + sum(flagSet{IdxSubject, IdxData}>=Bias);
                DataCorrectNoReal = DataCorrectNoReal + (mean(flagSet{IdxSubject, IdxData})>=Bias);
                TotalNoReal = TotalNoReal + size(flagSet{IdxSubject, IdxData}, 2);
            else
                CorrectNoFake = CorrectNoFake + sum(flagSet{IdxSubject, IdxData}<Bias);
                DataCorrectNoFake = DataCorrectNoFake + (mean(flagSet{IdxSubject, IdxData})<Bias);
                TotalNoFake = TotalNoFake + size(flagSet{IdxSubject, IdxData}, 2);
            end
        end
    end
    DataCorrectRate(:, round((Bias+5.01)*100)) = [(DataCorrectNoReal+DataCorrectNoFake) / 240; DataCorrectNoReal / 60; DataCorrectNoFake / 180];
    CorrectRate(:, round((Bias+5.01)*100)) = [(CorrectNoReal+CorrectNoFake) / (TotalNoReal+TotalNoFake); CorrectNoReal / TotalNoReal; CorrectNoFake / TotalNoFake];
end      