% clear all, close all, clc
% 
% %% read client and imposter testing data name
% addpath('./libsvm-3.19/matlab');
% addpath('./HOOF');
% 
% %% extract feature
% detector = vision.CascadeObjectDetector('MinSize', [50,50]);
% fileIndex = {'001', '002', '004', '006', '007', '008', '012', '016', '018', '025', '027', '103', '105', '108', '110'};
% fileDirec = {'attack/fixed/attack_print_', 'attack/fixed/attack_print_', 'attack/hand/attack_print_', 'attack/hand/attack_print_', 'real/', 'real/', 'real/', 'real/'; ...
%     'highdef_photo_adverse', 'highdef_photo_controlled', 'highdef_photo_adverse', 'highdef_photo_controlled', 'webcam_authenticate_adverse_1', 'webcam_authenticate_adverse_2', 'webcam_authenticate_controlled_1', 'webcam_authenticate_controlled_2'};
% fileNo = size(fileIndex, 2);
% TrainFeatureSet = cell(fileNo, 8);
% bins = 9;
% blocks = 10;
% for IdxSubject = 1 : fileNo
%     for IdxData = 1 : 8
%         Name = ['./PRINT-ATTACK/train/' fileDirec{1, IdxData} 'client' fileIndex{IdxSubject} '_session01_' fileDirec{2, IdxData} '.mov'];
%         Mov = VideoReader(Name);
%         NumFrame(IdxSubject, IdxData) = Mov.NumberOfFrames;
%         opticalFlow = vision.OpticalFlow('ReferenceFrameSource', 'Input Port', 'OutputValue', 'Horizontal and vertical components in complex form', 'Method', 'Lucas-Kanade');
%         TrainFeature = [];
%         frame_now = rgb2gray(read(Mov, 1));
%         for IdxFrame = 2 : NumFrame(IdxSubject, IdxData)
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
%                 TrainFeature = [TrainFeature;Feature];
%             end
%         end
%         TrainFeatureSet{IdxSubject, IdxData} = TrainFeature;
%         disp([num2str(IdxSubject) ', ' num2str(IdxData)])
%         clear Mov;
%     end
% end
% save OF_SVM_PRINT_ATTACK_raw.mat TrainFeatureSet

%% training
load OF_SVM_PRINT_ATTACK_raw.mat
TrainFeatureAll = [];
TrainTruth = [];
for IdxSubject = 1 : 15
    for IdxData = 1 : 8
        TrainFeatureAll = [TrainFeatureAll;TrainFeatureSet{IdxSubject,IdxData}];
        if IdxData <= 4
            TrainTruth = [TrainTruth;-ones(size(TrainFeatureSet{IdxSubject,IdxData},1),1)];
        else
            TrainTruth = [TrainTruth;ones(size(TrainFeatureSet{IdxSubject,IdxData},1),1)];
        end
    end
    disp(num2str(IdxSubject));
end
length = size(TrainTruth, 1);
TrainFeatureAll(isnan(TrainFeatureAll)) = 0;
% MinMax = minmax(TrainFeatureAll')';
% TrainFeatureAll = (TrainFeatureAll-kron(MinMax(1,:),ones(length,1)))./(kron(MinMax(2,:),ones(length,1))-kron(MinMax(1,:),ones(length,1)));
model = svmtrain(TrainTruth, TrainFeatureAll, '-t 0');
[TrainLabel, TrainAccuracy, TrainValue] = svmpredict(ones(length,1), TrainFeatureAll, model);
save OF_SVM_PRINT_ATTACK.mat model