clear all, close all, clc

%% read client and imposter testing data name
addpath('./libsvm-3.19/matlab');
addpath('./lbp-0.3.3');
Map_u2_16 = getmapping(16, 'u2');
Map_u2_8 = getmapping(8, 'u2');

%% extract feature
detector = vision.CascadeObjectDetector('MinSize', [50,50]);
fileIndex = {'001', '002', '004', '006', '007', '008', '012', '016', '018', '025', '027', '103', '105', '108', '110'};
fileDirec = {'attack/fixed/attack_print_', 'attack/fixed/attack_print_', 'attack/hand/attack_print_', 'attack/hand/attack_print_', 'real/', 'real/', 'real/', 'real/'; ...
    'highdef_photo_adverse', 'highdef_photo_controlled', 'highdef_photo_adverse', 'highdef_photo_controlled', 'webcam_authenticate_adverse_1', 'webcam_authenticate_adverse_2', 'webcam_authenticate_controlled_1', 'webcam_authenticate_controlled_2'};
fileNo = size(fileIndex, 2);
TrainFeatureSet = cell(fileNo, 8);
for IdxSubject = 1 : fileNo
    for IdxData = 1 : 8
        Name = ['./PRINT-ATTACK/train/' fileDirec{1, IdxData} 'client' fileIndex{IdxSubject} '_session01_' fileDirec{2, IdxData} '.mov'];
        Mov = VideoReader(Name);
        NumFrame = Mov.NumberOfFrames;
        TrainFeature = [];
        for IdxFrame = 1 : NumFrame
            data = read(Mov, IdxFrame);
            box = step(detector, data);
            if size(box, 1) == 1
                frame_now = rgb2gray(imresize(data(box(2):box(2)+box(4), box(1):box(1)+box(3), :), [64,64]));
                Feature = [LBP_feature(frame_now, Map_u2_16, Map_u2_8), DoG(frame_now, 0.5, 1)'];
                TrainFeature = [TrainFeature;Feature];
            end
        end
        TrainFeatureSet{IdxSubject, IdxData} = TrainFeature;
        disp([num2str(IdxSubject) ', ' num2str(IdxData)])
        clear Mov;
    end
end
% save DoG&LBP_SVM_PRINT_ATTACK_raw.mat TrainFeatureSet

%% training
% load DoG&LBP_SVM_PRINT_ATTACK_raw.mat
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
MinMax = minmax(TrainFeatureAll')';
TrainFeatureAll = (TrainFeatureAll-kron(MinMax(1,:),ones(length,1)))./(kron(MinMax(2,:),ones(length,1))-kron(MinMax(1,:),ones(length,1)));
model = svmtrain(TrainTruth, TrainFeatureAll, '-t 0');
[TrainLabel, TrainAccuracy, TrainValue] = svmpredict(ones(length,1), TrainFeatureAll, model);
save DoG&LBP_SVM_PRINT_ATTACK.mat model MinMax
        

