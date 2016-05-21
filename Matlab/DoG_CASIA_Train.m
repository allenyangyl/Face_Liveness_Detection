clear all, close all, clc

%% read client and imposter testing data name
addpath('./libsvm-3.19/matlab');
addpath('./lbp-0.3.3');
Map_u2_16 = getmapping(16, 'u2');
Map_u2_8 = getmapping(8, 'u2');

%% extract features
detector = vision.CascadeObjectDetector('MinSize', [100,100]);
TrainFeatureSet = cell(20, 8);
NumFrame = zeros(20, 8);
for IdxSubject = 1 : 20
    for IdxData = 1 : 8
        if IdxData <= 8
            Name = ['./CASIA/train_release/' num2str(IdxSubject) '/' num2str(IdxData) '.avi'];
        else
            Name = ['./CASIA/train_release/' num2str(IdxSubject) '/HR_' num2str(IdxData-8) '.avi'];
        end
        Mov = VideoReader(Name);
        NumFrame(IdxSubject, IdxData) = Mov.NumberOfFrames;
        TrainFeature = [];
        for IdxFrame = 1 : NumFrame(IdxSubject, IdxData)
            data = read(Mov, IdxFrame);
            box = step(detector, data);
            if size(box, 1) == 1
                frame_now = rgb2gray(imresize(data(box(2):box(2)+box(4), box(1):box(1)+box(3), :), [64,64]));
                Feature = DoG(frame_now, 0.5, 1)';
                TrainFeature = [TrainFeature;Feature];
            end
        end
        TrainFeatureSet{IdxSubject, IdxData} = TrainFeature;
        disp([num2str(IdxSubject) ', ' num2str(IdxData)])
        clear Mov;
    end
end
save DoG_SVM_CASIA_raw.mat TrainFeatureSet

%% training
load DoG_SVM_CASIA_raw.mat
TrainFeatureAll = [];
TrainTruth = [];
for IdxSubject = 1 : 20
    for IdxData = 1 : 8
        TrainFeatureAll = [TrainFeatureAll;TrainFeatureSet{IdxSubject,IdxData}];
        if IdxData <= 2
            TrainTruth = [TrainTruth;ones(size(TrainFeatureSet{IdxSubject,IdxData},1),1)];
        else
            TrainTruth = [TrainTruth;-ones(size(TrainFeatureSet{IdxSubject,IdxData},1),1)];
        end
    end
    disp(num2str(IdxSubject));
end
length = size(TrainTruth, 1);
MinMax = minmax(TrainFeatureAll')';
TrainFeatureAll = (TrainFeatureAll-kron(MinMax(1,:),ones(length,1)))./(kron(MinMax(2,:),ones(length,1))-kron(MinMax(1,:),ones(length,1)));
model = svmtrain(TrainTruth, TrainFeatureAll, '-t 0');
[TrainLabel, TrainAccuracy, TrainValue] = svmpredict(ones(length,1), TrainFeatureAll, model);
save DoG_SVM_CASIA.mat model MinMax