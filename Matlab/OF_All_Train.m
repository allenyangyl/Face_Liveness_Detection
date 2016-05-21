clear all, close all, clc

%% load libsvm
addpath('./libsvm-3.19/matlab');
TrainFeatureAll = [];
TrainTruth = [];

%% PRINT-ATTACK
load OF_SVM_PRINT_ATTACK_raw.mat
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

%% CASIA
load OF_SVM_CASIA_raw.mat
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

%% Training
length = size(TrainTruth, 1);
TrainFeatureAll(isnan(TrainFeatureAll)) = 0;
model = svmtrain(TrainTruth, TrainFeatureAll, '-t 0');
[TrainLabel, TrainAccuracy, TrainValue] = svmpredict(ones(length,1), TrainFeatureAll, model);
save OF_SVM_All.mat model