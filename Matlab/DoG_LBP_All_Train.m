clear all, close all, clc

%% add path
addpath('./libsvm-3.19/matlab');
addpath('./lbp-0.3.3');

%% NUAA
% load
ClientTrainNormalizedName = importdata('./NUAA/client_train_normalized.txt');
ClientTrainNormalizedNumber = size(ClientTrainNormalizedName, 1);
ImposterTrainNormalizedName = importdata('./NUAA/imposter_train_normalized.txt');
ImposterTrainNormalizedNumber = size(ImposterTrainNormalizedName, 1);
% mapping
Map_u2_16 = getmapping(16, 'u2');
Map_u2_8 = getmapping(8, 'u2');
% features of client training data
ClientTrainFeature = zeros(ClientTrainNormalizedNumber, 833+64*64);
for i = 1 : ClientTrainNormalizedNumber
    ClientTrain = imread(['./NUAA/ClientNormalized/' ClientTrainNormalizedName{i}]);
    ClientTrainFeature(i, :) = [LBP_feature(ClientTrain, Map_u2_16, Map_u2_8), DoG(ClientTrain, 0.5, 1)'];
end
% features of imposter training data
ImposterTrainFeature = zeros(ImposterTrainNormalizedNumber, 833+64*64);
for i = 1 : ImposterTrainNormalizedNumber
    ImposterTrain = imread(['./NUAA/ImposterNormalized/' ImposterTrainNormalizedName{i}]);  
    ImposterTrainFeature(i, :) = [LBP_feature(ImposterTrain, Map_u2_16, Map_u2_8), DoG(ImposterTrain, 0.5, 1)'];
end
% features and truths
TrainTruth = [ones(ClientTrainNormalizedNumber,1);-ones(ImposterTrainNormalizedNumber,1)];
TrainFeatureAll = [ClientTrainFeature;ImposterTrainFeature];
clear ClientTrainFeature ImposterTrainFeature

%% CASIA
load DoG&LBP_SVM_CASIA_raw.mat
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
clear TrainFeatureSet

%% PRINT-ATTACK
load DoG&LBP_SVM_PRINT_ATTACK_raw.mat
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
clear TrainFeatureSet

%% training
length = size(TrainTruth, 1);
MinMax = minmax(TrainFeatureAll')';
TrainFeatureAll = (TrainFeatureAll-kron(MinMax(1,:),ones(length,1)))./(kron(MinMax(2,:),ones(length,1))-kron(MinMax(1,:),ones(length,1)));
model = svmtrain(TrainTruth, TrainFeatureAll, '-t 0');
[TrainLabel, TrainAccuracy, TrainValue] = svmpredict(ones(length,1), TrainFeatureAll, model);
save DoG&LBP_SVM_All.mat model MinMax