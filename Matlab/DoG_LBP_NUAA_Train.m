clear all, close all, clc

%% read client and imposter training data name
ClientTrainNormalizedName = importdata('./NUAA/client_train_normalized.txt');
ClientTrainNormalizedNumber = size(ClientTrainNormalizedName, 1);
ImposterTrainNormalizedName = importdata('./NUAA/imposter_train_normalized.txt');
ImposterTrainNormalizedNumber = size(ImposterTrainNormalizedName, 1);
addpath('./libsvm-3.19/matlab');
addpath('./lbp-0.3.3');

%% mapping
Map_u2_16 = getmapping(16, 'u2');
Map_u2_8 = getmapping(8, 'u2');

%% LBP feature of client training data
ClientTrainFeature = zeros(ClientTrainNormalizedNumber, 833+64*64);
for i = 1 : ClientTrainNormalizedNumber
    ClientTrain = imread(['./NUAA/ClientNormalized/' ClientTrainNormalizedName{i}]);
    ClientTrainFeature(i, :) = [LBP_feature(ClientTrain, Map_u2_16, Map_u2_8), DoG(ClientTrain, 0.5, 1)'];
end

%% LBP feature of imposter training data
ImposterTrainFeature = zeros(ImposterTrainNormalizedNumber, 833+64*64);
for i = 1 : ImposterTrainNormalizedNumber
    ImposterTrain = imread(['./NUAA/ImposterNormalized/' ImposterTrainNormalizedName{i}]);  
    ImposterTrainFeature(i, :) = [LBP_feature(ImposterTrain, Map_u2_16, Map_u2_8), DoG(ImposterTrain, 0.5, 1)'];
end

%% SVM training
MinMax = minmax([ClientTrainFeature;ImposterTrainFeature]')';
ClientTrainFeature = (ClientTrainFeature-kron(MinMax(1,:),ones(ClientTrainNormalizedNumber,1)))./(kron(MinMax(2,:),ones(ClientTrainNormalizedNumber,1))-kron(MinMax(1,:),ones(ClientTrainNormalizedNumber,1)));
ImposterTrainFeature = (ImposterTrainFeature-kron(MinMax(1,:),ones(ImposterTrainNormalizedNumber,1)))./(kron(MinMax(2,:),ones(ImposterTrainNormalizedNumber,1))-kron(MinMax(1,:),ones(ImposterTrainNormalizedNumber,1)));
Truth = [ones(ClientTrainNormalizedNumber,1);-ones(ImposterTrainNormalizedNumber,1)];
model = svmtrain(Truth, [ClientTrainFeature;ImposterTrainFeature], '-t 0');
[ClientTrainLabel, ClientTrainAccuracy, ClientTrainValue] = svmpredict(ones(ClientTrainNormalizedNumber,1), ClientTrainFeature, model);
[ImposterTrainLabel, ImposterTrainAccuracy, ImposterTrainValue] = svmpredict(-ones(ImposterTrainNormalizedNumber,1), ImposterTrainFeature, model);
save DoG&LBP_SVM_NUAA.mat model MinMax