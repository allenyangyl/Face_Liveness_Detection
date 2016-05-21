clear all, close all, clc

%% read client and imposter training data name
ClientTrainNormalizedName = importdata('./NUAA/client_train_normalized.txt');
ClientTrainNormalizedNumber = size(ClientTrainNormalizedName, 1);
ImposterTrainNormalizedName = importdata('./NUAA/imposter_train_normalized.txt');
ImposterTrainNormalizedNumber = size(ImposterTrainNormalizedName, 1);
addpath('./libsvm-3.19/matlab');

%% DoG feature of client training data
ClientTrainFeature = zeros(ClientTrainNormalizedNumber, 64*64);
for i = 1 : ClientTrainNormalizedNumber
    ClientTrain = imread(['./NUAA/ClientNormalized/' ClientTrainNormalizedName{i}]);
    ClientTrainFeature(i, :) = DoG(ClientTrain, 0.5, 1);
end

%% DoG feature of imposter training data
ImposterTrainFeature = zeros(ImposterTrainNormalizedNumber, 64*64);
for i = 1 : ImposterTrainNormalizedNumber
    ImposterTrain = imread(['./NUAA/ImposterNormalized/' ImposterTrainNormalizedName{i}]);  
    ImposterTrainFeature(i, :) = DoG(ImposterTrain, 0.5, 1);
end

%% SVM training
MinMax = minmax([ClientTrainFeature;ImposterTrainFeature]')';
ClientTrainFeature = (ClientTrainFeature-kron(MinMax(1,:),ones(ClientTrainNormalizedNumber,1)))./(kron(MinMax(2,:),ones(ClientTrainNormalizedNumber,1))-kron(MinMax(1,:),ones(ClientTrainNormalizedNumber,1)));
ImposterTrainFeature = (ImposterTrainFeature-kron(MinMax(1,:),ones(ImposterTrainNormalizedNumber,1)))./(kron(MinMax(2,:),ones(ImposterTrainNormalizedNumber,1))-kron(MinMax(1,:),ones(ImposterTrainNormalizedNumber,1)));
Truth = [ones(ClientTrainNormalizedNumber,1);-ones(ImposterTrainNormalizedNumber,1)];
model = svmtrain(Truth, [ClientTrainFeature;ImposterTrainFeature], '-t 0');
[ClientTrainLabel, ClientTrainAccuracy, ClientTrainValue] = svmpredict(ones(ClientTrainNormalizedNumber,1), ClientTrainFeature, model);
[ImposterTrainLabel, ImposterTrainAccuracy, ImposterTrainValue] = svmpredict(-ones(ImposterTrainNormalizedNumber,1), ImposterTrainFeature, model);
save DoG_SVM_NUAA.mat model MinMax