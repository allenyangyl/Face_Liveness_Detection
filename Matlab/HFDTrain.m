clear all, close all, clc

%% read client and imposter training data name
ClientTrainNormalizedName = importdata('./NUAA/client_train_normalized.txt');
ClientTrainNormalizedNumber = size(ClientTrainNormalizedName, 1);
ImposterTrainNormalizedName = importdata('./NUAA/imposter_train_normalized.txt');
ImposterTrainNormalizedNumber = size(ImposterTrainNormalizedName, 1);
addpath('./libsvm-3.19/matlab');

figure;
color = 'brgymck';
ThresholdFrequency = [0, 10, 20, 50, 100, 200, 500];
for n = 1 : 7
    %% HFD of client training data
    ClientTrainHFD = zeros(1, ClientTrainNormalizedNumber);
    for i = 1 : ClientTrainNormalizedNumber
        ClientTrain = imread(['./NUAA/ClientNormalized/' ClientTrainNormalizedName{i}]);
        ClientTrainHFD(i) = HFD(ClientTrain, ThresholdFrequency(n));
    end

    %% HFD of imposter training data
    ImposterTrainHFD = zeros(1, ImposterTrainNormalizedNumber);
    for i = 1 : ImposterTrainNormalizedNumber
        ImposterTrain = imread(['./NUAA/ImposterNormalized/' ImposterTrainNormalizedName{i}]);  
        ImposterTrainHFD(i) = HFD(ImposterTrain, ThresholdFrequency(n));
    end

    %% ROC curve
    Thfd = 0 : 0.001 :0.5;
    sensitivity = [1, size(Thfd, 2)];
    specificity = [1, size(Thfd, 2)];
    for i = 1 : size(Thfd, 2)
        %% sensitivity and specificity
        true_positive = sum(ClientTrainHFD >= Thfd(i));
        true_negative = sum(ImposterTrainHFD < Thfd(i));
        false_positive = sum(ImposterTrainHFD >= Thfd(i));
        false_negative = sum(ClientTrainHFD < Thfd(i));
        sensitivity(i) = true_positive/(true_positive+false_negative);
        specificity(i) = true_negative/(false_positive+true_negative);
    end
    hold on
    plot(1-specificity, sensitivity, color(n))
end