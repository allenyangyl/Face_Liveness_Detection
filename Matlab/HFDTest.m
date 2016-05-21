% clear all, close all, clc

%% read client and imposter testing data name
ClientTestNormalizedName = importdata('./NUAA/client_test_normalized.txt');
ClientTestNormalizedNumber = size(ClientTestNormalizedName, 1);
ImposterTestNormalizedName = importdata('./NUAA/imposter_test_normalized.txt');
ImposterTestNormalizedNumber = size(ImposterTestNormalizedName, 1);

figure;
color = 'brgymck';
ThresholdFrequency = [0, 10, 20, 50, 100, 200, 500];
for n = 1 : 7
    %% HFD of client Testing data
    ClientTestHFD = zeros(1, ClientTestNormalizedNumber);
    for i = 1 : ClientTestNormalizedNumber
        ClientTest = imread(['./NUAA/ClientNormalized/' ClientTestNormalizedName{i}]);
        ClientTestHFD(i) = HFD(ClientTest, ThresholdFrequency(n));
    end

    %% HFD of imposter Testing data
    ImposterTestHFD = zeros(1, ImposterTestNormalizedNumber);
    for i = 1 : ImposterTestNormalizedNumber
        ImposterTest = imread(['./NUAA/ImposterNormalized/' ImposterTestNormalizedName{i}]);  
        ImposterTestHFD(i) = HFD(ImposterTest, ThresholdFrequency(n));
    end

    %% ROC curve
    Thfd = 0 : 0.001 :0.5;
    sensitivity = [1, size(Thfd, 2)];
    specificity = [1, size(Thfd, 2)];
    for i = 1 : size(Thfd, 2)
        %% sensitivity and specificity
        true_positive = sum(ClientTestHFD >= Thfd(i));
        true_negative = sum(ImposterTestHFD < Thfd(i));
        false_positive = sum(ImposterTestHFD >= Thfd(i));
        false_negative = sum(ClientTestHFD < Thfd(i));
        sensitivity(i) = true_positive/(true_positive+false_negative);
        specificity(i) = true_negative/(false_positive+true_negative);
    end
    hold on
    plot(1-specificity, sensitivity, color(n))
end