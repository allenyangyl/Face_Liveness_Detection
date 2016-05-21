clear all, close all, clc

%% read client and imposter testing data name
ClientTestNormalizedName = importdata('./NUAA/client_test_normalized.txt');
ClientTestNormalizedNumber = size(ClientTestNormalizedName, 1);
ImposterTestNormalizedName = importdata('./NUAA/imposter_test_normalized.txt');
ImposterTestNormalizedNumber = size(ImposterTestNormalizedName, 1);
addpath('./libsvm-3.19/matlab');
addpath('./lbp-0.3.3');
load('LBP_SVM.mat');
Map_u2_16 = getmapping(16, 'u2');
Map_u2_8 = getmapping(8, 'u2');
model_LBP = model;
MinMax_LBP = MinMax;
load DoG_SVM.mat
model_DoG = model;
MinMax_DoG = MinMax;

%% client testing data
% tic
ClientFlag = zeros(ClientTestNormalizedNumber, 1);
for i = 1 : ClientTestNormalizedNumber
    ClientTest = imread(['./NUAA/ClientNormalized/' ClientTestNormalizedName{i}]);
    ClientTestFeature_LBP = LBP_feature(ClientTest, Map_u2_16, Map_u2_8);
    ClientTestFeature_LBP = (ClientTestFeature_LBP-MinMax_LBP(1,:))./(MinMax_LBP(2,:)-MinMax_LBP(1,:));
    [~, ~, TestValue_LBP] = svmpredict(1, ClientTestFeature_LBP, model_LBP, '-q');
    ClientTestFeature_DoG = DoG(ClientTest, 0.5, 1);
    ClientTestFeature_DoG = (ClientTestFeature_DoG'-MinMax_DoG(1,:))./(MinMax_DoG(2,:)-MinMax_DoG(1,:));
    [~, ~, TestValue_DoG] = svmpredict(1, ClientTestFeature_DoG, model_DoG, '-q');
    ClientFlag(i) = TestValue_LBP+TestValue_DoG;
%     disp(['frame:',num2str(i),'; flag:',num2str(ClientFlag(i))]);
%     toc
end

%% LBP feature of imposter testing data
ImposterFlag = zeros(ImposterTestNormalizedNumber, 1);
for i = 1 : ImposterTestNormalizedNumber
    ImposterTest = imread(['./NUAA/ImposterNormalized/' ImposterTestNormalizedName{i}]);  
    ImposterTestFeature_LBP = LBP_feature(ImposterTest, Map_u2_16, Map_u2_8);
    ImposterTestFeature_LBP = (ImposterTestFeature_LBP-MinMax_LBP(1,:))./(MinMax_LBP(2,:)-MinMax_LBP(1,:));
    [~, ~, TestValue_LBP] = svmpredict(1, ImposterTestFeature_LBP, model_LBP, '-q');
    ImposterTestFeature_DoG = DoG(ImposterTest, 0.5, 1);
    ImposterTestFeature_DoG = (ImposterTestFeature_DoG'-MinMax_DoG(1,:))./(MinMax_DoG(2,:)-MinMax_DoG(1,:));
    [~, ~, TestValue_DoG] = svmpredict(1, ImposterTestFeature_DoG, model_DoG, '-q');
    ImposterFlag(i) = TestValue_LBP+TestValue_DoG;
%     disp(['frame:',num2str(i),'; flag:',num2str(ImposterFlag(i))]);
%     toc
end
