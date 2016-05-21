clear all, close all, clc

ClientTrainNormalizedName = importdata('./NUAA/client_train_normalized.txt');
ClientTrainNormalizedNumber = size(ClientTrainNormalizedName, 1);
ImposterTrainNormalizedName = importdata('./NUAA/imposter_train_normalized.txt');
ImposterTrainNormalizedNumber = size(ImposterTrainNormalizedName, 1);
ClientTestNormalizedName = importdata('./NUAA/client_test_normalized.txt');
ClientTestNormalizedNumber = size(ClientTestNormalizedName, 1);
ImposterTestNormalizedName = importdata('./NUAA/imposter_test_normalized.txt');
ImposterTestNormalizedNumber = size(ImposterTestNormalizedName, 1);

for i = 1 : ClientTrainNormalizedNumber
    ClientTrain = imread(['./NUAA/ClientNormalized/' ClientTrainNormalizedName{i}]);
    imwrite(ClientTrain, ['.\NUAA\train\real\' num2str(i-1, '%05d') '.png']);
end

for i = 1 : ImposterTrainNormalizedNumber
    ImposterTrain = imread(['./NUAA/ImposterNormalized/' ImposterTrainNormalizedName{i}]);  
    imwrite(ImposterTrain, ['.\NUAA\train\fake\' num2str(i-1, '%05d') '.png']);
end

for i = 1 : ClientTestNormalizedNumber
    ClientTest = imread(['./NUAA/ClientNormalized/' ClientTestNormalizedName{i}]);
    imwrite(ClientTest, ['.\NUAA\test\real\' num2str(i-1, '%05d') '.png']);
end

count_fake = 0;
for i = 1 : ImposterTestNormalizedNumber
    ImposterTest = imread(['./NUAA/ImposterNormalized/' ImposterTestNormalizedName{i}]);  
    imwrite(ImposterTest, ['.\NUAA\test\fake\' num2str(i-1, '%05d') '.png']);
end