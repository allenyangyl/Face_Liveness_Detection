% clear all, close all, clc
% 
% %% read client and imposter testing data name
% addpath('./libsvm-3.19/matlab');
% addpath('./lbp-0.3.3');
% Map_u2_16 = getmapping(16, 'u2');
% Map_u2_8 = getmapping(8, 'u2');
% load DoG&LBP_SVM_PRINT_ATTACK.mat
% 
% %% extract feature
% detector = vision.CascadeObjectDetector('MinSize', [50,50]);
% fileIndex = {'009', '011', '013', '014', '019', '020', '021', '023', '024', '026', '028', '031', '102', '104', '106', '107', '109', '112', '115', '117'};
% fileDirec = {'attack/fixed/attack_print_', 'attack/fixed/attack_print_', 'attack/hand/attack_print_', 'attack/hand/attack_print_', 'real/', 'real/', 'real/', 'real/'; ...
%     'highdef_photo_adverse', 'highdef_photo_controlled', 'highdef_photo_adverse', 'highdef_photo_controlled', 'webcam_authenticate_adverse_1', 'webcam_authenticate_adverse_2', 'webcam_authenticate_controlled_1', 'webcam_authenticate_controlled_2'};
% fileNo = size(fileIndex, 2);
% flagSet = cell(fileNo, 8);
% for IdxSubject = 1 : fileNo
%     for IdxData = 1 : 8
%         Name = ['./PRINT-ATTACK/test/' fileDirec{1, IdxData} 'client' fileIndex{IdxSubject} '_session01_' fileDirec{2, IdxData} '.mov'];
%         Mov = VideoReader(Name);
%         NumFrame = Mov.NumberOfFrames;
%         flag = [];
%         for IdxFrame = 1 : NumFrame
%             data = read(Mov, IdxFrame);
%             box = step(detector, data);
%             if size(box, 1) == 1
%                 frame_now = rgb2gray(imresize(data(box(2):box(2)+box(4), box(1):box(1)+box(3), :), [64,64]));
%                 TestFeature = [LBP_feature(frame_now, Map_u2_16, Map_u2_8), DoG(frame_now, 0.5, 1)'];
%                 TestFeature = (TestFeature-MinMax(1,:))./(MinMax(2,:)-MinMax(1,:));
%                 [TestLabel, TestAccuracy, TestValue] = svmpredict(1, TestFeature, model, '-q');
%                 flag = [flag,TestValue];
%             end
%         end
%         flagSet{IdxSubject, IdxData} = flag;
%         disp([num2str(IdxSubject) ', ' num2str(IdxData)])
%         clear Mov;
%     end
% end
% save DoG&LBP_SVM_PRINT_ATTACK_testfeature.mat flagSet

%% testing
load DoG&LBP_SVM_PRINT_ATTACK_testfeature.mat flagSet
DataCorrectRate = zeros(3, 601);
CorrectRate = zeros(3, 601);
for Bias = -3:0.01:3;
    CorrectNoReal = 0;
    CorrectNoFake = 0;
    TotalNoReal = 0;
    TotalNoFake = 0;
    DataCorrectNoReal = 0;
    DataCorrectNoFake = 0;
    for IdxSubject = 1 : 20
        for IdxData = 1 : 8
            if IdxData > 4
                CorrectNoReal = CorrectNoReal + sum(flagSet{IdxSubject, IdxData}>=Bias);
                DataCorrectNoReal = DataCorrectNoReal + (mean(flagSet{IdxSubject, IdxData})>=Bias);
                TotalNoReal = TotalNoReal + size(flagSet{IdxSubject, IdxData}, 2);
            else
                CorrectNoFake = CorrectNoFake + sum(flagSet{IdxSubject, IdxData}<Bias);
                DataCorrectNoFake = DataCorrectNoFake + (mean(flagSet{IdxSubject, IdxData})<Bias);
                TotalNoFake = TotalNoFake + size(flagSet{IdxSubject, IdxData}, 2);
            end
        end
    end
    DataCorrectRate(:, round((Bias+3.01)*100)) = [(DataCorrectNoReal+DataCorrectNoFake) / 160; DataCorrectNoReal / 80; DataCorrectNoFake / 80];
    CorrectRate(:, round((Bias+3.01)*100)) = [(CorrectNoReal+CorrectNoFake) / (TotalNoReal+TotalNoFake); CorrectNoReal / TotalNoReal; CorrectNoFake / TotalNoFake];
end