clear all, close all, clc

detector = vision.CascadeObjectDetector('MinSize', [50,50]);
fileIndex = {'001', '002', '004', '006', '007', '008', '012', '016', '018', '025', '027', '103', '105', '108', '110'};
fileDirec = {'attack/fixed/attack_print_', 'attack/fixed/attack_print_', 'attack/hand/attack_print_', 'attack/hand/attack_print_', 'real/', 'real/', 'real/', 'real/'; ...
    'highdef_photo_adverse', 'highdef_photo_controlled', 'highdef_photo_adverse', 'highdef_photo_controlled', 'webcam_authenticate_adverse_1', 'webcam_authenticate_adverse_2', 'webcam_authenticate_controlled_1', 'webcam_authenticate_controlled_2'};
fileNo = size(fileIndex, 2);
count_real = 0;
count_fake = 0;
for IdxSubject = 1 : fileNo
    for IdxData = 1 : 8
        Name = ['./PRINT-ATTACK/train/' fileDirec{1, IdxData} 'client' fileIndex{IdxSubject} '_session01_' fileDirec{2, IdxData} '.mov'];
        Mov = VideoReader(Name);
        NumFrame = Mov.NumberOfFrames;
        for IdxFrame = 1 : NumFrame
            data = read(Mov, IdxFrame);
            box = step(detector, data);
            if size(box, 1) == 1
                frame_now = data(box(2):box(2)+box(4), box(1):box(1)+box(3), :);
                if IdxData > 4
                    imwrite(frame_now, ['.\PRINT-ATTACK\train_\real\' num2str(count_real, '%05d') '.png']);
                    count_real = count_real + 1;
                else
                    imwrite(frame_now, ['.\PRINT-ATTACK\train_\fake\' num2str(count_fake, '%05d') '.png']);
                    count_fake = count_fake + 1;
                end
            end
        end
        disp([num2str(IdxSubject) ', ' num2str(IdxData)])
        clear Mov;
    end
end