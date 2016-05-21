clear all, close all, clc

detector = vision.CascadeObjectDetector('MinSize', [100,100]);
count_real = 0;
count_fake = 0;
for IdxSubject = 1 : 30
    for IdxData = 1 : 8
        if IdxData <= 8
            Name = ['./CASIA/test_release/' num2str(IdxSubject) '/' num2str(IdxData) '.avi'];
        else
            Name = ['./CASIA/test_release/' num2str(IdxSubject) '/HR_' num2str(IdxData-8) '.avi'];
        end
        Mov = VideoReader(Name);
        for IdxFrame = 1 : Mov.NumberOfFrames
            data = read(Mov, IdxFrame);
            box = step(detector, data);
            if size(box, 1) == 1
                frame_now = data(box(2):box(2)+box(4), box(1):box(1)+box(3), :);
                if IdxData <= 2
                    imwrite(frame_now, ['.\CASIA\test\real\' num2str(count_real, '%05d') '.png']);
                    count_real = count_real + 1;
                else
                    imwrite(frame_now, ['.\CASIA\test\fake\' num2str(count_fake, '%05d') '.png']);
                    count_fake = count_fake + 1;
                end
            end
        end
        disp([num2str(IdxSubject) ', ' num2str(IdxData)])
        clear Mov;
    end
end