function Y = LBP_feature(X, Map_u2_16, Map_u2_8)
%% histogram
Hist_u2_16_2=lbp(X, 2, 16, Map_u2_16,'h');
Hist_u2_8_2=lbp(X, 2, 8, Map_u2_8,'h');
Hist_u2_8_1_1=lbp(X(1:26,1:26), 1, 8, Map_u2_8,'h');   
Hist_u2_8_1_2=lbp(X(1:26,20:45), 1, 8, Map_u2_8,'h'); 
Hist_u2_8_1_3=lbp(X(1:26,39:64), 1, 8, Map_u2_8,'h'); 
Hist_u2_8_1_4=lbp(X(20:45,1:26), 1, 8, Map_u2_8,'h');   
Hist_u2_8_1_5=lbp(X(20:45,20:45), 1, 8, Map_u2_8,'h'); 
Hist_u2_8_1_6=lbp(X(20:45,39:64), 1, 8, Map_u2_8,'h'); 
Hist_u2_8_1_7=lbp(X(39:64,1:26), 1, 8, Map_u2_8,'h');   
Hist_u2_8_1_8=lbp(X(39:64,20:45), 1, 8, Map_u2_8,'h'); 
Hist_u2_8_1_9=lbp(X(39:64,39:64), 1, 8, Map_u2_8,'h'); 
Y = [Hist_u2_16_2, Hist_u2_8_2, Hist_u2_8_1_1, Hist_u2_8_1_2, Hist_u2_8_1_3, Hist_u2_8_1_4, Hist_u2_8_1_5, Hist_u2_8_1_6, Hist_u2_8_1_7, Hist_u2_8_1_8, Hist_u2_8_1_9];
end