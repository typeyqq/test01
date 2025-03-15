clc
clear all
X=load('Raw_Data.txt');

%求参考值
[m1,p1]=max(X);
[m2,p2]=min(X);
c1  = [m2(1), (m1(1)-m2(1))/4+m2(1), (m1(1)-m2(1))/4*2+m2(1), (m1(1)-m2(1))/4*3+m2(1), m1(1) ]
c2  = [m2(2), (m1(2)-m2(2))/4+m2(2), (m1(2)-m2(2))/4*2+m2(2), (m1(2)-m2(2))/4*3+m2(2), m1(2) ]

%求解权重
[S,w1]=Entropy_method(X);
[w2]=yqq_quanzhong5();
fprintf('1样本的综合得分：%f 和 %f\n', w1(1),w1(2));
fprintf('2样本的综合得分：%f 和 %f\n', w2(1),w2(2));
