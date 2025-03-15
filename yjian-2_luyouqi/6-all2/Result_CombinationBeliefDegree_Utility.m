clc
clear all
data=load('Belief_Degree_Data.txt');
n=size(data);

%求解权重
%X=load('Raw_Data.txt');
%Ind=[2 1 1]; %正向和反向
%[S,w]=Entropy_method(X,Ind);

r=[0.9, 0.9, 0.9, 0.9, 0.9];
w=[0.2782,0.0886,0.1808,0.1892,0.2634];

A = zeros(24,5);%加载的数据有700行，12列，将每行4列为一组，组成四条证据
for i=1:n(1)
    com=[data(i,[1 2 3 4 5]);data(i,[6 7 8 9 10]);data(i,[11 12 13 14 15]);data(i,[16 17 18 19 20]);data(i,[21 22 23 24 25])];
    %com=[data(i,[1 2 3]);data(i,[4 5 6]);data(i,[7 8 9])];
    S=ER_Rule(r,w,com);
    A(i,[1,2,3,4,5]) = [S(3) S(4) S(5) S(6) S(7)];%也可以A(i,:) = [S(3) S(4) S(5) S(6)];
    %A(i,[1,2,3]) = [S(3) S(4) S(5)];
    zs(i)=S(3)*1+S(4)*2+S(5)*3+S(6)*4+S(7)*5;
    %zs(i)=S(3)*2+S(4)*4+S(5)*6;
end
writematrix(A,'CombinationBeliefDegree.txt')%输出组合置信度
writematrix(zs','Utility.txt')%输出效用
plot(zs','-*');