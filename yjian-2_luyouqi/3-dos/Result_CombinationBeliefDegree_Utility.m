clc
clear all
data=load('Belief_Degree_Data.txt');
n=size(data);

%求解权重
%X=load('Raw_Data.txt');
%Ind=[2 1 1]; %正向和反向
%[S,w]=Entropy_method(X,Ind);

r=[0.9, 0.9];%可靠值
w=[0.5151 , 0.4849];%权重值，使用quanzhong计算且二者之和要为1￥￥￥￥￥改点，或熵值法优化

A = zeros(96,5);%加载的数据有24行，10列，将每行5列为一组，组成2条证据
for i=1:n(1)
    com=[data(i,[1 2 3 4 5]);data(i,[6 7 8 9 10])];
    %得到一个2X5的矩阵com，一行一行取出
    %com=[data(i,[1 2 3]);data(i,[4 5 6]);data(i,[7 8 9])];
    S=ER_Rule(r,w,com);%引入可靠值和权重值，使得两个置信度进行融合得到融合之后的置信度
    %S 包含了最小可信度、最大可信度和每个假设的最终置信度，供后续分析和决策使用。
    S
    A(i,[1,2,3,4,5]) = [S(3) S(4) S(5) S(6) S(7)];%也可以A(i,:) = [S(3) S(4) S(5) S(6)];
    %此处是死规矩，S调用了ER_Rule的一个方法，使得从第三个开始，有几个参考值写几个
    %A(i,[1,2,3]) = [S(3) S(4) S(5)];
%     zs(i)=S(3)*1+S(4)*2+S(5)*3+S(6)*4+S(7)*5;
    %zs(i)=S(3)*0.2+S(4)*0.4+S(5)*0.6+S(6)*0.8+S(7)*1;%得到96个融合之后的输出
    zs(i)=S(3)*1+S(4)*2+S(5)*3+S(6)*4+S(7)*5;%得到96个融合之后的输出
    %输出参考值怎么选就怎么设
    %zs(i)=S(3)*2+S(4)*4+S(5)*6;
end
writematrix(A,'CombinationBeliefDegree.txt')%输出组合置信度
writematrix(zs','Utility.txt')%输出效用
%融合结果

% plot(zs','-*');
% xlabel('时间');
% ylabel('网络安全态势值');
% hold on
% data1=load('Test.txt');
% plot(data1,'r');