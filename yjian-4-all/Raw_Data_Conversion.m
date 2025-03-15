clc
clear all
data = load('Raw_Data.txt');
n = size(data);
y1 = [ 1, 1.25, 1.5, 1.75, 2 ];%
y2 = [ 1, 1.25, 1.5, 1.75, 2 ];%
y3 = [ 1, 1.25, 1.5, 1.75, 2.07 ];%

A = zeros(n); %//A存储为最后转成的置信度  因为加载的文件有500行16列
for i = 1:n(1)%一行一行的处理
    output1 = zeros(1,5);%用来存储第i行的第一个属性数据转化为的置信度
    output2 = zeros(1,5);%用来存储第i行的第二个属性数据转化为的置信度
    output3 = zeros(1,5);%用来存储第i行的第三个属性数据转化为的置信度
%     output4 = zeros(1,5);%用来存储第i行的第四个属性数据转化为的置信度
%     output5 = zeros(1,4);%用来存储第i行的第五个属性数据转化为的置信度
    for j = 1:3 %在raw_data.txt中有三列
        switch j
            case 1
                output1 = Belief_Degree_Ascending(y1,data(i,j)); %原始数据转置信度               
            case 2
                output2 = Belief_Degree_Ascending(y2,data(i,j));             
            case 3
                output3 = Belief_Degree_Ascending(y3,data(i,j));     
%             case 4
%                 output4 = Belief_Degree_Ascending(y4,data(i,j));
%             case 5
%                output5 = Belief_Degree_Ascending(y5,data(i,j));
        end
    end
    A(i,[1,2,3,4,5])=output1;
    A(i,[6,7,8,9,10])=output2;
    A(i,[11,12,13,14,15])=output3;
%     A(i,[16,17,18,19,20])=output4;
%     A(i,[17,18,19,20])=output5;
end
%A=A';
%B=A(:);
%C=reshape(B,3,2100);
%A=C';
writematrix(A,'Belief_Degree_Data.txt')


