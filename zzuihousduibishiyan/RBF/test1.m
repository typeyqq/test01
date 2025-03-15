clc
clear
close all

%---------------------------------------------------
% 产生训练样本与测试样本，每一列为一个样本

P1 = [rand(3,5),rand(3,5)+1,rand(3,5)+2];
T1 = [repmat([1;0;0],1,5),repmat([0;1;0],1,5),repmat([0;0;1],1,5)];

P2 = [rand(3,5),rand(3,5)+1,rand(3,5)+2];
T2 = [repmat([1;0;0],1,5),repmat([0;1;0],1,5),repmat([0;0;1],1,5)];

%---------------------------------------------------
% 归一化

PN1= mapminmax(P1);
PN2 = mapminmax(P2);

%---------------------------------------------------
% 训练

switch 2
case 1
        
% 神经元数是训练样本个数
spread = 1;                   % 此值越大,覆盖的函数值就大(默认为1)
net = newrbe(PN1,T1,spread);

case 2
    
% 神经元数逐步增加,最多就是训练样本个数
goal = 1e-4;                    % 训练误差的平方和(默认为0)
spread = 1;                   % 此值越大,需要的神经元就越少(默认为1)
MN = size(PN1,2);               % 最大神经元数(默认为训练样本个数)
DF = 1;                         % 显示间隔(默认为25)
net = newrb(PN1,T1,goal,spread,MN,DF);

case 3
    
spread = 1;                   % 此值越大,需要的神经元就越少(默认为1)
net = newgrnn(PN1,T1,spread);
    
end

%---------------------------------------------------
% 测试

Y1 = sim(net,PN1);             % 训练样本实际输出
Y2 = sim(net,PN2);             % 测试样本实际输出

Y1 = full(compet(Y1));         % 竞争输出
Y2 = full(compet(Y2));          

%---------------------------------------------------
% 结果统计

Result = ~sum(abs(T1-Y1))                 % 正确分类显示为1
Percent1 = sum(Result)/length(Result)      % 训练样本正确分类率

Result = ~sum(abs(T2-Y2))                 % 正确分类显示为1
Percent2 = sum(Result)/length(Result)      % 测试样本正确分类率


