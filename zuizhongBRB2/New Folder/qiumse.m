clear all 
clc
TrainData=load('mse2.txt');%读取训练数据
shiji = TrainData(:,1);
T=length(TrainData);
out=zeros(32,11);  
for i=1:11
    for n=1:T
        out(n,i) = ( TrainData(n,i) - shiji(n) )^2;  
        %计算每个样本的预测输出值 y(n) 与实际观测值 TrainData(n,3) 之间的差异，并将其平方  
    end
    outmse(i) = sum(out(:,i)) /T;
end
outmse