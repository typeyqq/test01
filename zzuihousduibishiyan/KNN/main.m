clear all
close all
clc
%{
traini=load('train9.txt');
testi=load('test9.txt');

train_data=traini(:,1:55);
train_label=traini(:,56);
test_data=testi(:,1:55);
y2=testi(:,56);

% train_data=load('data1.txt')
% train_label=load('data1_label.txt')
% test_data=load('test1.txt')
% y2=load('test1_label.txt')

mdl = fitcknn(train_data,train_label);  %此时默认值k = 1
y = predict(mdl,test_data)
%MSE= mean((y-y2).^2);
b=mse(y-y2);
a=sqrt(mean((y-y2).^2)); 
c=1-sqrt(mean(abs((y-y2))/y2.^2));
d=mean(c);
MSE=b
k=1:32;
 plot(k,y2(k),'-',k,y(k),'-*');
 grid on;
%}
% 读取训练集和测试集数据
% 读取训练集和测试集数据
% 读取训练集和测试集数据
train_data = load('train9.txt');
test_data = load('test9.txt');

% 提取训练集和测试集的输入特征和目标值
train_X = train_data(:, 1:55);
train_Y = train_data(:, 56);

test_X = test_data(:, 1:55);
test_Y = test_data(:, 56);

% 创建BP神经网络模型
net = fitnet(10); % 10个隐层神经元，可以根据需要调整

% 设置神经网络参数
net.trainParam.epochs = 1000; % 设置训练迭代次数，可以根据需要调整

% 训练神经网络模型
net_trained = train(net, train_X', train_Y');

% 使用训练好的模型进行预测
predicted_Y = net_trained(test_X');

% 计算Cohen’s d
[~, ~, ~, stats] = ttest2(test_Y, predicted_Y');

% 打印Cohen’s d 和 p 值
fprintf('Cohen’s d: %.4f\n', stats.tstat);
fprintf('p-value: %.4f\n', stats.df);

% 画出预测结果和实际结果的图
figure;
plot(test_Y, '-o', 'DisplayName', 'Actual Values');
hold on;
plot(predicted_Y, '-o', 'DisplayName', 'Predicted Values');
legend('show');
xlabel('Sample');
ylabel('Value');
title('Actual vs Predicted Values');
grid on;
