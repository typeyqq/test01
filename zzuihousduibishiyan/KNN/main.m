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

mdl = fitcknn(train_data,train_label);  %��ʱĬ��ֵk = 1
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
% ��ȡѵ�����Ͳ��Լ�����
% ��ȡѵ�����Ͳ��Լ�����
% ��ȡѵ�����Ͳ��Լ�����
train_data = load('train9.txt');
test_data = load('test9.txt');

% ��ȡѵ�����Ͳ��Լ�������������Ŀ��ֵ
train_X = train_data(:, 1:55);
train_Y = train_data(:, 56);

test_X = test_data(:, 1:55);
test_Y = test_data(:, 56);

% ����BP������ģ��
net = fitnet(10); % 10��������Ԫ�����Ը�����Ҫ����

% �������������
net.trainParam.epochs = 1000; % ����ѵ���������������Ը�����Ҫ����

% ѵ��������ģ��
net_trained = train(net, train_X', train_Y');

% ʹ��ѵ���õ�ģ�ͽ���Ԥ��
predicted_Y = net_trained(test_X');

% ����Cohen��s d
[~, ~, ~, stats] = ttest2(test_Y, predicted_Y');

% ��ӡCohen��s d �� p ֵ
fprintf('Cohen��s d: %.4f\n', stats.tstat);
fprintf('p-value: %.4f\n', stats.df);

% ����Ԥ������ʵ�ʽ����ͼ
figure;
plot(test_Y, '-o', 'DisplayName', 'Actual Values');
hold on;
plot(predicted_Y, '-o', 'DisplayName', 'Predicted Values');
legend('show');
xlabel('Sample');
ylabel('Value');
title('Actual vs Predicted Values');
grid on;
