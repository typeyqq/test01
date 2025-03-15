clear all;
close all;
clc;
%{
% num=xlsread('sample.xlsx','Sheet1','A1:E60');
% 
% input_train=num(1:20,1:4)';
% output_train=num(1:20,5)';
% input_test=num(21:60,1:4)';
% output_test=num(21:60,5)';
traini=load('train9.txt');
testi=load('test9.txt');

input_train=traini(:,1:2)';
output_train=traini(:,3)';
input_test=testi(:,1:2)';
output_test=testi(:,3)';
%  input_train=load('a1.txt')';
%  output_train=load('a2.txt')';
%  input_test=load('b1.txt')';
%  output_test=load('b2.txt')';

[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%初始化网络 输入 输出 隐层 节点个数 
%设置迭代次数，学习率，目标值
net=newff(inputn,outputn,20);
net.trainParam.epochs=200;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00004;
net.performFcn = 'mse'; 

for m=1:32
net=train(net,inputn,outputn);

%归一化
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net,inputn_test);
BPoutput(m,:)=mapminmax('reverse',an,outputps);

perf(m) = perform(net,BPoutput,output_test)
%c(m)=abs(BPoutput-output_test).^2/500;
b(m)=mse(BPoutput(m,:)-output_test)
a(m)=sqrt(mean((BPoutput(m,:)-output_test).^2)); 
c(m)=1-sqrt(mean(abs((BPoutput(m,:)-output_test))/output_test.^2));
d=mean(c);
end
%wei=[0.678106900708194,0.702571150449864,0.699880408950343,0.842641146355010,0.789445298323350,0.870947616480532,0.855295968367721,0.894385243066930,0.931807551183291,0.517184100358014,0.793797519549421,0.804277365045593,0.728315410970017,0.780705193980077,0.703706127161247,0.791818066139252,0.673460249416703,0.580326777720796,0.177724414146562,0.746064002652374,0.938387209621773,0.929833562581731,0.852480833996518,0.605115994506684,0.745581263862958,0.643224685837694,0.778334421065947,0.979831399250232,0.801114065547266,0.919698157342081,0.897891491967886,0.381352309966871,0.592615039308465,0.749807412259861,0.856093992980454,0.789235430090329,0.744704367306722,0.435655245191876,0.823398432227687,1.14942520312819,0.872555015299946,0.951860027202406,0.903812798260609,1.08776508670297,0.624013237477088,0.512539012083356,0.561813281147299,0.813872116035808,1.04571371605249,0.471161472332502,0.836820843792250,0.925164783592394,0.908677072930406,0.655110708898177,0.500875089614910,0.691251381432440,0.551225672846753,0.640296974275489,0.795520083006309,0.732849387611368,0.485683199186935,0.569995650371894,0.512865646407961,0.516794572225136,0.780072109349443,0.702022002027623,0.354916830786238,0.476049760058213,0.232053911378026,0.426558862936562,0.595208330736575,0.857713072077734,0.840504908285192,0.854887436984132,0.775766422417367,0.707666942055055,0.855408481734850,0.652554661596886,0.762725845661359,0.473746197878810,0.546644925077736,0.828943557619348,0.458565137991975,0.720127895054319,0.925107320386746,0.889633814582502,0.694416932818543,0.955529245158086,0.900455115153628,0.910563758751932,0.878431435673380,0.743580619866475,0.842218177429013,0.937603927915343,0.580013789067385,0.544555034454778,0.725321189622473,0.642997619041340,0.593278675303359,0.655152485751857];
%a=mse(wei-output_test);
%k=1:100;
% plot(k,output_test(k),'--g',k,BPoutput(k),'*');
% grid on;

for k=1:32
yuce1(k,1)=BPoutput(1,k);
% yuce16(k,1)=BPoutput(16,k);
% yuce32(k,1)=BPoutput(32,k);
end
MSE1=mse(yuce1-output_test)
% MSE16=mse(yuce16-output_test)
% MSE32=mse(yuce32-output_test)

plot(output_test(1:32));
hold on
plot(yuce1(1:32),'-o');
% hold on
% plot(yuce16(1:32),'r');
% hold on
% plot(yuce32(1:32),'-*');
%}
% 1. 读取训练集和测试集数据
% 读取训练集和测试集数据
% 读取训练集和测试集数据
train_data = load('train9.txt'); % 64行56列
test_data = load('test9.txt');   % 32行56列

% 提取训练集和测试集的输入和输出
train_input = train_data(:, 1:2);
train_output = train_data(:, 3);
test_input = test_data(:, 1:2);
test_output = test_data(:, 3);

% 构建BP神经网络模型
net = feedforwardnet(10); % 这里10是隐藏层神经元的数量，可以根据实际情况调整

% 设置训练参数
net.trainParam.epochs = 100; % 训练轮数
net.trainParam.lr = 0.01;    % 学习率

% 训练BP神经网络
net = train(net, train_input', train_output');

% 使用训练好的模型进行预测
predicted_output = net(test_input');

% 计算Wilcoxon signed-rank test的P值
[p, h, stats] = signtest(predicted_output, test_output);

% 计算Cohen’s d
d = computeCohensD(predicted_output, test_output);

% 打印结果
disp(['Wilcoxon signed-rank test P值: ' num2str(p)]);
disp(['Cohen’s d: ' num2str(d)]);

% 定义函数计算Cohen’s d
function d = computeCohensD(x, y)
    n1 = length(x);
    n2 = length(y);
    
    mean_diff = mean(x) - mean(y);
    
    s1 = std(x);
    s2 = std(y);
    
    pooled_std = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1 + n2 - 2));
    
    d = mean_diff / pooled_std;
end
