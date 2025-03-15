% 参数设置
clc
% 生成模拟数据
num_samples = 32;
actual_data = randi([0, 1], 1, num_samples);
observed=load('test.txt');%读取训练数据


% 模拟网络安全数据
%observed_data = simulate_network_data(actual_data);
%observed_data=TrainData(:,1);
x=[observed(:,1),observed(:,2)];

[n,m]=size(x); % n=23个国家, m=5个指标

nn = size(x);
y1 = [ 1, 1.05 , 1.11 , 1.5 , 3.7];
y2 = [ 1, 1.25, 1.5, 1.75, 2.05 ];
A = zeros(nn); %//A存储为最后转成的置信度  因为加载的文件有500行16列
data=x;
for i = 1:nn(1)%一行一行的处理
    output1 = zeros(1,5);%用来存储第i行的第一个属性数据转化为的置信度
    output2 = zeros(1,5);%用来存储第i行的第二个属性数据转化为的置信度
    for j = 1:2 %在data.txt中有三列
        switch j
            case 1
                output1 = belief_D(y1,data(i,j)); %原始数据转置信度
            case 2
                output2 = belief_D(y2,data(i,j));
        end
    end
    A(i,[1,2,3,4,5])=output1;
    A(i,[6,7,8,9,10])=output2;
end
data=A;
a=x';
[~,ps]=mapminmax(x');%确认此处归一化是归到-1到1之间，每行为一个单位，按比例缩小或者放大
ps.ymin=0.002; % 归一化后的最小值0.002
ps.ymax=0.996; % 归一化后的最大值0.996
ps.yrange=ps.ymax-ps.ymin; % 归一化后的极差,若不调整该值, 则逆运算会出错，yrange表示归一化之后的范围归一化之后的上减下，没修改的话是2，改了的是上面差
X=mapminmax(x',ps);%按照规定的上下限返回
X=X';  % X为归一化后的数据, 23行(国家), 5列(指标)
%% 计算第j个指标下，第i个记录占该指标的比重p(i,j)
for i=1:n
    for j=1:m
        p(i,j)=X(i,j)/sum(X(:,j));%每个属性单独进行
    end
end
%% 计算第j个指标的熵值e(j)
k=1/log(n);%log（10）的意思就是ln(10)
for j=1:m
    e(j)=-k*sum(p(:,j).*log(p(:,j)));
end
d=ones(1,m)-e;  % 计算信息熵冗余度
w=d./sum(d);    % 求权值w
r=[0.9, 0.9];%可靠值
A = zeros(32,5);%加载的数据有24行，10列，将每行5列为一组，组成2条证据
zs=zeros(1,32);
for i=1:nn(1)
    com=[data(i,[1 2 3 4 5]);data(i,[6 7 8 9 10])];
    S=ER_R(r,w,com);%引入可靠值和权重值，使得两个置信度进行融合得到融合之后的置信度
    A(i,[1,2,3,4,5]) = [S(3) S(4) S(5) S(6) S(7)];%也可以A(i,:) = [S(3) S(4) S(5) S(6)];
    zs(i)=S(3)*0.2+S(4)*0.4+S(5)*0.6+S(6)*0.8+S(7)*1;%得到96个融合之后的输出
    
   
end

num_samples = 32;
sigma_attack = 0.5; % 攻击信号的标准差
sigma_defense = 0.3; % 防御信号的标准差

% 生成攻击信号和防御信号
%attack_signal = sigma_attack * randn(1, num_samples);
attack_signal =x(1)';
%defense_signal = sigma_defense * randn(1, num_samples);
defense_signal =x(2)';
% 生成实际数据
%actual_data = randn(1, num_samples);
actual_data =zs;
zs=zs';
% 模拟攻击和防御
simulated_data = actual_data + attack_signal - defense_signal;

% 使用数字孪生技术进行重构
reconstructed_data = deconv(simulated_data, attack_signal + defense_signal);

% 评估结果
mse = mean((actual_data - reconstructed_data).^2);


fprintf('均方误差 (MSE): %f\n', mse);

%0.0676