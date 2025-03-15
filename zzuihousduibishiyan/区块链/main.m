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
network_data = zs;

% 步骤2：模拟硬件安全模块对数据进行安全处理
secure_module_output = secureHardwareModule(network_data);

% 步骤3：模拟区块链技术，将处理后的数据写入区块链
blockchain = createBlockchain();
blockchain = addToBlockchain(blockchain, secure_module_output);

% 步骤4：进行复杂的风险评估
risk_score = complexRiskAssessment(network_data, secure_module_output);
a=secure_module_output';
% 步骤5：显示结果
disp('网络安全风险评估结果：');
disp(['原始数据：', num2str(network_data)]);
disp(['安全模块处理后的数据：', num2str(secure_module_output)]);
disp(['复杂风险评分：', num2str(risk_score)]);
% 计算 MSE
mse_value = mean((network_data - secure_module_output).^2);
% 显示 MSE 结果
disp(['MSE 值：', num2str(mse_value)]);


% 定义更复杂的硬件安全模块函数
function output = secureHardwareModule(input_data)
    % 在实际情况下，此函数将包含真实的硬件安全模块实现
    % 例如，采用深度学习、加密等技术进行更复杂的安全处理
    % 这里仅为示例，假设硬件安全模块使用简单的神经网络进行处理
    net = feedforwardnet(10);
    net.trainParam.epochs = 100;
    net = train(net, input_data, input_data);
    output = sim(net, input_data);
end

% 定义更复杂的区块链创建函数
function blockchain = createBlockchain()
    % 在实际情况下，此函数将包含真实的区块链创建逻辑
    % 例如，生成创世块、初始化链表等
    blockchain = {};
end

% 定义更复杂的添加到区块链函数
function blockchain = addToBlockchain(blockchain, data)
    % 在实际情况下，此函数将包含真实的区块链添加逻辑
    % 例如，创建新块、添加到链表等
    new_block = struct('data', data, 'timestamp', datetime('now'));
    blockchain = [blockchain, new_block];
end

% 定义更复杂的风险评估函数
function risk_score = complexRiskAssessment(original_data, processed_data)
    % 在实际情况下，此函数将包含真实的网络安全风险评估逻辑
    % 例如，使用深度学习模型、异常检测等技术进行更复杂的风险评估
    % 这里仅为示例，使用神经网络重建误差作为风险评分
    reconstruction_error = norm(original_data - processed_data);
    risk_score = reconstruction_error;
end
