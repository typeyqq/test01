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
    if(zs(i)>0.35)
        zs(i)=1;
    else
        zs(i)=0;
    end
    if(observed(i,3)>0.2)
        observed(i,3)=1;
    else
        observed(i,3)=0;
    end
end


% 模拟网络安全数据
observed_data = observed(:,3)';
actual_data=zs;
% 参数设置
transition_prob = [0.7, 0.3; 0.3, 0.7]; % 转移概率矩阵
emission_prob = [0.9, 0.1; 0.1, 0.9]; % 发射概率矩阵

% Viterbi算法
estimated_data = viterbi_algorithm(observed_data, transition_prob, emission_prob);

% 风险评估
risk_score = assess_risk(actual_data, estimated_data);

% 显示结果
disp('实际数据:');
disp(actual_data);
disp('观测数据:');
disp(observed_data);
disp('估计数据:');
disp(estimated_data);
disp('风险评分:');
disp(risk_score);
% 计算均方误差（MSE）
% ... （在上面的代码基础上）

% 计算均方误差 (MSE)
mse = mean((actual_data - estimated_data).^2);
disp('均方误差 (MSE):');
disp(mse);

% 模拟网络数据的函数
function observed_data = simulate_network_data(actual_data)
    noise_prob = 0.2; % 噪声概率
    observed_data = actual_data;
    noise = rand(size(actual_data)) < noise_prob;
    observed_data(noise) = 1 - observed_data(noise);
end

% Viterbi算法的实现
function estimated_data = viterbi_algorithm(observed_data, transition_prob, emission_prob)
    num_states = size(transition_prob, 1);
    num_obs = length(observed_data);

    % 初始化
    viterbi_matrix = zeros(num_states, num_obs);
    backpointer_matrix = zeros(num_states, num_obs);

    % 初始状态的概率
    viterbi_matrix(:, 1) = log(0.5) + log(emission_prob(:,observed_data(1) + 1));

    % 递推计算
    for t = 2:num_obs
        for s = 1:num_states
            [max_prob, prev_state] = max(viterbi_matrix(:, t-1) + log(transition_prob(:, s)));
            viterbi_matrix(s, t) = max_prob + log(emission_prob(s, (observed_data(t) + 1) ));
            backpointer_matrix(s, t) = prev_state;
        end
    end

    % 回溯路径
    estimated_data = zeros(1, num_obs);
    [~, last_state] = max(viterbi_matrix(:, end));
    estimated_data(end) = last_state - 1;
    for t = num_obs-1:-1:1
        estimated_data(t) = backpointer_matrix(estimated_data(t+1) + 1, t+1) - 1;
    end
end

% 风险评估函数
function risk_score = assess_risk(actual_data, estimated_data)
    incorrect_predictions = nnz(actual_data ~= estimated_data);
    total_samples = length(actual_data);
    risk_score = incorrect_predictions / total_samples;
end
