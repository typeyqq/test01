function [s,w]=Entropy_method(x)
% 函数shang.m, 实现用熵值法求各指标(列）的权重及各数据行的得分
% x为原始数据矩阵, 一行代表一个国家, 每列对应一个指标
% s返回各行得分, w返回各列权重
[n,m]=size(x); % n=23个国家, m=5个指标
%% 数据的归一化处理
% Matlab2010b,2011a,b版本都有bug,需如下处理. 其它版本直接用[X,ps]=mapminmax(x',0,1);即可
a=x';
[X,ps]=mapminmax(x');%确认此处归一化是归到-1到1之间，每行为一个单位，按比例缩小或者放大
ps.ymin=0.002; % 归一化后的最小值0.002
ps.ymax=0.996; % 归一化后的最大值0.996
%ps.xmin：输入数据 x 的最小值。         ps.xmax：输入数据 x 的最大值。
%ps.ymin：归一化后的数据的最小值。      ps.ymax：归一化后的数据的最大值。
ps.yrange=ps.ymax-ps.ymin; % 归一化后的极差,若不调整该值, 则逆运算会出错，yrange表示归一化之后的范围归一化之后的上减下，没修改的话是2，改了的是上面差
X=mapminmax(x',ps);%按照规定的上下限返回
% mapminmax('reverse',xx,ps); % 反归一化, 回到原数据
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
s=w*p';         % 求综合得分 [\code]
end