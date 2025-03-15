function bestx=yqqtwo(x0,G,Aeq,beq,ub,lb)
    %"ub" 和 "lb" 分别指代上界和下界，它们通常用于限制参数空间中的搜索范围。
    %X0规则乘以输入属性的参考值再加上规则个1和属性个1
    %G是训练轮数  %weights是策略参数   %
    %sigma是步长  %xmean是当前的均值向量   %mueff用于控制协方差矩阵的自适应调整
N = length(x0);               % 目标变量数量/问题维度(维数）
xmean = x0;    % 目标变量起始点（初始种群）
sigma = 0.5;          % 坐标标准偏差（步长）
%stopfitness = 1e-10;  % 如果适合度<适合度停止（最小化）（误差）10的负10次方
        %迭代次数

% 策略参数设置：选择

%设置最大和最小的限制，每一组返回的直接和上一个的最大值之间建立平均值，初始的是1，而响应的把他进行合理的放缩，初始值设置为1
%这里设置一个

% 策略参数设置：选择log10(23/log10(4))
lambda = 15+floor(3*log(N));  % 种群规模、后代数量（后代的数量） lambda 通常表示种群的大小，即每一代的种群中包含的个体数量,23
zushu=ceil(log10(lambda)/log10(4));%得到分的组，此处分成5组
yqq1=ceil(2*lambda/zushu);%此时得到的yqq1作为每组的个体数
yqq2=ceil(2*lambda-yqq1*(zushu-1));
mu = yqq1/2;               % 用于重组的父母/点数    %此处mu为11.5
weights = log(mu+1/2)-log(1:mu)'; %   用于加权重组的muXone阵列
mu = floor(mu);                   %之后mu作为父母
weights = weights/sum(weights);     % 归一化重组权重数组  %sum(weights.^2)=0.1483   %sum(weights)^2=1   %weights：这是用于重组时的个体权重
mueff=sum(weights)^2/sum(weights.^2); %mueff用于控制协方差矩阵的自适应调整   %    .^是把矩阵里的所有值都平方倍    1/0.1483=6.7410
bestx=1000;mse=100; bvb=0
while (mse> 0.00000000001)&&(bvb<3)
    for v=1:(zushu-1)
        %bests(:,v)=yqqtwo2(N,xmean,sigma,ub,lb,yqq1,mu,mueff,G,Aeq,beq,weights);
        [xmean1, sigma1, mueff1, bestx1]=yqqtwo2(N,xmean,sigma,yqq1,mu,mueff,G,Aeq,beq,weights,ub,lb);

        if fun_test( bestx1)<mse
            mse=fun_test(bestx1);
            xmean2=xmean1; sigma2=sigma1; mueff2=mueff1;bestx2=bestx1;
        end
    end
        [xmean1, sigma1, mueff1, bestx1]=yqqtwo2(N,xmean,sigma,yqq2,mu,mueff,G,Aeq,beq,weights,ub,lb);
        if fun_test(bestx1)<mse
            mse=fun_test(bestx1);
            xmean2=xmean1; sigma2=sigma1; mueff2=mueff1;bestx2=bestx1;
        end
        bvb=bvb+1;
        for k=1:25*5
            ub(k) = ub(k)-1/4*(ub(k)-xmean2(k));
            lb(k) = lb(k)+1/4*(lb(k)+xmean2(k));
            %81和82染上1
        end
        xmean=xmean2; sigma=sigma2; mueff=mueff2;bestx=bestx2;
end
  
end