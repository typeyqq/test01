function bestx=test_p_cma_es(x0,G,Aeq,beq,ub,lb)
    %"ub" 和 "lb" 分别指代上界和下界，它们通常用于限制参数空间中的搜索范围。
    %X0规则乘以输入属性的参考值再加上规则个1和属性个1
    %G是训练轮数  %weights是策略参数   %
    %sigma是步长  %xmean是当前的均值向量   %mueff用于控制协方差矩阵的自适应调整
    
% --------------------  Initialization ---------------初始化
%用户定义的输入参数（需要编辑）
%strfitnessfct='frosenbrock'；%目标/适应度函数的名称
%FES=0；
N = length(x0);               % 目标变量数量/问题维度(维数）
num=0;
xmean = x0;    % 目标变量起始点（初始种群）
sigma = 0.5;          % 坐标标准偏差（步长）
%stopfitness = 1e-10;  % 如果适合度<适合度停止（最小化）（误差）10的负10次方
stopfitness = 0.002;
stopeval = G;         %迭代次数

% 策略参数设置：选择
lambda = 10+floor(3*log(N));  % 种群规模、后代数量（后代的数量） lambda 通常表示种群的大小，即每一代的种群中包含的个体数量
%log(82)==4.4067,X3向下取整为13，此处的lambda是23    %floor是向下取整
mu = lambda/2;               % 用于重组的父母/点数    %此处mu为11.5
weights = log(mu+1/2)-log(1:mu)'; %   用于加权重组的muXone阵列
mu = floor(mu);                   %之后mu作为父母
weights = weights/sum(weights);     % 归一化重组权重数组  %sum(weights.^2)=0.1483   %sum(weights)^2=1   %weights：这是用于重组时的个体权重
mueff=sum(weights)^2/sum(weights.^2); %mueff用于控制协方差矩阵的自适应调整   %    .^是把矩阵里的所有值都平方倍    1/0.1483=6.7410
%mueff 的值用于控制协方差矩阵的更新速度，影响算法的性能
%lambda：这是种群规模，它表示每一代的种群中包含的个体数量。在代码中，它的计算方式是 lambda = 10 + floor(3 * log(N))，
%其中 N 是问题的维度（或者说参数的数量）。计算结果应该是向下取整的结果，即 lambda 是一个整数。
%在你的示例中，lambda 的计算结果是 23，这意味着每一代的种群中包含 23 个个体。
%mu：这是用于重组的父母数量。通常，选择 mu 个个体作为父代进行重组，以生成下一代的后代。
%在你的代码中，mu 的计算方式是 mu = lambda / 2，即取种群规模 lambda 的一半作为父代的数量。在你的示例中，
%mu 的计算结果是 11.5，但由于 mu 通常需要是一个整数，可能需要进行舍入或者取最接近的整数值，例如，向下取整为 11。
%weights：这是用于重组时的个体权重，它影响哪些父代个体被选为重组的一部分。
%在你的代码中，weights 的计算方式是 weights = log(mu+1/2) - log(1:mu)'。这个计算方式涉及到对数操作和对 mu 的不同值进行减法操作，
%以确定个体在重组中的权重。weights 是一个包含了 mu 个权重值的数组。
%mueff 是一个与权重数组 weights 相关的参数，用于控制协方差矩阵的自适应调整

% 策略参数设置：自适应
cc = (4 + mueff/N) / (N+4 + 2*mueff/N); %mueff用于控制协方差矩阵的自适应调整
cs = (mueff+2) / (N+mueff+5);  %%！！！

%"cc" 控制协方差矩阵的自适应更新。具体来说，它用于控制协方差矩阵特征值的累积变化。
%较小的 "cc" 值会导致较慢的特征值更新，使得协方差矩阵保持较稳定，更适合处理问题的局部结构。
%较大的 "cc" 值会导致更快的特征值变化，使协方差矩阵更灵活，更适合处理问题的全局结构。
%"cs" 控制步长（或者说标准差）的逐步逼近。步长用于调整生成的候选解的幅度。
%较小的 "cs" 值会导致较慢的步长逼近，使得步长的变化幅度较小。
%较大的 "cs" 值会导致更快的步长逼近，使步长更加灵活，有助于在搜索空间中更远的范围内探索。

%总之，"cc" 控制协方差矩阵的更新速度，而 "cs" 控制步长的更新速度。
%这两个参数影响CMA-ES算法在搜索空间中的探索策略，可以通过调整它们来改变算法的性能。
%根据问题的性质，通常需要进行实验和调优，以找到最佳的参数设置，以便算法更好地适应不同类型的优化问题。

c1 = 2 / ((N+1.3)^2+mueff);    % learning rate for rank-one update of C   C排名第一更新的学习率
cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)^2+mueff));  %￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥
%cmu是均值向量的逐步逼近参数，用于调整均值向量xmean的更新大小
damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs; % damping for sigma
%damps" 是用于减小协方差矩阵的特征值（eigenvalues）的衰减因子

pc = zeros(N,1); ps = zeros(N,1);   % evolution paths for  C and sigma  C和sigma的进化路径
B = eye(N,N);        %eye是单位矩阵     % B defines the coordinate system  B定义坐标系
D = ones(N,1);                      % diagonal D defines the scaling  对角线D定义缩放
C = B * diag(D.^2) * B';            % covariance matrix C  协方差矩阵C
invsqrtC = B * diag(D.^-1) * B';    % C^-1/2
%"invsqrtC" 代表协方差矩阵的逆平方根。它是协方差矩阵的逆矩阵的平方根的逆。
%在P-CMA-ES中，"invsqrtC" 的作用是用于生成新的候选解。通过与均值向量相乘，
%它调整了生成的新候选解的幅度和方向，以便更好地匹配当前协方差矩阵的分布。

eigeneval = 0;                      % track update of B and D  跟踪B和D的更新
chiN=N^0.5*(1-1/(4*N)+1/(21*N^2));  % expectation of  预期
%   ||N(0,I)|| == norm(randn(N,1)) 
%out.dat = []; out.datx = [];  % for plotting output  用于绘制输出

% -------------------- Generation Loop ------------产生回路
counteval = 0;  % the next 40 lines contain the 20 lines of interesting code
while (counteval < stopeval) &&(sigma>stopfitness)
    % Generate and evaluate lambda offspring   生成和评估lambda子代
    for k=1:lambda   %%%---147
        %CMA-ES算法通过不断生成和评估后代来优化问题的目标函数。进化过程的主要步骤如下：
        %生成后代：对于种群中的每个个体，生成一个后代个体。后代的生成通常是通过从多元正态分布中采样得到的，
        %这个分布由均值向量 xmean 和协方差矩阵 C 控制。每个后代的生成方式如下
        arx(:,k) = xmean + sigma * B * (D .* randn(N,1)); %B方向，D幅度
        %其中，xmean 是当前种群的均值向量,它代表了当前种群中所有个体的平均位置，
        %sigma 是步长，用于控制搜索空间中的步长大小。B: 这是一个与协方差矩阵逆平方根有关的矩阵。它用于控制候选解的分布。
        % D: 这是一个与特征值有关的向量，通常用于调整协方差矩阵逆平方根的缩放。randn(N,1) 生成了一个多元正态分布的随机向量。
        % arx(:,k): 这是一个候选解的向量，它是算法生成的一个潜在的解。arx 表示 "archive of offspring," 也就是候选解的存档或记录。
        %竖行（列）的含义：每一列 arx(:, k) 表示一个单独的候选解，也就是在当前种群中的一个个体的位置。
        %横行（行）的含义：每一行代表问题的一个不同的参数或维度。
        %如果问题的参数维度是 N，那么 arx 的每一行包含了一个候选解在不同维度上的分量值。这些分量值构成了候选解的参数向量。
        %其中，xij 表示第 i 个维度上的第 j 个候选解分量的值。例如，x12 表示第一个维度上的第二个候选解的分量值。
        %xmean：这是当前种群中解向量的均值。
        %sigma：这是一个标准差或缩放因子，用于控制新解的变化程度。
        %B：这是协方差矩阵的特征向量矩阵，用于控制解向量在不同方向上的变化。
        %D：这是一个权重向量，通常用于调整随机生成的值的幅度。
        for kk=1:N 
            if arx(kk,k)>ub(kk)
                arx(kk,k)=ub(kk);
            end
            if arx(kk,k)<lb(kk)
                arx(kk,k)=lb(kk);
            end
        end  %%此处表明不超过上限和下限
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%投影算子%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [aa,b]=size(Aeq);% [aa, b] 是一个赋值语句，它将矩阵 Aeq 的行数赋给变量 aa，将列数赋给变量 b
        %aa是行==34，b是列==82
        a=0;
        for i=1:aa
            if beq(i)==1
                a=a+1;    %得到规则数a
            end
        end                                              %这里a得到规则数     ，b是总数，规则数X输出的参考值+规则数+属性数
        num=zeros(1,a);                                  %创建一个和规则数相同大小的1X16的数组，容纳参考值的个数
        for i=1:a
            for j=1:b
                if Aeq(i,j)==1
                    num(i)=num(i)+1;  %记录Aeq中每行1的个数，每个规则的输出参考值的数目
                    ini(i)=j;
                end
            end
            ini(i)=ini(i)-num(i)+1;   %记录Aeq中每行1的起始列号
        end
        for i=1:a  %a为16个规则              %%%%%%%%142
            
            %num里的数是记载着Aeq的一行1的个数，也就是参考值的个数。此处是对每个规则的不同参考值进行计算
            A=ones(1,num(i));       %总之就是i个规则和他的4个参考值，num里装的是1X16，里面装16个4
            arx(ini(i):(ini(i)+num(i)-1),k)=  arx(ini(i):(ini(i)+num(i)-1),k)...
                                                        - A'*inv(A*A')*...
                                                        (A*          arx(ini(i):(ini(i)+num(i)-1),k)     -1);
            %1到4，5到8，9到12，每相隔隔参考点进行一次，最后到61到64截至。    关于B = inv(A);  计算矩阵 A 的逆矩阵并存储在 B 中
            %inv(A*A')解释一下就是A和A的转置矩阵之积得到对称矩阵然后求这个对称矩阵的逆矩阵，为定值
            for j=ini(i):(ini(i)+num(i)-1)%对每个规则的参考值进行选取
                yu=0;
                if arx(j,k)<0
                    %yu在PCMA-ES 中通常代表分数分位点或移动分位点，这是一个用于决定哪些个体将被保留和哪些将被淘汰的阈值。
                    %所以此处小于0的值直接舍弃   
                    yu=yu+arx(j,k);
                    arx(j,k)=0;
                end
            end
            index=0;
            for j=ini(i):(ini(i)+num(i)-1)    %对每个规则的输出参考值进行选取
                if arx(j,k)~=0                %~ 表示逻辑非
                    index=index+1;     %index表示这几个参考值不是0的数目，以每个规则作为一个单位。
                end
            end
            yu=yu/index;
            for j=ini(i):(ini(i)+num(i)-1)    %对每个规则的输出参考值进行选取
                if arx(j,k)~=0
                    arx(j,k)=arx(j,k)+yu;
                end
            end
            
           %arx(ini(i):(ini(i)+num(i)-1),k)=arx(ini(i):(ini(i)+num(i)-1),k)./sum(arx(ini(i):(ini(i)+num(i)-1),k));    
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        arfitness(k) = fun(arx(:,k)'); % 此处送进去的是经过优化之后的库，返回的是一个衡量优化结果的值，越小的话越好。
        %23个个体一一进去实验，返回的是差异比较，可以直观的展示个个后代的优化情况。
    end   %23个个体单位到这里截至
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    counteval = counteval+1;%计数单位，和200进行比较
    % Sort by fitness and compute weighted mean into xmean  按适应度排序并将加权平均值计算到xmean中
    [arfitness, arindex] = sort(arfitness);  % 最小化，sort是排序，返回的arfitness是排序之后的，而arindex是索引
    xold = xmean;  %82X11 * 11X1=82X1
    xmean = arx(:,arindex(1:mu)) * weights;  % recombination, new mean value   重组，新平均值,xmean是重量，weights是比例
            %取arindex里的1到mu（11）列也就是取8，12，13，14，9，1....17一共11列最好的其他其全部不要，然后和个体质量相乘*****************
    %累积：更新进化路径
    ps = (1-cs) * ps ...%sqrt是计算给定的平方根
        + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold) / sigma;
    hsig = sum(ps.^2)/(1-(1-cs)^(2*counteval/lambda))/N < 2 + 4/(N+1);
    pc = (1-cc) * pc ...
        + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma;
    
    % Adapt covariance matrix C   自适应协方差矩阵C
    %artmp 是用于计算协方差矩阵 C 的一部分，它是一组差分向量，用于更新协方差矩阵
    artmp = (1/sigma) * (arx(:,arindex(1:mu)) - repmat(xold,1,mu));  % mu difference vectors mu差分矢量
    C = (1-c1-cmu) * C ...                   % regard old matrix  关于旧矩阵
        + c1 * (pc * pc' ...                % plus rank one update   加上排名第一的更新
        + (1-hsig) * cc*(2-cc) * C) ... % minor correction if hsig==0   如果hsig==0，则进行轻微校正
        + cmu * artmp * diag(weights) * artmp'; % plus rank mu update   加秩mu更新
    %协方差矩阵描述了后代的生成分布，通过它，算法可以控制新生成的后代个体在参数空间中的分布范围。
    %协方差矩阵的特征向量（B）和特征值（D）用于控制生成的分布方向和幅度。
    %特征向量定义了新生成的个体在哪些方向上相对较多，而特征值则决定了在这些方向上的幅度。
    %通过调整协方差矩阵，可以使后代在搜索空间中集中于有望找到更好解的方向。
            %自适应调整协方差矩阵：PCA-CMA-ES算法具有自适应性，可以根据后代的表现来自动调整协方差矩阵。
        %如果后代的个体表现良好，协方差矩阵可能会在有望找到更好解的方向上进行调整，以加速收敛。
        %如果后代表现较差，协方差矩阵可能会相应地进行调整，以更广泛地探索解空间。
            %影响搜索策略：协方差矩阵的调整会影响算法的搜索策略。通过适当调整协方差矩阵，
        %算法可以更好地适应问题的性质，可以在不同阶段采用不同的搜索策略。
        %较小的特征值对应的方向通常用于探索局部结构，而较大的特征值对应的方向通常用于探索全局结构。
            %维持多样性：协方差矩阵的调整还有助于维持种群的多样性。它可以确保种群在搜索空间中不会陷入局部最优解，并能够在全局范围内进行探索。
        %总之，协方差矩阵在PCA-CMA-ES算法中是一个关键的参数，通过它可以控制后代的生成分布，自适应地调整搜索策略，
        %并在优化问题中实现更好的探索和收敛。通过不断调整协方差矩阵，算法可以适应不同类型的优化问题并提高搜索的效率。
        %这使得PCA-CMA-ES算法成为一种强大的优化方法，特别适用于复杂的高维问题。

    % Adapt step size sigma   调整步长
    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1));    %exp就是自然常数e为底的指数函数，为0就不变
    sigmazu(counteval)=sigma;
    %ps 是种群中的个体的平均位置，norm(ps) 表示ps的各个元素平方之和开根号
  
    % Update B and D from C  从C更新B和D
    if counteval - eigeneval > lambda/(c1+cmu)/N/10  % to achieve O(N^2)
        eigeneval = counteval;
        C = triu(C) + triu(C,1)'; % enforce symmetry  强制对称  %triu是取上三角
        [B,D] = eig(C);           % 计算矩阵C的特征值和特征向量的函数是eig(C)      B 是包含特征向量的矩阵。D 是包含特征值的对角矩阵。
        %V 是一个矩阵，它包含了矩阵 A 的特征向量， D 是一个对角矩阵，它包含了矩阵 A 的特征值，D 的对角线上的元素就是特征值
        D = sqrt(diag(D));        % 把D的对角矩阵上的特征值取出来开根
        invsqrtC = B * diag(D.^-1) * B'; %invsqrtC" 代表协方差矩阵的逆平方根。它是协方差矩阵的逆矩阵的平方根的逆
    end
    % Break, if fitness is good enough or condition exceeds 1e14, better termination methods are advisable
    %休息，如果身体状况足够好或状况超过1e14，建议采用更好的终止方法
    if counteval==1
        bestf=arfitness(1);
        bestx=arx(:,arindex(1));
    else
        if arfitness(1)<bestf
            bestf=arfitness(1);%想办法使得最小
            bestx=arx(:,arindex(1));%
        end
    end
    % if arfitness(1) <= stopfitness || max(D) > 1e7 * min(D)
    %   break;
    % end
    counteval
end % while, end generation loop
end