function bestx=cmaes(x,G,Aeq,beq,ub,lb)
% (mu/mu_w, lambda)-CMA-ES
% CMA-ES: Evolution Strategy with Covariance Matrix Adaptation for
% CMA-ES：具有协方差矩阵自适应的进化策略
% nonlinear function minimization. To be used under the terms of the
% GNU General Public License (http://www.gnu.org/copyleft/gpl.html).
% Copyright: Nikolaus Hansen, 2003-09.
% e-mail: hansen[at]lri.fr
% This code is an excerpt from cmaes.m and implements the key parts
% of the algorithm. It is intendend to be used for READING and
% UNDERSTANDING the basic flow and all details of the CMA-ES
% *algorithm*. Use the cmaes.m code to run serious simulations: it
% is longer, but offers restarts, far better termination options,
% and supposedly quite useful output.
%这段代码摘录自cmaes.m，实现了关键部分算法。它旨在用于阅读和了解CMA-ES的基本流程和所有细节
%*算法*。使用cmaes.m代码来运行严肃的模拟：它更长，但提供了重新启动、更好的终止选项，
%并且被认为是非常有用的输出。

%
% URL: http://www.lri.fr/~hansen/purecmaes.m
% References: See end of file. Last change: October, 21, 2010

% --------------------  Initialization --------------------------------
% User defined input parameters (need to be edited)
%strfitnessfct = 'frosenbrock';  % name of objective/fitness function
%FES=0;
N = length(x);               % number of objective variables/problem dimension（维数）
xmean = x;    % objective variables initial point（初始种群）
sigma = 0.5;          % coordinate wise standard deviation (step size)（步长）
stopfitness = 1e-10;  % stop if fitness < stopfitness (minimization)（误差）
stopeval = G;%1e3*N^2;   % stop after stopeval number of function evaluations（最大代数）
%训练轮数

% Strategy parameter setting: Selection
lambda = 10+floor(3*log(N));  % population size, offspring number（后代的数量）
mu = lambda/2;               % number of parents/points for recombination  用于重组的父母/点数
weights = log(mu+1/2)-log(1:mu)'; % muXone array for weighted recombination   用于加权重组的muXone阵列
mu = floor(mu);
weights = weights/sum(weights);     % normalize recombination weights array   归一化重组权重数组
mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i   和w_ix_i的方差有效性

% Strategy parameter setting: Adaptation  策略参数设置：自适应
cc = (4 + mueff/N) / (N+4 + 2*mueff/N); % time constant for cumulation for C  C的累积时间常数
cs = (mueff+2) / (N+mueff+5);  % t-const for cumulation for sigma control  西格玛控制中累积的t-const
c1 = 2 / ((N+1.3)^2+mueff);    % learning rate for rank-one update of C   C排名第一更新的学习率
cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)^2+mueff));  % and for rank-mu update   对于秩mu更新
damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs; % damping for sigma
% usually close to 1 通常接近1
% Initialize dynamic (internal) strategy parameters and constants 初始化动态（内部）策略参数和常量
pc = zeros(N,1); ps = zeros(N,1);   % evolution paths for C and sigma  C和sigma的进化路径
B = eye(N,N);                       % B defines the coordinate system  B定义坐标系
D = ones(N,1);                      % diagonal D defines the scaling  对角线D定义缩放
C = B * diag(D.^2) * B';            % covariance matrix C  协方差矩阵C
invsqrtC = B * diag(D.^-1) * B';    % C^-1/2
eigeneval = 0;                      % track update of B and D  跟踪B和D的更新
chiN=N^0.5*(1-1/(4*N)+1/(21*N^2));  % expectation of  预期
%   ||N(0,I)|| == norm(randn(N,1)) 
%out.dat = []; out.datx = [];  % for plotting output  用于绘制输出

% -------------------- Generation Loop ------------产生回路
counteval = 0;  % the next 40 lines contain the 20 lines of interesting code
while counteval < stopeval
    
    % Generate and evaluate lambda offspring   生成和评估lambda子代
    for k=1:lambda
        arx(:,k) = xmean + sigma * B * (D .* randn(N,1)); % m + sig * Normal(0,C)   randn  回归方程  82X23
        %加一个归一化
        
        for kk=1:N
            [X,ps]=mapminmax(x');%确认此处归一化是归到-1到1之间，每行为一个单位，按比例缩小或者放大
            ps.ymin=lb(kk); % 归一化后的最小值0.002    x是96X2
            ps.ymax=ub(kk); % 归一化后的最大值0.996
            %ps.xmin：输入数据 x 的最小值。         ps.xmax：输入数据 x 的最大值。
            %ps.ymin：归一化后的数据的最小值。      ps.ymax：归一化后的数据的最大值。
            ps.yrange=ps.ymax-ps.ymin; % 归一化后的极差,若不调整该值, 则逆运算会出错，yrange表示归一化之后的范围归一化之后的上减下，没修改的话是2，改了的是上面差
            X=mapminmax(x',ps)
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%投影算子%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [aa,b]=size(Aeq);
        a=0;
        for i=1:aa
            if beq(i)==1
                a=a+1;
            end
        end
        num=zeros(1,a);
        for i=1:a
            for j=1:b
                if Aeq(i,j)==1
                    num(i)=num(i)+1;  %记录Aeq中每行1的个数
                    ini(i)=j;
                end
            end
            ini(i)=ini(i)-num(i)+1;   %记录Aeq中每行1的起始列号
        end
        for i=1:a
            A=ones(1,num(i));
            arx(ini(i):(ini(i)+num(i)-1),k)=arx(ini(i):(ini(i)+num(i)-1),k)-A'*inv(A*A')*(A*arx(ini(i):(ini(i)+num(i)-1),k)-1);
            for j=ini(i):(ini(i)+num(i)-1)
                yu=0;
                if arx(j,k)<0
                    
                    arx(j,k)=0;
                    yu=yu+arx(j,k);
                end
            end
            index=0;
            for j=ini(i):(ini(i)+num(i)-1)
                if arx(j,k)~=0
                    index=index+1;
                end
            end
            yu=yu/index;
            for j=ini(i):(ini(i)+num(i)-1)
                if arx(j,k)~=0
                    arx(j,k)=arx(j,k)+yu;
                end
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        arfitness(k) = fun(arx(:,k)'); % objective function call
        %FES=FES+1;
    end
    counteval = counteval+1;
    % Sort by fitness and compute weighted mean into xmean  按适应度排序并将加权平均值计算到xmean中
    [arfitness, arindex] = sort(arfitness);  % minimization
    xold = xmean;
    xmean = arx(:,arindex(1:mu)) * weights;  % recombination, new mean value   重组，新平均值
    
    % Cumulation: Update evolution paths    累积：更新进化路径
    ps = (1-cs) * ps ...
        + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold) / sigma;
    hsig = sum(ps.^2)/(1-(1-cs)^(2*counteval/lambda))/N < 2 + 4/(N+1);
    pc = (1-cc) * pc ...
        + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma;
    
    % Adapt covariance matrix C   自适应协方差矩阵C
    artmp = (1/sigma) * (arx(:,arindex(1:mu)) - repmat(xold,1,mu));  % mu difference vectors mu差分矢量
    C = (1-c1-cmu) * C ...                   % regard old matrix  关于旧矩阵
        + c1 * (pc * pc' ...                % plus rank one update   加上排名第一的更新
        + (1-hsig) * cc*(2-cc) * C) ... % minor correction if hsig==0   如果hsig==0，则进行轻微校正
        + cmu * artmp * diag(weights) * artmp'; % plus rank mu update   加秩mu更新
    
    % Adapt step size sigma   调整步长西格玛
    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1));
    %if sigma<10e-4
    %       sigma=0.5;
    %end
    % Update B and D from C  从C更新B和D
    if counteval - eigeneval > lambda/(c1+cmu)/N/10  % to achieve O(N^2)
        eigeneval = counteval;
        C = triu(C) + triu(C,1)'; % enforce symmetry  强制对称
        [B,D] = eig(C);           % eigen decomposition, B==normalized eigenvectors  特征分解，B==归一化特征向量
        D = sqrt(diag(D));        % D contains standard deviations now  D现在包含标准偏差
        invsqrtC = B * diag(D.^-1) * B';
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


