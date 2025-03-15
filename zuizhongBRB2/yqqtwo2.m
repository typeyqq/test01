function [xmean, sigma, mueff, bestx] = yqqtwo2(N,xmean, sigma, yqq1, mu, mueff,G,Aeq,beq,weights,ub,lb)
% 策略参数设置：自适应
stopfitness = 0.002;
d=10;
% 方法1: 基于问题规模的设定
some_scale_factor = 0.5; % 根据问题调整该参数
d = sqrt(N) * some_scale_factor;

% 方法2: 基于标准差的设定
%some_scale_factor = 0.5; % 根据问题调整该参数
%d = sigma * mean(D) * some_scale_factor;

L = 25; %L是总规则的个数
stopeval = G;         %迭代次数
cc = (4 + mueff/N) / (N+4 + 2*mueff/N); %mueff用于控制协方差矩阵的自适应调整
cs = (mueff+2) / (N+mueff+5);  %%！！！

c1 = 2 / ((N+1.3)^2+mueff);    % learning rate for rank-one update of C   C排名第一更新的学习率
cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)^2+mueff));  %￥￥￥￥￥￥￥￥￥￥￥￥
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
% 绘制数据分布的密度图


    for k=1:yqq1   %%%---147
        arx(:,k) = xmean + sigma * B * (D .* randn(N,1)); %B方向，D幅度    82X23
        while sqrt( sum( (xmean(:)-arx(:,k)).^2 ) )>d 
            arx(:,k) = xmean + sigma * B * (D .* randn(N,1)); 
        end
        
 
          for p=1:5:L*5 %根据结果数量修改
                if (arx(p,k)>=arx(p+1,k)&&arx(p+1,k)>=arx(p+2,k)&&arx(p+2,k)>=arx(p+3,k)&&arx(p+3,k)>=arx(p+4,k))... 
                    ||(arx(p,k)<=arx(p+1,k)&&arx(p+1,k)>=arx(p+2,k)&&arx(p+2,k)>=arx(p+3,k)&&arx(p+3,k)>=arx(p+4,k)...
                    ||arx(p,k)<=arx(p+1,k)&&arx(p+1,k)<=arx(p+2,k)&&arx(p+2,k)>=arx(p+3,k)&&arx(p+3,k)>=arx(p+4,k))...
                    ||(arx(p,k)<=arx(p+1,k)&&arx(p+1,k)<=arx(p+2,k)&&arx(p+2,k)<=arx(p+3,k)&&arx(p+3,k)>=arx(p+4,k)...
                    ||arx(p,k)<=arx(p+1,k)&&arx(p+1,k)<=arx(p+2,k)&&arx(p+2,k)<=arx(p+3,k)&&arx(p+3,k)<=arx(p+4,k)) 
                else %找出不符合的置信度
                    for pp=p:p+5
                        %if x(pp)==0
                        if (arx(p,k)>=arx(p+1,k)||arx(p+2,k)>=arx(p+3,k)||arx(p+3,k)>=arx(p+4,k))
                        %arx(p,k)=0;
                        
                        elseif(arx(p,k)<=arx(p+1,k)&&arx(p+1,k)<=arx(p+2,k)&&arx(p+2,k)>=arx(p+3,k))
                        %arx(p,k)=0;
                        end
                        %elseif(arx(p,k)<=arx(p+1,k)&&arx(p+1,k)<=arx(p+2,k)&&arx(p+3,k)<=arx(p+4,k))
                        %arx(p,k)=0;
                        %end
                        end
                        if abs(arx(pp,k)-arx(pp+1,k))<=d/3
                        arx(pp,k)=arx(pp,k);
                        end
                end
          end
        for kk=1:N 
            if arx(kk,k)>ub(kk)
                arx(kk,k)=ub(kk);
            end
            if arx(kk,k)<lb(kk)
                arx(kk,k)=lb(kk);
            end
        end
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
        for i=1:a  %a为16个规则       
            arx(ini(i):(ini(i)+num(i)-1),k)=arx(ini(i):(ini(i)+num(i)-1),k)./sum(arx(ini(i):(ini(i)+num(i)-1),k));       
            norm_BB = norm(arx(ini(i):(ini(i)+num(i)-1),k));
            arx(ini(i):(ini(i)+num(i)-1),k) = arx(ini(i):(ini(i)+num(i)-1),k) / norm_BB;
            arx(ini(i):(ini(i)+num(i)-1),k)=arx(ini(i):(ini(i)+num(i)-1),k)./sum(arx(ini(i):(ini(i)+num(i)-1),k));
        end
        arfitness(k) = fun(arx(:,k)'); % 此处送进去的是经过优化之后的库，返回的是一个衡量优化结果的值，越小的话越好。
    end  
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    counteval = counteval+1;%计数单位，和200进行比较
    [arfitness, arindex] = sort(arfitness);  % 最小化，sort是排序，返回的arfitness是排序之后的，而arindex是索引
    xold = xmean;  %82X11 * 11X1=82X1
    xmean = arx(:,arindex(1:mu)) * weights;  % recombination, new mean value   重组，新平均值,xmean是重量，weights是个体的权值
    
    ps = (1-cs) * ps ...%sqrt是计算给定的平方根
        + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold) / sigma;
    hsig = sum(ps.^2)/(1-(1-cs)^(2*counteval/yqq1))/N < 2 + 4/(N+1);
    pc = (1-cc) * pc ...
        + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma;
    
    % Adapt covariance matrix C   自适应协方差矩阵C
    %artmp 是用于计算协方差矩阵 C 的一部分，它是一组差分向量，用于更新协方差矩阵
    artmp = (1/sigma) * (arx(:,arindex(1:mu)) - repmat(xold,1,mu));  % mu difference vectors mu差分矢量
    C = (1-c1-cmu) * C ...                   % regard old matrix  关于旧矩阵
        + c1 * (pc * pc' ...                % plus rank one update   加上排名第一的更新
        + (1-hsig) * cc*(2-cc) * C) ... % minor correction if hsig==0   如果hsig==0，则进行轻微校正
        + cmu * artmp * diag(weights) * artmp'; % plus rank mu update   加秩mu更新
    
    % Adapt step size sigma   调整步长
    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1));    %exp就是自然常数e为底的指数函数，为0就不变
    sigmazu(counteval)=sigma;
    %ps 是种群中的个体的平均位置，norm(ps) 表示ps的各个元素平方之和开根号
  
    % Update B and D from C  从C更新B和D
    if counteval - eigeneval > yqq1/(c1+cmu)/N/10  % to achieve O(N^2)
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