function f=fun(x)   %主要功能是计算某种误差（可能是均方误差）的值。用于衡量模型的预测值与实际观测值之间的差异

global L  %规则
global M  %属性点
N=5;    %同main，是参考点的个数
%global AllData
global TrainData

T = length( TrainData );

for k=1:L
    for n=1:N
        beta1(k,n)=x((k-1)*N+n);%取更改之后的16个规则的权重
    end
end
for k = 1:L
    RuleWT(k) = x(L*N +k);%取规则权重
end
for k = 1:M
    AttributeWT(k) = x(L*N+L+k);%取属性权重，就是激活权重
end
%beta1、RuleWT、AttributeWT、TrainData这些变量是用于模糊推理系统的权重和参数。
for n = 1:T  %导入测试集            %%至95    %所以每个测试的数据相互之间是独立的
    %% 输入信息的转换为匹配度
    l1 = TrainData (n,1); %由实际的输入和参考值得到第k个规则中的第i个输入xi相对于参考值Aki的置信度。
    l2 = TrainData (n,2);
    y1=[3.67 3.019 2.346 1.673 1];    % x1的参考值   同时确保最大值最小值把数据包裹住 输入的参考值
    y2=[1.98 1.73025 1.4865 1.242 0.999];     % x2的参考值
    T1=length(y1);
    T2=length(y2);
    In=zeros(L,M);
    for i=1:T1-1          %这两for是根据给出的n定位到满足卡在对应的参考值之间的参考值，所以一个给出的n只会得到一个大循环
        for j=1:T2-1
            if l1<=y1(i) & l1>y1(i+1)   %%对于两个输出均为上升趋势的情况
                %寻找train里的第n行第1列的数据位于参考值两者之间
                if l2<=y2(j) & l2>y2(j+1)
                    %寻找train里的第n行第2列的数据位于何参考值之间
                    a2=(y1(i)-l1)/(y1(i)-y1(i+1)); %对左端点的置信度  例如0.95位于1与0.8之间 （1-0.95）/（1-0.8）
                    a1=(l1-y1(i+1))/(y1(i)-y1(i+1)); %对右端点的置信度 例如0.95位于1与0.8之间 （0.95-0.8）/（1-0.8）
                    b2=(y2(j)-l2)/(y2(j)-y2(j+1)); %对左端点的置信度
                    b1=(l2-y2(j+1))/(y2(j)-y2(j+1)); %对右端点的置信度
                    for k=1:T1
                        In((k-1)*T2+j,2)=b1;    %首先是第j个然后每隔4个就是每隔一个输出的参考值进行一次赋值
                        In((k-1)*T2+j+1,2)=b2;  %在上面的基础之上加一
                    end
                    In((i-1)*T2+1:i*T2,1)=a1;   %由于测试机的第一个数据并不在1与2的参考值之间所以此处的第一次的i为2，到2*T2的位置
                    In(i*T2+1:(i+1)*T2,1)=a2;   %
                end
            end
        end
    end
    InputD(: , : , n ) = In ;%三维数组，每一个测试数集对应一个16X2的数组，为什么是2，2个输入的属性
    %把一组
    
    for  k = 1:L     %在大的120数据的循环之下对16个规则进行循环--66
        weight(k) = 1;
        for m = 1:M
            weight(k) = weight(k) *  InputD (k,m,n);  %置信度权重之积得到M的规则重量，所以此处的是显示的是规则作为循环单位
            
        end
        if weight(k) == 0
            AM(k) = 0;
        else
            %AM(k) =  ( RuleWT(k) * ( InputD(k,1,n) ) ^AttributeWT(1) ) * ( RuleWT(k) * ( InputD(k,2,n) )^AttributeWT(2) ) ;
            AM(k) =  RuleWT(k) * (( InputD(k,1,n) ) ^AttributeWT(1) * ( InputD(k,2,n) )^AttributeWT(2) ) ;
            %InputD(k,1,n)在第n个数据中，在第k个规则下，两个属性权重之中的一个相较于参考值的置信度
        end
    end
    AU = sum( AM );      %AM和AU前一个是激活权重的分子，后一个是执行规则的分母。
    %AU：AU 代表适应性更新，用于调整模糊规则的激活度。AU 是 PCMAES算法的一部分，是自适应优化算法的核心组成部分。
    %它的目标是通过在搜索过程中动态地调整规则的激活度，以更好地适应目标函数的拓扑特性。
    %AU 可以帮助算法更有效地搜索最优解，因为它根据不同的问题和搜索进展调整了规则的激活度，以更好地适应目标函数的特性。
    %AM (Adaptation Matrix)：AM 代表适应性矩阵，也是 PCMAES 算法的一部分。适应性矩阵用于调整种群中个体的变异步长，
    %以便更好地适应目标函数的拓扑特性。在这个上下文中，AM 用来调整模糊规则的权重，以便更好地匹配输入和输出之间的关系。
    %AM 的更新是算法的关键步骤，它帮助跟踪目标函数的拓扑结构，以便更好地搜索最优解。
    %总之，AU 和 AM 是用于自适应地调整模糊规则的参数，以提高模糊推理系统的性能。它们帮助算法更好地适应不同的优化问题，
    %并在搜索过程中调整规则权重和激活度，以更有效地找到最佳解决方案。这种自适应性可以提高算法的搜索效率和收敛性。
    for k =1:L
        ActivationW(n,k) = (AM(k)/ AU );      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%calculate激活权重
        %ActivationW 是一个表示激活权重的变量。激活权重用于确定每个模糊规则在模糊推理系统中的激活程度。
        %在模糊推理系统中，每个规则的激活程度取决于输入数据与规则的匹配度，激活权重用于加权不同规则的输出，以生成最终的系统输出
    end
    Doutput = [0.6836 0.5 0.4 0.3 0.2];   %此处是输出的参考值
    Sum1 = sum(  beta1' );
    for j = 1:N
        temp1(j) = 1;
        for k = 1:L
            Belief1 (k,j) = ActivationW(n,k) * beta1 (k,j) +1 - ActivationW(n,k) * Sum1(k);      %%%%%%%%系数的前半部分
            temp1(j) = temp1(j) * Belief1(k,j);
        end
    end
    temp2 = sum (temp1);
    temp3 = 1;
    for k = 1:L
        Belief2(k) = 1 - ActivationW(n,k)* Sum1(k);
        temp3 = temp3 * Belief2(k);                     %%%%%%%%%%%%系数的后半部分
    end
    Value = (temp2 - (N-1) * temp3)^-1;
    temp4 = 1;
    for k = 1:L
        temp4 = temp4 * (1 - ActivationW(n,k));
    end
    for j = 1:N
        BeliefOut(j) = ( Value * ( temp1(j) - temp3)) / ( 1 - Value * temp4);    %书P12的相较于结果的置信度（2-6）
    end
    y(n) = Doutput * (BeliefOut)';    %% BRB输出，预测输出值y(n)
    Mse_original(n) = ( y(n) - TrainData(n,3) )^2;  %计算每个样本的预测输出值 y(n) 与实际观测值 TrainData(n,3) 之间的差异，并将其平方
end
f = sum(Mse_original) /T;
end