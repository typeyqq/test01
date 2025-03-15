%% Third Blood for IEEE Access
clc
clear all
global L
global M
global KK
%global KKK

global AllData
global TrainData
L = 25; %L是总规则的个数
M = 2; %输入属性的个数
N = 5;%参考值的个数，输入的参考值
%KKK=zeros(240,3);
%% ==========the training data and testing data input =====
%训练数据和测试数据输入
[BRB_I ] = generator (1);
% for i = 1:60
%     Tra0inData(i,:) = AllData (2*i,:);
% end
TrainData=load('train1.txt');%读取训练数据
AllData=load('test.txt');%读取测试数据
%Decoupling_I = [ 1 0
%                 0 1 ];        %%% initial value for decoupling model
%                               %%%耦合是指两个或两个以上的体系或两种运动形式间通过相互作用而彼此影响以至联合起来的现象。 
%                               %%%解耦就是用数学方法将两种运动分离开来处理问题
RuleW_I = ones(L);             %% 规则权重的初始值,16X16的1
%读取规则权重
AttributeW_I = ones(M);        %% 属性权重的初始值，2X2的1
%读取属性权重
%% =======the optimization model==========优化模型
for k=1:L            %L条规则
    for n=1:N        %%%%N条属性
        x0((k-1)*N+n)=BRB_I(k,n);   %%% belief degree
        %16X4整体放入，假定的输入属性的权重
    end
end
%此处运行完毕之后是把初始参考值横向存储在x0之中
%每个规则四个参考值16X4=64
for k=1:L
    x0(L*N + k) = RuleW_I (k);      %% rule weight
    %65到80染上1
end
%此处后面接上16个规则的规则权重，全是1的
for k=1:M
    x0(L*N + L + k) = AttributeW_I (k);   %% 属性权重
    %81和82染上1
end
%再加2个1的
%for k=1:2
%    for n=1:2
%    x0(L*N+L+M+ (k-1)*2 +n) = Decoupling_I (k,n);   %% decoupling weight !!!!仅仅局限于2*2的解耦矩阵
%    end
%end
x0 = x0';%转置
lb=zeros(L*N+L+M ,1);%82X1
ub=ones(L*N+L+M ,1);%82X1
Aeq=zeros(L+L+M, L*N+L+M);%34X82

for k=1:L
    for n=1:N
        Aeq(k,(k-1)*N+n)=1;%没有全给只给部分赋值1
    end
end

beq=ones(L+L+M,1);%34X1
for i =1:L+M 
    beq(L+i) = 0;%16规则内1，外的归0
end
G=100;%训练的轮数     
g=100;
A = [];b = [];
for m=1:1
    Xbest = yqqtwo(x0,G,Aeq,beq,ub,lb);
    %Xbest = test_qqq(x0,G,Aeq,beq,ub,lb);
    %Xbest = ttext_to_cmaes(x0,G,Aeq,beq,ub,lb);
    %Xbest = test_p_cma_es(x0,G,Aeq,beq,ub,lb);
    %Xbest = test_down(x0,G,Aeq,beq,ub,lb);
    ybest = test_p_cma_es(x0,g,Aeq,beq,ub,lb);
    %ybest = yuanzhuang(x0,g,Aeq,beq,ub,lb);
    %X0规则乘以输入属性的参考值再加上规则个1和属性个1
    %G是训练轮数  %
    for k=1:L
        for n=1:N
            %BeliefF(k,n)=Xbest((k-1)*N+n);      % the optimized BRB   优化的BRB
            %得到优化后的brb个个规则的置信度
            xBeliefF(k,n)=ybest((k-1)*N+n);
        end
    end
    for k=1:L
        %RuleWF (k)  = Xbest(L*N+k);      % the opeimized rule weight    opeimized规则权重
        yRuleWF (k)  = ybest(L*N+k); 
    end
    for k = 1:M
        %AttributeWF(k) = Xbest(L*N+L+k);  % the optimized attribute weight   优化的属性权重
        yAttributeWF(k) = ybest(L*N+L+k);
    end
    %分离合在一起的16X4+16+2
    MSE(m)=fun_test(Xbest);
    Z(:,m)=KK';
    yMSE(m)=fun_test(ybest);
    Y(:,m)=KK';
    figure(m);
    k=1:96;
    %k为1到240
    %测试数据的个数，test里的测试数据的个数
    plot(k,AllData(k,3),'.',k,Z(k),'r',k,Y(k),'g');
    %plot(k,AllData(k,3),'.',k,Y(k),'g');
    
    %绘制点图
    %粗为100，细50，点为实际
end
yMSE(m) = fun_test(ybest)
MSE(m) = fun_test(Xbest)
A=MSE(m)/yMSE(m)  %得到优化参数
