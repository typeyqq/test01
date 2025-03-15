%% Third Blood for IEEE Access
clc
clear all
global L
global M
global KK
%global KKK

global AllData
global TrainData
L = 25; %L���ܹ���ĸ���
M = 2; %�������Եĸ���
N = 5;%�ο�ֵ�ĸ���������Ĳο�ֵ
%KKK=zeros(240,3);
%% ==========the training data and testing data input =====
%ѵ�����ݺͲ�����������
[BRB_I ] = generator (1);
% for i = 1:60
%     Tra0inData(i,:) = AllData (2*i,:);
% end
TrainData=load('train1.txt');%��ȡѵ������
AllData=load('test.txt');%��ȡ��������
%Decoupling_I = [ 1 0
%                 0 1 ];        %%% initial value for decoupling model
%                               %%%�����ָ�������������ϵ���ϵ�������˶���ʽ��ͨ���໥���ö��˴�Ӱ�������������������� 
%                               %%%�����������ѧ�����������˶����뿪����������
RuleW_I = ones(L);             %% ����Ȩ�صĳ�ʼֵ,16X16��1
%��ȡ����Ȩ��
AttributeW_I = ones(M);        %% ����Ȩ�صĳ�ʼֵ��2X2��1
%��ȡ����Ȩ��
%% =======the optimization model==========�Ż�ģ��
for k=1:L            %L������
    for n=1:N        %%%%N������
        x0((k-1)*N+n)=BRB_I(k,n);   %%% belief degree
        %16X4������룬�ٶ����������Ե�Ȩ��
    end
end
%�˴��������֮���ǰѳ�ʼ�ο�ֵ����洢��x0֮��
%ÿ�������ĸ��ο�ֵ16X4=64
for k=1:L
    x0(L*N + k) = RuleW_I (k);      %% rule weight
    %65��80Ⱦ��1
end
%�˴��������16������Ĺ���Ȩ�أ�ȫ��1��
for k=1:M
    x0(L*N + L + k) = AttributeW_I (k);   %% ����Ȩ��
    %81��82Ⱦ��1
end
%�ټ�2��1��
%for k=1:2
%    for n=1:2
%    x0(L*N+L+M+ (k-1)*2 +n) = Decoupling_I (k,n);   %% decoupling weight !!!!����������2*2�Ľ������
%    end
%end
x0 = x0';%ת��
lb=zeros(L*N+L+M ,1);%82X1
ub=ones(L*N+L+M ,1);%82X1
Aeq=zeros(L+L+M, L*N+L+M);%34X82

for k=1:L
    for n=1:N
        Aeq(k,(k-1)*N+n)=1;%û��ȫ��ֻ�����ָ�ֵ1
    end
end

beq=ones(L+L+M,1);%34X1
for i =1:L+M 
    beq(L+i) = 0;%16������1����Ĺ�0
end
G=100;%ѵ��������     
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
    %X0��������������ԵĲο�ֵ�ټ��Ϲ����1�����Ը�1
    %G��ѵ������  %
    for k=1:L
        for n=1:N
            %BeliefF(k,n)=Xbest((k-1)*N+n);      % the optimized BRB   �Ż���BRB
            %�õ��Ż����brb������������Ŷ�
            xBeliefF(k,n)=ybest((k-1)*N+n);
        end
    end
    for k=1:L
        %RuleWF (k)  = Xbest(L*N+k);      % the opeimized rule weight    opeimized����Ȩ��
        yRuleWF (k)  = ybest(L*N+k); 
    end
    for k = 1:M
        %AttributeWF(k) = Xbest(L*N+L+k);  % the optimized attribute weight   �Ż�������Ȩ��
        yAttributeWF(k) = ybest(L*N+L+k);
    end
    %�������һ���16X4+16+2
    MSE(m)=fun_test(Xbest);
    Z(:,m)=KK';
    yMSE(m)=fun_test(ybest);
    Y(:,m)=KK';
    figure(m);
    k=1:96;
    %kΪ1��240
    %�������ݵĸ�����test��Ĳ������ݵĸ���
    plot(k,AllData(k,3),'.',k,Z(k),'r',k,Y(k),'g');
    %plot(k,AllData(k,3),'.',k,Y(k),'g');
    
    %���Ƶ�ͼ
    %��Ϊ100��ϸ50����Ϊʵ��
end
yMSE(m) = fun_test(ybest)
MSE(m) = fun_test(Xbest)
A=MSE(m)/yMSE(m)  %�õ��Ż�����
