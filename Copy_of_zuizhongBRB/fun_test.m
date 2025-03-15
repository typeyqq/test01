%% ##########  ��⺯��   ���Լ�
function f=fun_test(x)%����ϳɵ�LXM+L+N  16X4+16+2=82
global L
global M
global KK
N = 5;
global AllData %test������

T = length( AllData );
for k=1:L
    for n=1:N
        beta1(k,n)=x((k-1)*N+n);
    end
end
for k = 1:L
    RuleWT(k) = x(L*N +k);
end
for k = 1:M
    AttributeWT(k) = x(L*N+L+k);
end

for n = 1:T
    %% ������Ϣ��ת��Ϊƥ���
    l1 = AllData (n,1);
    l2 = AllData (n,2);
     y1=[3.7 3.019 2.346 1.673 1];    % x1�Ĳο�ֵ   ͬʱȷ�����ֵ��Сֵ�����ݰ���ס ����Ĳο�ֵ
    y2=[2 1.73025 1.4865 1.242 0.999];     % x2�Ĳο�ֵ
    T1=length(y1);
    T2=length(y2);
    In=zeros(L,M);
    for i=1:T1-1
        for j=1:T2-1
            if l1<=y1(i) & l1>y1(i+1)   %%�������������Ϊ�������Ƶ����
                if l2<=y2(j) & l2>y2(j+1)
                    a2=(y1(i)-l1)/(y1(i)-y1(i+1)); %����˵�����Ŷ�
                    a1=(l1-y1(i+1))/(y1(i)-y1(i+1)); %���Ҷ˵�����Ŷ�
                    b2=(y2(j)-l2)/(y2(j)-y2(j+1)); %����˵�����Ŷ�
                    b1=(l2-y2(j+1))/(y2(j)-y2(j+1)); %���Ҷ˵�����Ŷ�
                    for k=1:T1
                        In((k-1)*T2+j,2)=b1;
                        In((k-1)*T2+j+1,2)=b2;
                    end
                    In((i-1)*T2+1:i*T2,1)=a1;
                    In(i*T2+1:(i+1)*T2,1)=a2;
                end
            end
        end
    end
    InputD(: , : , n ) = In ;
    
    for  k = 1:L
        weight(k) = 1;
        for m = 1:M
            weight(k) = weight(k) *  InputD (k,m,n);
        end
        if weight(k) == 0
            AM(k) = 0;
        else
 %          AM(k) =  ( RuleWT(k) * ( InputD(k,1,n) ) ^AttributeWT(1) ) * ( RuleWT(k) * ( InputD(k,2,n) )^AttributeWT(2) ) ;
            AM(k) =  RuleWT(k) * (( InputD(k,1,n) ) ^AttributeWT(1) * ( InputD(k,2,n) )^AttributeWT(2) ) ;
        end
    end
    AU = sum( AM );
    for k =1:L
        ActivationW(n,k) = (AM(k)/ AU );      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%calculate����Ȩ��
    end
     Doutput = [0.6836 0.5 0.4 0.3 0.2]; 
    Sum1 = sum(  beta1' );
    for j = 1:N
        temp1(j) = 1;
        for k = 1:L
            Belief1 (k,j) = ActivationW(n,k) * beta1 (k,j) +1 - ActivationW(n,k) * Sum1(k);      %%%%%%%%ϵ����ǰ�벿��
            temp1(j) = temp1(j) * Belief1(k,j);
            
        end
        %KKK(n,j)=temp1(j);
    end
    
    temp2 = sum (temp1);
    temp3 = 1;
    for k = 1:L
        Belief2(k) = 1 - ActivationW(n,k)* Sum1(k);
        temp3 = temp3 * Belief2(k);                     %%%%%%%%%%%%ϵ���ĺ�벿��
    end
    Value = (temp2 - (N-1) * temp3)^-1;
    temp4 = 1;
    for k = 1:L
        temp4 = temp4 * (1 - ActivationW(n,k));
    end
    for j = 1:N
        %BeliefOut(n,j) = ( Value * ( temp1(j) - temp3)) / ( 1 - Value * temp4);
        BeliefOut(j) = ( Value * ( temp1(j) - temp3)) / ( 1 - Value * temp4);
    end
    %y(n) = Doutput * (BeliefOut(n,:))';    %% BRB���
    y(n) = Doutput * (BeliefOut)';
    %Doutput��Ϊ����ο�ֵ
    KK(n)=y(n);
    
    % hold on;
    
    Mse_original(n) = ( y(n) - AllData(n,3) )^2;
end
f = sum(Mse_original) /T;%���ȡƽ��
end