%--֧����������SVM����������
%--��      �ߣ� ��   ΰ ��S311070053-���繤��ѧԺ-���������̴�ѧ ��
%--Created By: Zhangwei�� Harbin Engineering University,HEU ��
%--Email:  pigishere@foxmail.com
%--Running enviroment: MATLAB 2010b 7.11.0
%--2012-10-17

%%
%--��ձ���
tic
clear all
close all
clc

%%
%------ѵ�����ݵ���ȡ��Ԥ����-----%
load AUVData.txt;                %���ر�·����������
Xo = AUVData(:, 1:6);            %ѡȡԭʼ�����������6�������
Label_orig = AUVData(:, 7);      %ѡȡԭʼ���ݱ�ǩ
[m, n] = size(Xo);

%���� ���ݹ�һ��
Xmax = max(Xo);
Xmin = min(Xo);
X = zeros(m, n);
for i = 1 : m
    X(i,:) = (Xo(i,:)-Xmin)./(Xmax-Xmin); 
end

%���� ���ݱ�׼��
XX = zeros(m, n);
Xmean = mean(X);
Xstd = std(X, 0, 1);
for i = 1 : m
    XX(i,:) = (X(i,:)-Xmean)./Xstd; 
end

%���� ��������ֵ��ȡ
[phi, lambda] = eig(cov(XX));       %Э����õ�������������������
A = diag(lambda);
eigvalue = sort(A, 'descend');      %������������ֵ
% Amax1 = eigvalue(1);
% Amax2 = eigvalue(2);
% Amax3 = eigvalue(3);
phimax1 = phi(:, 6);                %�������ֵ��Ӧ����������
phimax2 = phi(:, 5);
phimax3 = phi(:, 4);

%���� ��ά����
X1 = XX * phimax1;
X2 = XX * phimax2;
X3 = XX * phimax3;
data = [X1 X2];                     %ֻѡȡ������������ά������ѵ���Ͳ���
% data = [X1 X2 X3]; 

%���� ��ȡѵ����������ͼ����ʾ
Xn = data(150:455, :);              %��ȡѵ������
Yn = Label_orig(150:455, :);
figure(1)
plot(Xn(:,1), Xn(:,2), 'r*');       %����������ֵ��ͼ
title('ѵ��������');
hold on

%%
                 %------SVM������ѵ��-----%
%���� 1��SVM��������Ҫ����ѡ��
epsilon = 1e-8;
C = 20;
sigma = 0.1;
fprintf('C =\n    %f\n', C);
fprintf('sigma =\n    %f\n', sigma);

%���� 2�����ι滮���
[mn, nn] = size(Xn);
H = zeros(mn, mn);           %��������������H
for i = 1 : mn               %�˺���Ϊ������˺�����K(xi,xj)=exp(-||xi-xj||^2/(2*sigma^2))
    for j = 1 : mn
     H(i,j) = Yn(i)*Yn(j)*exp(-(Xn(i,:)-Xn(j,:))*(Xn(i,:)-Xn(j,:))'/(2*sigma^2));
    end
end                          %����H[i,j] == yi * yj * K(xi,xj)
f = -ones(mn,1);
H = H + 1e-8 * eye(size(H));
lb = zeros(mn,1);
ub = C * ones(mn,1);         %һ�׷�����������Լ����
Aeq = Yn';
beq = 0;
options = optimset;          %������ι滮������options�ṹ�����
options.LargeScale = 'off';
options.Display = 'off';
st = cputime;
tic
%[x,fval,exitflag,output,lambda] = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options)
[alpha_n,fval,exitflag,output]=quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);
time_train = toc;           %ѵ������ʱ��

%%
%���� ѵ��������ӻ�
fprintf('SVMѵ��ʱ�� =\n    %f\n', time_train);
fprintf('����ֵ =\n    %f\n', fval);
fprintf('CPUʱ�� =\n    %f\n', st);
fprintf('ExitFlag =\n    %d\n', exitflag);    %ֹͣ��־��1--�ﵽԤ�ھ��ȣ�

svi_n = find(alpha_n>epsilon & alpha_n<C);    %֧�������±�
n_sv = length(svi_n);                         %֧����������n_sv
fprintf('֧������ռѵ���������ı���=\n    %f%%\n',100*n_sv/mn);
plot(Xn(svi_n,1), Xn(svi_n,2), 'bo');

N_Yn1 = find(Yn == 1);
n_y1 = length(N_Yn1);
rho_y1 = n_y1/mn;
fprintf('ʵ����������ռѵ���������ı���=\n    %f%%\n',100*rho_y1);

%֧����������ؼ���
%���� 1����ȡ֧����������SV_n
Xsv = zeros(n_sv,2);
for i = 1 : n_sv
    Xsv(i,:) = Xn(svi_n(i), :);
end
%���� 2����ȡ֧��������Ӧ��alphaֵ
alpha_sv = zeros(n_sv,1); 
for i = 1 : n_sv
    alpha_sv(i,:) = alpha_n(svi_n(i), :);
end
%���� 3����ȡ��������֧������Xsvz
Num_svz = find(Yn==1 & alpha_n>epsilon & alpha_n<C);
n_svz = length(Num_svz);
Xsvz = zeros(n_svz,2);
for k = 1 : n_svz
    Xsvz(k,:) = Xn(Num_svz(k),:);
end
%���� 4����ȡ��������֧������Xsvf
Num_svf = find(Yn==-1 & alpha_n>epsilon & alpha_n<C);
n_svf = length(Num_svf);
Xsvf = zeros(n_svf,2);
for k = 1 : n_svf
    Xsvf(k,:) = Xn(Num_svf(k),:);
end
%---- 5���������Լ��б���sgn(x)
bz = 0;
for k = 1 : n_svz
    for i = 1 : mn
      bz=bz+alpha_n(i)*Yn(i)*exp(-(Xn(i,:)-Xsvz(k,:))*(Xn(i,:)-Xsvz(k,:))'/(2*sigma^2));
    end
end
bz = 1 - bz/n_svz;

bf = 0;
for k = 1 : n_svf
    for i = 1 : mn
      bf=bf+alpha_n(i)*Yn(i)*exp(-(Xn(i,:)-Xsvf(k,:))*(Xn(i,:)-Xsvf(k,:))'/(2*sigma^2));
    end
end
bf = -1 - bf/n_svf;

b = (bz + bf)/2;
% b = bf;
% b = bz;

%���� ?����ȡѵ������֧������Ysv
Ysv = zeros(n_sv, 1);
for i = 1 : n_sv
    Ysv(i, :) = Yn(svi_n(i), :);
end

ys = 0;
as = 0;
for k = 1 : n_sv
    ys = ys + Ysv(k);
    for i = 1 : mn
      as=as + alpha_n(k)*Yn(i)*exp(-(Xn(i,:)-Xsv(k,:))*(Xn(i,:)-Xsv(k,:))'/(2*sigma^2));
    end
end
b = (ys-as)/n_sv;

%---- 5�������������������
x = sym('x');
y = sym('y');
omega_n = 0;
for i = 1 : mn
    omega_n =omega_n +alpha_n(i)*Yn(i)*exp(-([x,y]-Xn(i,:))*([x,y]-Xn(i,:))'/(2*sigma^2));
end
sgn_n = omega_n + b;
ezplot(sgn_n);
legend('������������', '��֧������','�����������');
title('SVMѵ�����');

%%
           %------SVM����������-----%
%���� ����ԭѵ����֧�������߽�ͼ
figure(2)
% ezplot(sgn_n);
plot(Xn(svi_n,1), Xn(svi_n,2), 'bo');          %����֧�������߽�
hold on
title('֧�������߽�');
xlabel('X1');
ylabel('X2');
hold on

%���� ��ȡ��������
Xt = data(1:149 ,:); 
Yt = Label_orig(1:149 ,:);
[mt, nt] = size(Xt);
plot(Xt(:, 1), Xt(:, 2), 'r*');               %���Ʋ��Լ���ά��ͼ
% legend('�����������','��֧��������',  '�����Լ�����');
legend('��֧��������',  '�����Լ�����');

%���� ���Լ��б�
ft = zeros(mt, 1);
for k = 1 : mt
    for i = 1 : mn
        ft(k) = ft(k) + alpha_n(i)*Yn(i)*exp(-(Xn(i,:)-Xt(k,:))*(Xn(i,:)-Xt(k,:))'/(2*sigma^2));
    end
    ft(k) = ft(k) + b;
end

Fz = find(ft>=epsilon & Yt==1);
nz = length(Fz);
Ff = find(ft<100*epsilon & Yt==-1);
nf = length(Ff);
fprintf('���Ծ���=\n    %f%%\n', 100*(nz + nf)/mt);
time_total= toc;
fprintf('��������ʱ��=\n    %f\n',time_total);
fprintf('��Ȩ���У�\n    <���������̴�ѧ-����ѧԺ-��ΰ>\n    ��������ؼ���˵����\n');