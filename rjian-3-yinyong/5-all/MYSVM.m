%--支持向量机（SVM）二分类器
%--作      者： 张   伟 （S311070053-机电工程学院-哈尔滨工程大学 ）
%--Created By: Zhangwei（ Harbin Engineering University,HEU ）
%--Email:  pigishere@foxmail.com
%--Running enviroment: MATLAB 2010b 7.11.0
%--2012-10-17

%%
%--清空变量
tic
clear all
close all
clc

%%
%------训练数据的提取及预处理-----%
load AUVData.txt;                %加载本路径现有数据
Xo = AUVData(:, 1:6);            %选取原始数据特征项（共6个特征项）
Label_orig = AUVData(:, 7);      %选取原始数据标签
[m, n] = size(Xo);

%—— 数据归一化
Xmax = max(Xo);
Xmin = min(Xo);
X = zeros(m, n);
for i = 1 : m
    X(i,:) = (Xo(i,:)-Xmin)./(Xmax-Xmin); 
end

%—— 数据标准化
XX = zeros(m, n);
Xmean = mean(X);
Xstd = std(X, 0, 1);
for i = 1 : m
    XX(i,:) = (X(i,:)-Xmean)./Xstd; 
end

%—— 数据特征值提取
[phi, lambda] = eig(cov(XX));       %协方差，得到特征向量、特征矩阵
A = diag(lambda);
eigvalue = sort(A, 'descend');      %降序排列特征值
% Amax1 = eigvalue(1);
% Amax2 = eigvalue(2);
% Amax3 = eigvalue(3);
phimax1 = phi(:, 6);                %最大特征值对应的特征向量
phimax2 = phi(:, 5);
phimax3 = phi(:, 4);

%—— 降维处理
X1 = XX * phimax1;
X2 = XX * phimax2;
X3 = XX * phimax3;
data = [X1 X2];                     %只选取两组特征（二维）进行训练和测试
% data = [X1 X2 X3]; 

%—— 提取训练集样本并图形显示
Xn = data(150:455, :);              %提取训练样本
Yn = Label_orig(150:455, :);
figure(1)
plot(Xn(:,1), Xn(:,2), 'r*');       %以两组特征值绘图
title('训练集数据');
hold on

%%
                 %------SVM分类器训练-----%
%—— 1、SVM分类器主要参数选择
epsilon = 1e-8;
C = 20;
sigma = 0.1;
fprintf('C =\n    %f\n', C);
fprintf('sigma =\n    %f\n', sigma);

%—— 2、二次规划求解
[mn, nn] = size(Xn);
H = zeros(mn, mn);           %建立半正定矩阵H
for i = 1 : mn               %核函数为径向基核函数：K(xi,xj)=exp(-||xi-xj||^2/(2*sigma^2))
    for j = 1 : mn
     H(i,j) = Yn(i)*Yn(j)*exp(-(Xn(i,:)-Xn(j,:))*(Xn(i,:)-Xn(j,:))'/(2*sigma^2));
    end
end                          %定义H[i,j] == yi * yj * K(xi,xj)
f = -ones(mn,1);
H = H + 1e-8 * eye(size(H));
lb = zeros(mn,1);
ub = C * ones(mn,1);         %一阶范数软间隔（盒约束）
Aeq = Yn';
beq = 0;
options = optimset;          %定义二次规划函数的options结构体参数
options.LargeScale = 'off';
options.Display = 'off';
st = cputime;
tic
%[x,fval,exitflag,output,lambda] = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options)
[alpha_n,fval,exitflag,output]=quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);
time_train = toc;           %训练所用时间

%%
%—— 训练结果可视化
fprintf('SVM训练时间 =\n    %f\n', time_train);
fprintf('最优值 =\n    %f\n', fval);
fprintf('CPU时间 =\n    %f\n', st);
fprintf('ExitFlag =\n    %d\n', exitflag);    %停止标志（1--达到预期精度）

svi_n = find(alpha_n>epsilon & alpha_n<C);    %支持向量下标
n_sv = length(svi_n);                         %支持向量个数n_sv
fprintf('支持向量占训练集样本的比例=\n    %f%%\n',100*n_sv/mn);
plot(Xn(svi_n,1), Xn(svi_n,2), 'bo');

N_Yn1 = find(Yn == 1);
n_y1 = length(N_Yn1);
rho_y1 = n_y1/mn;
fprintf('实际正类样本占训练集样本的比例=\n    %f%%\n',100*rho_y1);

%支持向量的相关计算
%—— 1、提取支持向量矩阵SV_n
Xsv = zeros(n_sv,2);
for i = 1 : n_sv
    Xsv(i,:) = Xn(svi_n(i), :);
end
%—— 2、提取支持向量对应的alpha值
alpha_sv = zeros(n_sv,1); 
for i = 1 : n_sv
    alpha_sv(i,:) = alpha_n(svi_n(i), :);
end
%—— 3、提取正类样本支持向量Xsvz
Num_svz = find(Yn==1 & alpha_n>epsilon & alpha_n<C);
n_svz = length(Num_svz);
Xsvz = zeros(n_svz,2);
for k = 1 : n_svz
    Xsvz(k,:) = Xn(Num_svz(k),:);
end
%—— 4、提取负类样本支持向量Xsvf
Num_svf = find(Yn==-1 & alpha_n>epsilon & alpha_n<C);
n_svf = length(Num_svf);
Xsvf = zeros(n_svf,2);
for k = 1 : n_svf
    Xsvf(k,:) = Xn(Num_svf(k),:);
end
%---- 5、建立测试集判别函数sgn(x)
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

%—— ?、提取训练样本支持向量Ysv
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

%---- 5、绘制样本分类决策面
x = sym('x');
y = sym('y');
omega_n = 0;
for i = 1 : mn
    omega_n =omega_n +alpha_n(i)*Yn(i)*exp(-([x,y]-Xn(i,:))*([x,y]-Xn(i,:))'/(2*sigma^2));
end
sgn_n = omega_n + b;
ezplot(sgn_n);
legend('：正常样本点', '：支持向量','：分类决策面');
title('SVM训练结果');

%%
           %------SVM分类器测试-----%
%—— 绘制原训练集支持向量边界图
figure(2)
% ezplot(sgn_n);
plot(Xn(svi_n,1), Xn(svi_n,2), 'bo');          %绘制支持向量边界
hold on
title('支持向量边界');
xlabel('X1');
ylabel('X2');
hold on

%—— 提取测试样本
Xt = data(1:149 ,:); 
Yt = Label_orig(1:149 ,:);
[mt, nt] = size(Xt);
plot(Xt(:, 1), Xt(:, 2), 'r*');               %绘制测试集二维点图
% legend('：分类决策面','：支持向量点',  '：测试集样本');
legend('：支持向量点',  '：测试集样本');

%—— 测试集判别
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
fprintf('测试精度=\n    %f%%\n', 100*(nz + nf)/mt);
time_total= toc;
fprintf('程序总需时间=\n    %f\n',time_total);
fprintf('版权所有：\n    <哈尔滨工程大学-机电学院-张伟>\n    引用请务必加以说明！\n');