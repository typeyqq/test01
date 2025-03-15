AllData=load('mse4_add.txt');%mse2
AllDatanew=load('dierpian.txt');
figure(1);
    k=1:42;
    % 绘制第一条线
plot(k, AllData(k, 1), '-ob', 'LineWidth', 1); 
hold on;

% 绘制第二条线
plot(k, AllData(k, 2), '-+r', 'LineWidth', 1);

% 绘制第三条线
plot(k, AllData(k, 3), '-sg', 'LineWidth', 1);

% 绘制第四条线
plot(k, AllData(k, 4),'Color',[0.5 0.5 0], 'LineWidth', 0.5);

% 绘制第五条线
plot(k, AllData(k, 5), 'Color',[0.5 0.25 0.25], 'LineWidth', 1);

% 绘制第六条线
plot(k, AllData(k, 6), 'Color',[0.25 0.25 0.5], 'LineWidth', 1);

% 绘制第七条线
plot(k, AllData(k, 7), 'Color',[0.3 0.5 0.2], 'LineWidth', 1);

% 绘制第八条线
plot(k, AllData(k, 8), 'Color',[0 0.5 0.5], 'LineWidth', 1);

% 绘制第九条线
plot(k, AllData(k, 9), 'Color',[0 0.25 0.75], 'LineWidth', 1);

% 绘制第十条线
plot(k, AllData(k, 10), 'Color',[0.1 0.4 0.5], 'LineWidth', 1);

% 绘制第十一条线
plot(k, AllData(k, 11), 'Color',[0.4 0.5 0.1], 'LineWidth', 1);

% 绘制第十二条线
plot(k, AllData(k, 12), 'Color',[0.8 0.1 0.1], 'LineWidth', 1);


% 添加标签和标题
%ylabel('The value of the industrial internet security situation');
ylabel('工业互联网安全形势的价值');
%xlabel('Group');
xlabel('组');
legend('实际值', 'S-CMA-ES模型','BRB-s模型');

clc
