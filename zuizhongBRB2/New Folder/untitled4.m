AllData=load('mse2.txt');%BRB-s
%AllData=load('mse4_add.txt');%S-CMA-ES
AllDatanew=load('dierpian.txt');
figure(1);
    k=1:32;
%plot(k,AllData(k,1),'b','LineWidth',2,k,AllData(k,2),'r','LineWidth',2);
%,k,AllData(k,3),'g',k,AllData(k,4),'-*',k,AllData(k,5),'-+',k,AllData(k,6), ...,
   %'-s',k,AllData(k,7),'-o',k,AllData(k,8),'-X',k,AllData(k,9),'-k',k,AllData(k,10),'-^',k,AllData(k,11),'-c');
    %plot(k,AllData(k,1),'b',k,AllData(k,2),'r',k,AllData(k,3),'g');
    %plot(k,AllDatanew(k,1),'b',k,AllDatanew(k,3),'-g',k,AllDatanew(k,2),'-r');
    %plot(k,AllDatanew(k,1),'b',k,AllDatanew(k,2),'-r');
    %legend('Actual value','BRB-s');
    
    % 绘制第一条线
plot(k, AllData(k, 1), '-ob', 'LineWidth', 2); 
hold on;

% 绘制第二条线
plot(k, AllData(k, 2), '-+r', 'LineWidth', 2);

% 绘制第三条线
plot(k, AllData(k, 3), '-sg', 'LineWidth', 2);



% 添加标签和标题
%ylabel('The value of the industrial internet security situation');
%xlabel('Group');
ylabel('工业互联网安全形势的价值');
%xlabel('Group');
xlabel('组');
%legend('实际值', 'S-CMA-ES模型','BRB-s模型');

    
    
    
    %legend('actual value', 'BRB-s','BRB','KNN','BP','RF','WOA','viterbi','blockchain','Digital twin based on deconvolution operation','Sqearman correlation coefficient method');
    legend('实际值', 'BRB-s','BRB');
    
    [p, h, stats] = signtest(AllData(k,1)', AllData(k,6));
% 计算Cohen’s d
d = computeCohensD(AllData(k,1)', AllData(k,6));
%  打印结果
disp(['Wilcoxon signed-rank test P值: ' num2str(p)]);
disp(['Cohen’s d: ' num2str(d)]);
% 定义函数计算Cohen’s d
function d = computeCohensD(x, y)
    n1 = length(x);
    n2 = length(y);
    mean_diff = mean(x) - mean(y);
    s1 = std(x);
    s2 = std(y);
    pooled_std = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1 + n2 - 2)); 
    d = mean_diff / pooled_std;
end