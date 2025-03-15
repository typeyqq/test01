% 假设有两个数据集 data1 和 data2
function [w]=yqq_quanzhong5()

x=load('Raw_Data.txt');
[X,ps]=mapminmax(x');%确认此处归一化是归到-1到1之间，每行为一个单位，按比例缩小或者放大
ps.ymin=0.002; % 归一化后的最小值0.002
ps.ymax=0.996; % 归一化后的最大值0.996
%ps.xmin：输入数据 x 的最小值。         ps.xmax：输入数据 x 的最大值。
%ps.ymin：归一化后的数据的最小值。      ps.ymax：归一化后的数据的最大值。
ps.yrange=ps.ymax-ps.ymin; % 归一化后的极差,若不调整该值, 则逆运算会出错，yrange表示归一化之后的范围归一化之后的上减下，没修改的话是2，改了的是上面差
x=mapminmax(x',ps)';%按照规定的上下限返回
% 假设数据存储在矩阵data中，每列为一组数据

%x=load('Raw_Data.txt');
% 假设 x 包含两组数据，每组数据为 96x1
% 数据标准化
%x = (x - mean(x)) ./ std(x);
% 计算变异系数
cv = std(x) ./ mean(x);%mean函数是一个求数组平均值,std是计算标准差。
% 计算权重
w = cv ./ sum(cv);
% 求综合得分
end

