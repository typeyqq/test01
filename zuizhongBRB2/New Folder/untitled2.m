x = 1:10;          % x 数据
y1 = x;            % 第一条线的 y 数据
y2 = x.^2;         % 第二条线的 y 数据

% 绘制第一条线并加粗
plot(x, y1, 'b', 'LineWidth', 2);
hold on;

% 绘制第二条线并加粗
plot(x, y2, 'r', 'LineWidth', 2);

% 添加图例、标题和坐标轴标签
legend('y = x', 'y = x^2');
title('多条线加粗示例');
xlabel('x');
ylabel('y');

% 关闭 hold 状态
hold off;
