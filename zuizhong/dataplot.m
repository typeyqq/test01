clc
clear all
data = load('Utility.txt');
k=1:96
%data = load('data.txt');
%k=1:160

% plot(k,data(k,1),'-',k,data(k,2),'-*');
plot(k,data(k,1),'.-','LineWidth',1.2);
grid on
set(gca,'fontsize',10,'fontname','Times');
%xlabel("Group",'fontname', 'times new roman','fontSize',13);
xlabel('组', 'FontName', 'SimHei');

%ylabel("The premise attribute1 of BRB model",'fontname', 'times new roman','fontSize',13); 

ylabel('工业互联网网络安全评估的实际价值', 'FontName', 'SimHei');

% ylabel("The premise attribute2 of BRB model",'fontname', 'times new roman','fontSize',13); 
% ylabel("The practical value of the Industrial Internet security situation",'fontname', 'times new roman','fontSize',13); 