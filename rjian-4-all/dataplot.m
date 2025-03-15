clc
clear all
%data = load('data1.txt');
data = load('Utility.txt');
k=1:96
% plot(k,data(k,1),'-',k,data(k,2),'-*');
plot(k,data(k,1),'-');
xlabel("组");
ylabel("BRB模型的前提属性1"); 
%ylabel("The premise attribute2 of BRB model"); 
%ylabel("The practical value of Industrial Internet security situation"); 