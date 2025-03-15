clear all;
close all;
clc;

traini=load('train9.txt');
testi=load('test9.txt');

train_data=traini(:,1:55);
train_label=traini(:,56);
test_data=testi(:,1:55);
y2=testi(:,56);

% train_data=load('data1.txt')
% train_label=load('data1_label.txt')
% test_data=load('test1.txt')
% y2=load('test1_label.txt')

factor = TreeBagger(5,train_data,train_label); %5是比较好的参数了 可以不用调
[predlabel,scores] = predict(factor,test_data);
s=str2num(char(predlabel));
b=mse(s-y2);
MSE=b
a=sqrt(mean((s-y2).^2)); 
c=1-sqrt(mean(abs((s-y2))/y2.^2));
d=mean(c);
k=1:32;
 plot(k,y2(k),k,s(k),'-*');
 grid on;
