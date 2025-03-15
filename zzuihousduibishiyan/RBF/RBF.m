%RBF‘§≤‚ƒ£–Õ
clear
clc

traini=load('train9.txt');
testi=load('test9.txt');

x1=traini(:,1:2);
y1=traini(:,3);
x2=testi(:,1:2);
y2=testi(:,3);

y1=y1';
y2=y2';


err_goal=0.00001;
sc=1;
% for m=1:20

net=newrb(x1',y1,err_goal,sc,30,1);
%≤‚ ‘
RBFoutput(1,:)=sim(net,x2');

% end
MSE=mse(y2-RBFoutput)
RBFoutput=RBFoutput';

k=1:11;
 plot(k,y2(k),k,RBFoutput(k),'-*');
 grid on;
