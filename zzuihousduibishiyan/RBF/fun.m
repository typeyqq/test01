function err = fun( x )
inputnum=2;
hiddennum=10;
outputnum=1;
x1=load('train.txt')
y1=load('train_label.txt')
% x2=load('test.txt')
% y2=load('test_label.txt')
y1=y1'
% y2=y2'
%½¨Ä£
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=reshape(B2, outputnum,1);
err_goal=0.01;
sc=1;
net=newrb(x1',y1,err_goal,sc,20,1);
%²âÊÔ
ty=sim(net,x1');
tE=y1-ty;
err=mse(tE);
end

