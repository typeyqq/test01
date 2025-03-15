close  all;
clear all;
clc;

data=load('data.txt');
jg=load('jg.txt')
data=data;
 data_zqs=zeros(20,1);
 data_zql=zeros(20,1);

for m=1:20
   for n=1:40
         if abs(jg(n)-data(n,m))<=1
            data_zqs(m,1) =data_zqs(m,1)+1
         end
    end
end
for m=1:20
    data_zql(m)=sum(data_zqs(m,1))/40;
end
 
