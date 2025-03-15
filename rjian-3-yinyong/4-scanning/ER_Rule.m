function [S]=ER_Rule(r,w,p)
%The analystical ER rule without local ignorance
%input:r: reliability 1xM;       w;weighxtM 1;       P:belief degree MxN 
%output: P_fin: final belief degree 1xN
%无局部无知的分析ER规则
%输入：r：可靠性1xM；w地磅M1；P： 置信度MxN
%输出：P_fin：最终置信度1xN
[m,n]=size(p);
P_fin=zeros(1,n);
mu1=0;
for i=1:n
    temp=1;
    for j=1:m
        temp1=0;
        for h=1:n
            temp1=p(j,h)+temp1;
        end
        temp=((1-r(j))+(w(j)*p(j,i))+(w(j)*(1-temp1)))*temp;
    end
    mu1=temp+mu1;
end
mu2=1;
for i1=1:m
    temp2=0;
    for j1=1:n
        temp2=p(i1,j1)+temp2;
    end
    mu2=((1-r(i1))+w(i1)*(1-temp2))*mu2;
end
mu2=(n-1)*mu2;
mu=mu1-mu2;
mu=1/mu;

for i2=1:n
    sec1=1;
    sec2=1;
    sec3=1;
    for j2=1:m
        temp3=0;
        for h2=1:n
            temp3=p(j2,h2)+temp3;
        end
        sec1=((1-r(j2))+w(j2)*p(j2,i2)+w(j2)*(1-temp3))*sec1;
        sec2=((1-r(j2))+w(j2)*(1-temp3))*sec2;
        sec3=(1-r(j2))*sec3;
    end
    numerator=mu*(sec1-sec2);
    denominator=1-(mu*sec3);
    P_fin(1,i2)=numerator/denominator;
end
    MPtheta=mu*sec3;
    r0min=(1-MPtheta*2)/(1-MPtheta);
    r0max=(1-MPtheta*(1+max(w)))/(1-MPtheta);
    r0=[r0min,r0max];
    S=[r0,P_fin];
    S
end

