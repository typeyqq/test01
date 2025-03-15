function xbest = ga( DIM )
DIM=DIM;
popsize=10;
xmax=1;
xmin=0;
%初始化种群
for k=1:popsize
 A(k,:)=rand(1,DIM);
end

for d=1:50  %进化代数
%计算染色体适应度值
for i=1:popsize
   f(i,1)=fun(A(i,:));
end
%寻找最有个体
[min_f, index] = min(f);
if d==1
 fbest=min_f;
 xbest=A(index,:);
end
if min_f<fbest
 fbest=min_f;
 xbest=A(index,:);
end
%选择
%计算每个染色体的选择概率
  ps=zeros(popsize,1);
  sum=0;
  for i=1:popsize
     sum=sum+f(i,1);
  end
  for i=1:popsize
     ps(i,1)=f(i,1)/sum;
  end
  %采用轮盘赌法选择新个体
  index1=[]; 
  for i=1:popsize  
      pick=rand;
      while pick==0    
         pick=rand;        
      end
      for i=1:popsize    
         pick=pick-ps(i,1);        
         if pick<0        
            index1=[index1 i];            
            break;  
         end
      end
  end
  A=A(index1,:);
  ps=ps(index1,1);
%交叉
pc=0.3;
for i=1:popsize
   pick=rand(1,DIM);
   while prod(pick)==0
     pick=rand(1,DIM);
   end
   index2=ceil(pick.*popsize);
   pick=rand;
     while pick==0      
         pick=rand;
     end
   if pick>pc
      continue;
   end
   pos=ceil(pick.*DIM); 
   B=A(index2(1),1:pos);
   A(index2(1),pos:DIM)=A(index2(2),pos:DIM);
   A(index2(2),1:pos)=B;
 end 
%变异
pm=0.15;
  for i=1:popsize   
    pick=rand;
    while pick==0
        pick=rand;
    end
    index3=ceil(pick*popsize);
    pick=rand;
    if pick>pm
        continue;
    end
 % 变异位置
        pick=rand;
        while pick==0      
            pick=rand;
        end
        pos=ceil(pick*DIM);
        %变异开始    
        A(index3,pos)=200*rand(1,1)-100;
   end
for i=1:popsize
   f(i,1)=fun(A(i,:));
end
%寻找最有个体
[min_f, index] = min(f);
if d==1
 fbest=min_f;
 xbest=A(index,:);
end
if min_f<fbest
 fbest=min_f;
 xbest=A(index,:);
end
end
end

