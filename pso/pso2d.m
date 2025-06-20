global l1_weight l1_bias l2_weight l2_bias l3_weight l3_bias l4_weight l4_bias ...
       desired_g l5_weight l5_bias l6_weight l6_bias de1_weight de1_bias de2_weight de2_bias all_fitness%#ok<*NUSED> 
load weight.mat
dataset=table2array(readtable('ori_g.xlsx'));
index=3;
desired_g=dataset(index,:);
pop=50;
dim=2;
constraint_dim=4;
%%Length Diameter
ub=[110,30];
lb=[30,10];

vmax=[30,10];
vmin=[-30,-10];
a=0.8;
b=0.8;
vmax=a*vmax;
vmin=b*vmin;
maxIter=50;
all_fitness=ones(pop,maxIter);

global position bp
position=zeros(maxIter,pop,2);
bp=zeros(maxIter,2);
fobj=@(X)fun(X);
times=zeros(maxIter,1);
[Best_Pos,Best_fitness,IterCurve]=pso(pop,dim,constraint_dim,ub,lb,...
    fobj,vmax,vmin,maxIter);
figure
plot(IterCurve,'r','linewidth',2);
grid on;
Real=dataset(index,1:6);
Result=Best_Pos;
disp("real: ")
Real
disp("prediction: ")
Result
x=squeeze(position(:,:,1));
y=squeeze(position(:,:,2));
aa=[Real Result];
pg=forward(Best_Pos);

plot(desired_g);
hold on
plot(pg)
hold off

function soc=soc_cal(d1,d2)
        sum_min= sum(min(d1,d2));
        sum_max= sum(max(d1,d2));
    soc=1-sum_min/sum_max;
end

function errors=mse(d1,d2)
    errors=sum((d1-d2).^2);
end

function [Best_Pos,Best_fitness,IterCurve]=pso(pop,dim,cdim,ub,lb,...
    fobj,vmax,vmin,maxIter)
global position bp all_fitness
best_position_iter=zeros(maxIter,2);
IterCurve=ones(1,maxIter);
c1=1.4;
c2=1.4;
wmax=0.9;
wmin=0.1;
V=initialization(pop,vmax,vmin,dim);
X=initialization(pop,ub,lb,dim);
Constrain_variables=zeros(pop,cdim);
Com=[Constrain_variables X];
%%calculate 1-4 value
for i=1:pop
    p1=rand(1,1);
    Com(i,2)=get_p(p1);
    n=round(rand(1,1)*3)+7;
    n1=round(p1*6+1);
    Com(i,4)=n;
    Com(i,3)=d_calculate(n1,Com(i,6),n);
    Com(i,1)=Com(i,5)/Com(i,6);
end
%%get initial fitness
fitness=zeros(1,pop);
for i=1:pop
    fitness(i)=fobj(Com(i,:));
end
pBest=Com(:,5:6);
pBestFitness=fitness;
[~,index]=min(fitness);
gBestFitness=fitness(index);
gBest=pBest(index,:);
GBEST=Com(index,:);
Xnew=pBest;
fitnessNew=fitness;
for t=1:maxIter
    tic
    w=wmax-(wmax-wmin)*t/maxIter;
    for i=1:pop
        local_best_p=1;
        local_best_n=1;
        local_best_fitness=inf;
        r1=rand(1,dim);
        r2=rand(1,dim);
        V(i,:)=w.*V(i,:)+c1.*r1.*(pBest(i,:)-X(i,:))+c2.*r2.*(gBest-X(i,:));
        V(i,:)=BoundaryCheck(V(i,:),vmax,vmin,dim);
        Xnew(i,:)=X(i,:)+V(i,:);
        Xnew(i,:)=BoundaryCheck(Xnew(i,:),ub,lb,dim);
        Com=[Constrain_variables Xnew];
        Com(i,1)=Com(i,5)/Com(i,6);
        for j=1:1:7
            Com(i,2)=pitch(j);
            for n=6:1:25
                Com(i,4)=n;
                Com(i,3)=d_calculate(j,Com(i,6),n);
                fitnessNew(i)=fobj(Com(i,:));
                if fitnessNew(i)<local_best_fitness
                    local_best_fitness=fitnessNew(i);
                    local_best_p=j;
                    local_best_n=n;
                end
            end
        end
        Com(i,2)=pitch(local_best_p);
        Com(i,4)=local_best_n;
        Com(i,3)=d_calculate(local_best_p,Com(i,6),local_best_n);
        fitnessNew(i)=fobj(Com(i,:));
        if fitnessNew(i)<pBestFitness(i)
            pBest(i,:)=Xnew(i,:);
            pBestFitness(i)=fitnessNew(i);
        end
        if fitnessNew(i)<gBestFitness
            gBestFitness=fitnessNew(i);
            gBest=Xnew(i,:);
            GBEST=Com(i,:);
        end
        all_fitness(i,t)=fitnessNew(i);
    end
    toc
    X=Xnew;
    Best_Pos=GBEST;
    bp(t,:)=gBest;
    scatter(Xnew(:,1),Xnew(:,2),10,'*');
    position(t,:,1)=Xnew(:,1);
    position(t,:,2)=Xnew(:,2);
    hold on
    Best_fitness=gBestFitness;
    IterCurve(t)=gBestFitness;
    disp(t)
    %Best_fitness
    best_position_iter(t,:)=gBest;

end
plot(best_position_iter(:,1),best_position_iter(:,2),LineWidth=2);
end

function p=pitch(a)
    pz=[407.16 376.31 351.64 339.3 283.78 271.44 252.93];
    p=pz(a);
end

function p=get_p(a)
    pz=[407.16 376.31 351.64 339.3 283.78 271.44 252.93];
    a=a*6;
    a=round(a+1);
    p=pz(a);
end

function d = d_calculate(pitch_index,w_z,n)
    p=[407.16 376.31 351.64 339.3 283.78 271.44 255.93];
    pitch=p(pitch_index);
    d=(pitch-n*w_z)/(n-1);
end

function [X]=initialization(pop,u,l,dim)
X=zeros(pop,dim);
    for i=1:pop
        for j=1:dim
            X(i,j)=(u(j)-l(j))*rand()+l(j);
        end 
    end
end
%%ReLU
function y = relu(x)
    y = max(0, x);
end

function fitness=fun(input)
    global l1_weight l1_bias l2_weight l2_bias l3_weight l3_bias l4_weight l4_bias ...
        l5_weight l5_bias l6_weight l6_bias de1_weight de1_bias de2_weight de2_bias desired_g
    y1=relu(input*l1_weight+l1_bias');
    y2=relu(y1*l2_weight+l2_bias');
    y3=relu(y2*l3_weight+l3_bias');
    y4=relu(y3*l4_weight+l4_bias');
    y5=relu(y4*l5_weight+l5_bias');
    y6=(y5*l6_weight+l6_bias');
    y8=relu(y6*de1_weight+de1_bias');
    output=y8*de2_weight+de2_bias';
    fitness=soc_cal(desired_g,output);
    if(input(3)<input(6)/2)
        fitness=inf;
    end
end

function [X]=BoundaryCheck(X,ub,lb,dim)
    for i=1:dim
        if X(i)>ub(i)
            X(i)=ub(i);
        end
        if X(i)<lb(i)
            X(i)=lb(i);
        end
    end
end
