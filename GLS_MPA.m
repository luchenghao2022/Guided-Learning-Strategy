% The code has been taken from the study:
%_________________________________________________________________________
%  Marine Predators Algorithm source code (Developed in MATLAB R2015a)
%
%  programming: Afshin Faramarzi & Seyedali Mirjalili
%
% paper:
%  A. Faramarzi, M. Heidarinejad, S. Mirjalili, A.H. Gandomi, 
%  Marine Predators Algorithm: A Nature-inspired Metaheuristic
%  Expert Systems with Applications
%  DOI: doi.org/10.1016/j.eswa.2020.113377
%  
%  E-mails: afaramar@hawk.iit.edu            (Afshin Faramarzi)
%           muh182@iit.edu                   (Mohammad Heidarinejad)
%           ali.mirjalili@laureate.edu.au    (Seyedali Mirjalili) 
%           gandomi@uts.edu.au               (Amir H Gandomi)
%_________________________________________________________________________


%_________________________________________________________________________
%  Guided Learning Strategy source code (Developed in MATLAB R2023a)
%
%  programming: Heming Jia & Chenghao Lu
%
% paper:
%  Heming Jia, Chenghao Lu,
%  Guided learning strategy: A novel update mechanism for metaheuristic algorithms design and improvement
%  
%  DOI:
%  
%  E-mails: jiaheming@fjsmu.edu.cn           (Heming Jia)
%           20210868203@fjsmu.edu.cn         (Chenghao Lu)
%_________________________________________________________________________


function [gbestval,gbest,Conv]=GLS_MPA(N,FESmax,lb,ub,dim,fobj)

gbest=zeros(1,dim); %best individual
gbestval=inf;       %the fitness value of the best individual

Conv=zeros(1,FESmax);
stepsize=zeros(N,dim);
fitness=inf(N,1);

Prey=initialization(N,dim,ub,lb);

Iter=0;
FADs=0.2;
P=0.5;

%% GLS initialization
lu=[lb;ub];                         %problem boundary information

A=30;
FES=0;
C=0;
Cmax=2500;
St=zeros(Cmax,dim);
%% GLS initialization ended

while 1

    for i=1:size(Prey,1)
        
        Flag4ub=Prey(i,:)>ub;
        Flag4lb=Prey(i,:)<lb;
        Prey(i,:)=(Prey(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        fitness(i,1)=feval(fobj,Prey(i,:));
        FES=FES+1;
        if fitness(i,1)<gbestval
            gbestval=fitness(i,1);
            gbest=Prey(i,:);
        end
        Conv(FES) = gbestval;
        if FES>=FESmax
            return
        end

        %------------------- Guided Learning Strategy -----------------
        C=C+1;
        St(C,:)=Prey(i,:);
        if C>=Cmax
            V0=std(St,0,1);
            C=0;
            FESold=FES;
            [gbest,gbestval,Prey,fitness,FES]=GLS(gbest,gbestval,Prey,fitness,lu,V0,A,fobj,FES,FESmax,i);
            Conv(FESold+1:FES) = gbestval;
            if FES >= FESmax
                return
            end
        end

    end

    if Iter==0
        fit_old=fitness;    Prey_old=Prey;
    end
    
    Inx=(fit_old<fitness);
    Indx=repmat(Inx,1,dim);
    Prey=Indx.*Prey_old+~Indx.*Prey;
    fitness=Inx.*fit_old+~Inx.*fitness;
    
    fit_old=fitness;    Prey_old=Prey;
    
    Elite=repmat(gbest,N,1);
    CF=(1-Iter/FESmax)^(2*Iter/FESmax);
    
    RL=0.05*levy(N,dim,1.5);   %Levy random number vector
    RB=randn(N,dim);           %Brownian random number vector
    
    for i=1:size(Prey,1)
        for j=1:size(Prey,2)
            R=rand();
            if Iter<FESmax/3

                stepsize(i,j)=RB(i,j)*(Elite(i,j)-RB(i,j)*Prey(i,j));
                Prey(i,j)=Prey(i,j)+P*R*stepsize(i,j);
                
            elseif Iter>FESmax/3 && Iter<2*FESmax/3
                
                if i>size(Prey,1)/2
                    stepsize(i,j)=RB(i,j)*(RB(i,j)*Elite(i,j)-Prey(i,j));
                    Prey(i,j)=Elite(i,j)+P*CF*stepsize(i,j);
                else
                    stepsize(i,j)=RL(i,j)*(Elite(i,j)-RL(i,j)*Prey(i,j));
                    Prey(i,j)=Prey(i,j)+P*R*stepsize(i,j);
                end
                
            else
                
                stepsize(i,j)=RL(i,j)*(RL(i,j)*Elite(i,j)-Prey(i,j));
                Prey(i,j)=Elite(i,j)+P*CF*stepsize(i,j);
                
            end
        end
    end
    
    for i=1:size(Prey,1)
        
        Flag4ub=Prey(i,:)>ub;
        Flag4lb=Prey(i,:)<lb;
        Prey(i,:)=(Prey(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        fitness(i,1)=feval(fobj,Prey(i,:));
        FES=FES+1;
        if fitness(i,1)<gbestval
            gbestval=fitness(i,1);
            gbest=Prey(i,:);
        end
        Conv(FES) = gbestval;
        if FES>=FESmax
            return
        end

        %------------------- Guided Learning Strategy -----------------
        C=C+1;
        St(C,:)=Prey(i,:);
        if C>=Cmax
            V0=std(St,0,1);
            C=0;
            FESold=FES;
            [gbest,gbestval,Prey,fitness,FES]=GLS(gbest,gbestval,Prey,fitness,lu,V0,A,fobj,FES,FESmax,i);
            Conv(FESold+1:FES) = gbestval;
            if FES >= FESmax
                return
            end
        end

    end
    
    if Iter==0
        fit_old=fitness;    Prey_old=Prey;
    end
    
    Inx=(fit_old<fitness);
    Indx=repmat(Inx,1,dim);
    Prey=Indx.*Prey_old+~Indx.*Prey;
    fitness=Inx.*fit_old+~Inx.*fitness;
    
    fit_old=fitness;    Prey_old=Prey;
    
    if rand()<FADs
        U=rand(N,dim)<FADs;
        Prey=Prey+CF*((lb+rand(N,dim).*(ub-lb)).*U);
    else
        r=rand();  Rs=size(Prey,1);
        stepsize=(FADs*(1-r)+r)*(Prey(randperm(Rs),:)-Prey(randperm(Rs),:));
        Prey=Prey+stepsize;
    end
    
    Iter=FES;

end

end

