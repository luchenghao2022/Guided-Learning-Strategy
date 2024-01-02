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

% Input parameters
% gbest     ->Current Best Individual
% gbestval  ->Current best individual fitness value
% pop       ->individual position
% popf      ->individual fitness value
% lu        ->problem boundary information
% V0        ->feedback result
% A         ->Guidance parameters
% fobj      ->fitness value function
% FES       ->number of current evaluations
% FESmax    ->Maximum number of evaluations
% i         ->Which individual has not been updated

% Output 
% gbest     ->Current Best Individual
% gbestval  ->Current best individual fitness value
% pop       ->individual position
% popf      ->individual fitness value
% FES       ->number of current evaluations


function [gbest,gbestval,pop,popf,FES]=GLS(gbest,gbestval,pop,popf,lu,V0,A,fobj,FES,FESmax,i)

[N,dim]=size(pop);
which_dim=V0>A;
pop_new=which_dim.*(gbest+tan(pi.*(rand(N,dim)-0.5)).*(lu(2,:)-lu(1,:))./V0)+...
    (~which_dim).*(rand(N,dim).*(lu(2,:)-lu(1,:)));
for i1 = 1 : N
    pop_new_use=pop_new(i1,:);
    pop_new_use(pop_new_use>lu(2,:))=rand.*(lu(2,pop_new_use>lu(2,:)) - lu(1,pop_new_use>lu(2,:))) + lu(1,pop_new_use>lu(2,:));
    pop_new_use(pop_new_use<lu(1,:))=rand.*(lu(2,pop_new_use<lu(1,:)) - lu(1,pop_new_use<lu(1,:))) + lu(1,pop_new_use<lu(1,:));
    popnew_usef=feval(fobj,pop_new_use);
    FES = FES + 1;
    if (i1<=i && popnew_usef < popf(i1) && isreal(pop_new_use) && sum(isnan(pop_new_use))==0)
        popf(i1) = popnew_usef;
        pop(i1, :) = pop_new_use;
    end
    if (popnew_usef < gbestval && isreal(pop_new_use) && sum(isnan(pop_new_use))==0)
        gbestval = popnew_usef;
        gbest = pop_new_use;
    end
    if FES >= FESmax
        return
    end
end

end