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

% --------------------------------------------
% fobj = @YourCostFunction
% dim = number of your variables
% Max_iteration = maximum number of iterations
% SearchAgents_no = number of search agents
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
% ---------------------------------------------------------

clear all
clc
format long

N=50; % Number of search agents

Function_name='F1';

[lb,ub,dim,fobj]=CEC2020(Function_name);

maxFEs=dim*10000; % Maximum number of evaluations

[gbestval,gbest,Conv]=GLS_MPA(N,maxFEs,lb.*ones(1,dim),ub.*ones(1,dim),dim,fobj);

% Convergence curve
semilogy(Conv,'Color','r')
title('Objective space')
xlabel('Iteration');
ylabel('Best score obtained so far');


display(['The best solution obtained by GLS_MPA is : ', num2str(gbest,10)]);
display(['The best optimal value of the objective function found by GLS_MPA is : ', num2str(gbestval,10)]);
disp('--------------------------------------');
