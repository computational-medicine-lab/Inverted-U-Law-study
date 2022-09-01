function [alph, beta, gamma, rho, sig2e, sig2v, xnew, signewsq, muone, a, stats] ...
    = mixedlearningcurve2(N, Z, background_prob, rhog, alphag, betag, sig2eg, sig2vg, startflag)
%Script to run the subroutines for binomial EM Updated by Anne Smith, Nov
%29, 2010 Michael Prerau Anne Smith, October 15th, 2003
% 
%variables to be reset by user:
%        N                        The discrete process cornum (1 by
%        num_trials) vector of number correct at each trial N(1,:)
%        
%                                 N(2,:)
%
%        Z                        The reaction time (continuous)
%
%        background_prob          probabilty of correct by chance (bias)
%
%        sigv                     sqrt(variance) of random walk

%other variables
%        x, s   (vectors)         hidden process and its variance (forward
%        estimate) xnew, signewsq (vectors) hidden process and its variance
%        (backward estimate) newsigsq                 estimate of random
%        walk variance from EM p      (vectors)         mode of prob
%        correct estimate from forward filter p05,p95   (vectors)      conf
%        limits of prob correct estimate from forward filter b
%        (vectors)         mode of prob correct estimate from backward
%        filter b05,b95   (vectors)      conf limits of prob correct
%        estimate from backward filter

stats = [];
xfilt=[];
cornum = N;



%PARAMETERS starting guess for rho
rho = rhog;  
%starting guess for beta
beta = betag;  
%starting guess for alpha
alph = alphag;  
%starting guess for sige = sqrt(sigma_eps squared)
sig2e = sig2eg;                    
%starting guess for sige = sqrt(sigma_v squared)
sig2v = sig2vg;  
gamma=0;
%set the value of mu from the chance of correct
muone = log(background_prob/(1-background_prob)) ;

%convergence criterion for sigma_eps_squared
cvgce_crit = 1e-3;

%----------------------------------------------------------------------------------
%loop through EM algorithm

xguess = 0;  %starting point for random walk x


for jk=1:3000
    
    %forward filter
    [xfilt, sfilt, xold, sold] = ...
        recfilter(N, Z, sig2e, sig2v, xguess, muone, rho, beta, alph, gamma);
    
    %backward filter
    [xnew, signewsq, a] = backest(xfilt, xold, sfilt, sold);
     
   if (startflag == 0)
        xnew(1) = 0;             %fixes initial value (no bias at all)
        signewsq(1) = sig2v^2;
   elseif(startflag == 2)
        xnew(1) = xnew(2);       %x(0) = x(1) means no prior chance probability
        signewsq(1) = signewsq(2);
   end
   
    %maximization step
    [alph, beta, gamma, rho, sig2e, sig2v, xnew, muone] = ...
         maximize_ic(N, Z, signewsq, xnew, a, muone, startflag);

   
    newsigsq(jk) = sig2v;
    
    signewsq(1) = sig2v;    %updates the initial value of the latent process variance
    
    xnew1save(jk) = xnew(1);
    
    %check for convergence of parameters
    stats = [stats; [alph beta sig2e sig2v]] ;
    if(jk>1)
        diffsv = stats(jk,:) - stats(jk-1,:);
        a1   = mean(abs(diffsv));
        if( diffsv(1) < cvgce_crit && diffsv(2) < cvgce_crit && diffsv(3) < cvgce_crit && diffsv(4) < cvgce_crit)
            fprintf(2, 'EM converged after %d  \n', jk)
            break
        end
    end
    
    xguess = xnew(1);
    
end

if(jk == 3000)
%      figure,plot(stats')
    fprintf(2,'failed to converge after %d steps; convergence criterion was %f \n', jk, cvgce_crit)
end
failed=0;
fprintf(2,' alpha is %f, beta is %f, sigesq is %f, sigvsq is %f \n', alph, beta, sig2e, sig2v);

end


function  [xhat, sigsq, xhatold, sigsqold] ... 
    = recfilter(N, Z, sig2e, sig2v, xguess, muone, rho, beta, alpha, gamma)

    %implements the forward recursive filtering algorithm on the spike
    %train data N variables:
    %        xhatold                    one-step prediction sigsqold
    %        one-step prediction variance xhat
    %        posterior mode sigsq                      posterior variance N
    %        The point process cornum (1 by num_trials)   vector of number
    %        correct at each trial N(1,:) totnum (1 by num_trials)   total
    %        number that could be correct at each trial
    %                                   N(2,:)
    %
    %        Z                          The reaction time (continuous)
    %
    %Parameteres:
    %        rho beta alpha muone

    failed=0;

    T = length(N);
    cornum = N;


    %set up some initial values
    xhat(1) = xguess;    
    sigsq(1) = sig2v;   

    count = 1;

    %number_fail saves the time steps if Newton method fails
    number_fail = [];

    %loop through all time
    for t=2:T+1
        xhatold(t)  = rho*xhat(t-1);
        sigsqold(t) = rho^2*sigsq(t-1) + sig2v;

        %calls x_newtonsolve to find solution to nonlinear posterior
        %prediction estimate
        [xhat(t),flagfail] = x_newtonsolve(muone,  xhatold(t), sigsqold(t), cornum(t-1), ...
             Z(t-1), alpha, beta, rho, gamma, sig2e);

        if flagfail>0
            number_fail = [number_fail t];
        end

        %calculates sigma k squared hat
        sigsq(t) = (1/sigsqold(t) + beta^2/sig2e + gamma^2*exp(muone+gamma*xhat(t))/(1+exp(muone+gamma*xhat(t)))^2)^-1 ;
    end

    if isempty(number_fail)<1
        fprintf(2,'Newton convergence failed at times %d \n', number_fail)
        failed=1;
        p=0;
        xhat=0;
        sigsq=0;
        xhatold=0;
        sigsqold=0;
        return;
    end
end

function [xnew, signewsq, a] = backest(x, xold, sigsq, sigsqold);

    %backward filter variables

    T = size(x,2);

    xnew(T)     = x(T);
    signewsq(T) = sigsq(T);
    for i = T-1 :-1: 2
       a(i)        = sigsq(i)/sigsqold(i+1);
       xnew(i)     = x(i) + a(i)*(xnew(i+1) - xold(i+1));
       signewsq(i) = sigsq(i) + a(i)*a(i)*(signewsq(i+1)-sigsqold(i+1));
    end

end

function  [alph, beta, gamma, rho, sig2e, sig2v, xnew, muone] ... 
    = maximize_ic(N, Z, signewsq, xnew, a, muone, startflag);


    K = length(N);

    gamma=1; %fixed

    %added by ACS 12/02/2010 to deal with different ics on x EM convergence
    %is very sensitive to specification of this part
    M          = K+1;  
    xnewt      = xnew(3:M);
    xnewtm1    = xnew(2:M-1);
    signewsqt  = signewsq(3:M);
    A          = a(2:end);
    covcalc    = signewsqt.*A;
    term1      = sum(xnewt.^2) + sum(signewsqt);
    term2      = sum(covcalc) + sum(xnewt.*xnewtm1);

    if (startflag  == 0)                   %fixed initial condition
     term3      = 2*xnew(2)*xnew(2) + 2*signewsq(2);
     term4      = xnew(end)^2 + signewsq(end);
    elseif( startflag == 2)                %estimated initial condition
     term3      = 1*xnew(2)*xnew(2) + 2*signewsq(2);
     term4      = xnew(end)^2 + signewsq(end);
     M = M-1;
    end

    sig2v   = (2*(term1-term2)+term3-term4)/M;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    WkK = sum(signewsq(2:end)+xnew(2:end).^2);
    Wkm1K = xnew(1)^2+sum(signewsq(2:end-1)+xnew(2:end-1).^2);
    Wkm1kK =xnew(1)*xnew(2)+sum(a(1:end-1).*signewsq(3:end)+xnew(2:end-1).*xnew(3:end)); 
    %  

    ab = inv([K sum(xnew(2:end));sum(xnew(2:end)) WkK])*[sum(Z);sum(xnew(2:end).*Z)];
    alph =ab(1);
    beta  =ab(2);
    sig2e = (1/(K))*(sum(Z.^2)+K*alph^2+beta^2*WkK-2*alph*sum(Z)-2*beta*sum(xnew(2:end).*Z)+2*alph*beta*sum(xnew(2:end)));

    rho = 1; %fixed
end

function [xres, timefail] = x_newtonsolve(muone, xold, sig2old, cornum, z, alpha, beta, rho,gamma, sig2e);

    %Solve for x hat using Newton's method

    timefail = 1; %time when the algorithm fails 

    %Set the initial guess for x hat to the old value of x
    x(1)=xold-rho*xold-((sig2old*beta)/(sig2old*beta^2+sig2e))*(z-alpha-beta*rho*xold)-...
        ((sig2e*sig2old)/(sig2old*beta^2+sig2e))*...
        (cornum - exp(muone)*exp(gamma*xold)/(1+exp(muone)*exp(gamma*xold)));

    for i = 1:400
        %Find x hat
        g(i) = x(i)-rho*xold-((sig2old*beta)/(sig2old*beta^2+sig2e))*(z-alpha-beta*rho*x(i))-...
            ((sig2e*sig2old)/(sig2old*beta^2+sig2e))*...
            (cornum - exp(muone)*exp(gamma*x(i))/(1+exp(muone)*exp(gamma*x(i))));

        %Find the first derivative
        gprime(i) = 1+((sig2e*sig2old)/(sig2old*beta^2+sig2e))*...
            (gamma*exp(muone+gamma*x(i)))/(1+exp(muone+gamma*x(i)))^2;

        %newton's method
        x(i+1)=x(i)-g(i)/gprime(i);
        xres=x(i+1); %Save the result

        %Check for convergence to zero
        if abs(xres-x(i))<1e-10
            timefail = 0; 
            return
        end
    end
    if(i==200) 
        timefail = 1;
        return    
    end
end