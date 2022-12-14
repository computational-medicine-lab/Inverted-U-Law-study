

% This script has been updated by Saman Khazaei on Sept 2nd 2022 to 
% perform a regression analysis for the quantified cognitive arousal and performance.
% This code correponds to: 

% [Khazaei, Saman, Md Rafiul Amin, and Rose T. Faghih. "Decoding a Neurofeedback-Modulated Cognitive Arousal 
% State to Investigate Performance Regulation by the Yerkes-Dodson Law." 
% 2021 43rd Annual International Conference of the IEEE Engineering in
% Medicine & Biology Society (EMBC). IEEE, 2021.]

% EM MPP filter function to find the arousal state created by:
% Dilranjan Wickramasuriya and updated by Saman Khazaei on Sept 2nd 2022
% Corresponding publication to cite: 
% [Wickramasuriya, Dilranjan S., and Rose T. Faghih. "A marked point process filtering approach for 
% tracking sympathetic arousal from skin conductance." IEEE Access 8 (2020): 68499-68513.]

% EM ESTIMATION to find performance state using one bin and one con [EM algorithm] has been written based on:
% Michael Prerau and  Anne Smith work so please cite their work if youv are using this code:
% [Prerau, Michael J., et al. 
% "Characterizing learning by simultaneous analysis of continuous and binary measures of performance." 
% Journal of neurophysiology 102.5 (2009): 3060-3072.] 
% And also:
% [Prerau, Michael J., et al. "A mixed filter algorithm for cognitive state estimation from simultaneously 
% recorded continuous and binary measures of performance."
% Biological cybernetics 99.1 (2008): 1-14.]


clear;
close all;
clc;

load('pre_MPP.mat');

load('data.mat')        % behavioral data (cor/inc cor - reaction time)
addpath('dependencies');

sub_num = [4]; % subject number

for subj = sub_num


    MPP_result = MPP_trials(pre_MPP,sub_num); % %RUN EM ESTIMATION to find arousal state using mpp observation [EM algorithm]

       %----------------------------------------------------------------------------------------
       % %RUN EM ESTIMATION to find performance state using one bin and one con [EM algorithm]
      

         pcrit = 0.95;
         n = data(subj).n; 
         backprobg = sum(n)/length(n);
         r = data(subj).r;
         startflag = 2;  %this can take values 0 or 2
                        %0 fixes the initial condition at backprobg 2 initial
                        %condition is estimated
            %these are all starting guesses
           ialpha = r(1); %guess it's the first value of Z
            ibeta  = -1;
            isig2e = 0.05;
            isig2v = 0.05;
            irho = 1; %fixed in this code
            cvgce_crit = 1e-6; %convergence criterion 


        [alph, beta, gamma, rho, sig2e, sig2v, xnew, signewsq, muone, a, stats] ...
                            = mixedlearningcurve2(n, r, backprobg, irho, ialpha, ibeta, isig2e, isig2v, startflag, cvgce_crit);
        %----------------------------------------------------------------------------------------
                   
        x_prf = [xnew(2:round(length(xnew)/2)-1),xnew(round(length(xnew)/2)+1:end-1)];
        

        pre_MPP = MPP_result(subj).pre_MPP;


        signal = pre_MPP(subj).signal_5exp;
        u = pre_MPP(subj).u_5exp;
        tu = pre_MPP(subj).tu_5exp;

        tri_ind_start_stop_5exp = pre_MPP(subj).tri_ind_start_stop_5exp;  

        K = length(u);


        r0 =  MPP_result(subj).r0;
        r1 =  MPP_result(subj).r1;

        x_smth = MPP_result(subj).x_smth;
        v_smth = MPP_result(subj).v_smth;

        p_smth = MPP_result(subj).p_smth;
        r_smth = MPP_result(subj).r_smth;

        % calling timing data

        c = pre_MPP(subj).c;
        v = pre_MPP(subj).v;

        calming = pre_MPP(subj).calming;
        vexing = pre_MPP(subj).vexing;

        calming_period = pre_MPP(subj).calming_period;
        vexing_period = pre_MPP(subj).vexing_period;

        calming_trials_nbacks_period = pre_MPP(subj).calming_trials_nbacks_period; 
        vexing_trials_nbacks_period = pre_MPP(subj).vexing_trials_nbacks_period;

        % mean consideration with respect to tu index

        % general between calming and vexing

        trial_number = length(tri_ind_start_stop_5exp);
        L1 = 0;
        for tn = 1:trial_number
            x_smth_ave_jt(tn) = mean(x_smth(tri_ind_start_stop_5exp(tn,1):tri_ind_start_stop_5exp(tn,2)));
            v_smth_ave_jt(tn) = mean(v_smth(tri_ind_start_stop_5exp(tn,1):tri_ind_start_stop_5exp(tn,2)));
            signal_ave_jt(tn) = mean(signal(tri_ind_start_stop_5exp(tn,1):tri_ind_start_stop_5exp(tn,2)));
            L2 = length(tri_ind_start_stop_5exp(tn,1):tri_ind_start_stop_5exp(tn,2));
            u_t(L1 + 1:L1 + L2) = u(tri_ind_start_stop_5exp(tn,1):tri_ind_start_stop_5exp(tn,2));
            L1 = length(u_t);
        end
        u_t_apx = u_t(1:704*8);
        u_t_apx_mean = reshape(u_t_apx,[8, 704]);

        x_smth_c_ave_jt = x_smth_ave_jt(1:length(x_smth_ave_jt)/2);
        x_smth_v_ave_jt = x_smth_ave_jt(length(x_smth_ave_jt)/2+1:length(x_smth_ave_jt));

        x_prf_c = x_prf(1:length(x_smth_ave_jt)/2);
        x_prf_v = x_prf(length(x_smth_ave_jt)/2+1:length(x_smth_ave_jt));

        x_smth_c_ave_jt_taskwise = (reshape(x_smth_c_ave_jt,[22 16]))';
        x_smth_v_ave_jt_taskwise = (reshape(x_smth_v_ave_jt,[22 16]))';

        x_prf_c_ave_taskwise = (reshape(x_prf_c,[22 16]))';
        x_prf_v_ave_taskwise = (reshape(x_prf_v,[22 16]))';

        st1 = 1;
        st3 = 1;

        for bl_nmb = 1:length(c)
            if c(bl_nmb,1) == 1
                x_smth_c_1b_jt(st1,:) = x_smth_c_ave_jt_taskwise(bl_nmb,:);
                x_prf_c_1b(st1,:) = x_prf_c_ave_taskwise(bl_nmb,:);
                st1 = st1 + 1;
            else
                x_smth_c_3b_jt(st3,:) = x_smth_c_ave_jt_taskwise(bl_nmb,:);
                x_prf_c_3b(st3,:) = x_prf_c_ave_taskwise(bl_nmb,:);
                st3 = st3 + 1;
            end
        end

        st1 = 1;
        st3 = 1;
        for bl_nmb = 1:length(v)
            if v(bl_nmb,1) == 1
                x_smth_v_1b_jt(st1,:) = x_smth_v_ave_jt_taskwise(bl_nmb,:);
                x_prf_v_1b(st1,:) = x_prf_v_ave_taskwise(bl_nmb,:);
                st1 = st1 + 1;
            else
                x_smth_v_3b_jt(st3,:) = x_smth_v_ave_jt_taskwise(bl_nmb,:);
                x_prf_v_3b(st3,:) = x_prf_v_ave_taskwise(bl_nmb,:);
                st3 = st3 + 1;
            end
        end


        for tn = 1:trial_number
           u_jt_apx_mean(tn) =  max(u_t_apx_mean(:,tn));
        end

        n = zeros(1, K);

        pt = find(u_jt_apx_mean > 0);
        n(pt) = 1;
        r = u_jt_apx_mean;

        certainty = 1 - normcdf(prctile(x_smth_ave_jt, 50) * ones(1, length(x_smth_ave_jt)), x_smth_ave_jt, sqrt(v_smth_ave_jt));

        u_plot = NaN * ones(1, K);
        u_plot(pt) = u_jt_apx_mean(pt);


        for tb = 1:8
            z_x_smth_c_3b(tb,:) = zscore(x_smth_c_3b_jt(tb,:));
            z_x_smth_c_1b(tb,:) = zscore(x_smth_c_1b_jt(tb,:));
            z_x_smth_v_3b(tb,:) = zscore(x_smth_v_3b_jt(tb,:));
            z_x_smth_v_1b(tb,:) = zscore(x_smth_v_1b_jt(tb,:));
            z_x_prf_c_3b(tb,:) = zscore(x_prf_c_3b(tb,:));
            z_x_prf_c_1b(tb,:) = zscore(x_prf_c_1b(tb,:));
            z_x_prf_v_3b(tb,:) = zscore(x_prf_v_3b(tb,:));
            z_x_prf_v_1b(tb,:) = zscore(x_prf_v_1b(tb,:));
        end

        % vexing 3 back regression analysis
        zx_smth_v_3b = (reshape(z_x_smth_v_3b',1,8*22))';
        zx_prf_v_3b = (reshape(z_x_prf_v_3b',1,8*22))';

        tbl_v_3b = table(zx_smth_v_3b,zx_prf_v_3b,'VariableNames',{'3b_Vxng_arsl','3b_Vxng_prf'});

        lm_v_3b = fitlm(tbl_v_3b,'purequadratic','RobustOpts','on');
        c_coef_v_3b = table2array(lm_v_3b.Coefficients(1,1));
        b_coef_v_3b = table2array(lm_v_3b.Coefficients(2,1));
        a_coef_v_3b = table2array(lm_v_3b.Coefficients(3,1));
        zx_prf_pred_v_3b = lm_v_3b.Fitted;

        %plotSlice(lm_v_3b)


        % vexing 1 back regression analysis
        zx_smth_v_1b = (reshape(z_x_smth_v_1b',1,8*22))';
        zx_prf_v_1b = (reshape(z_x_prf_v_1b',1,8*22))';

        tbl_v_1b = table(zx_smth_v_1b,zx_prf_v_1b,'VariableNames',{'1b_Vxng_arsl','1b_Vxng_prf'});

        lm_v_1b = fitlm(tbl_v_1b,'purequadratic','RobustOpts','on');
        c_coef_v_1b = table2array(lm_v_1b.Coefficients(1,1));
        b_coef_v_1b = table2array(lm_v_1b.Coefficients(2,1));
        a_coef_v_1b = table2array(lm_v_1b.Coefficients(3,1));
        zx_prf_pred_v_1b = lm_v_1b.Fitted;
        %plotSlice(lm_v_1b)



        % calming 3 back regression analysis
        zx_smth_c_3b = (reshape(z_x_smth_c_3b',1,8*22))';
        zx_prf_c_3b = (reshape(z_x_prf_c_3b',1,8*22))';

        tbl_c_3b = table(zx_smth_c_3b,zx_prf_c_3b,'VariableNames',{'3b_Clmng_arsl','3b_Clmng_prf'});

        lm_c_3b = fitlm(tbl_c_3b,'purequadratic','RobustOpts','on');
        c_coef_c_3b = table2array(lm_c_3b.Coefficients(1,1));
        b_coef_c_3b = table2array(lm_c_3b.Coefficients(2,1));
        a_coef_c_3b = table2array(lm_c_3b.Coefficients(3,1));
        zx_prf_pred_c_3b = lm_c_3b.Fitted;
        %plotSlice(lm_c_3b)


        % calming 1 back regression analysis
        zx_smth_c_1b = (reshape(z_x_smth_c_1b',1,8*22))';
        zx_prf_c_1b = (reshape(z_x_prf_c_1b',1,8*22))';

        tbl_c_1b = table(zx_smth_c_1b,zx_prf_c_1b,'VariableNames',{'1b_Clmng_arsl','1b_Clmng_prf'});

        lm_c_1b = fitlm(tbl_c_1b,'purequadratic','RobustOpts','on');
        c_coef_c_1b = table2array(lm_c_1b.Coefficients(1,1));
        b_coef_c_1b = table2array(lm_c_1b.Coefficients(2,1));
        a_coef_c_1b = table2array(lm_c_1b.Coefficients(3,1));
        zx_prf_pred_c_1b = lm_c_1b.Fitted;
        %plotSlice(lm_c_1b)


        zx_smth = (zscore(x_smth_ave_jt))';
        zx_prf = (zscore(x_prf))';

        tbl_full = table(zx_smth,zx_prf,'VariableNames',{'arsl','prf'});

        lm_full = fitlm(tbl_full,'purequadratic','RobustOpts','on');
        %plotSlice(lm_full)

        % calming section regression analysis

        s = 0;
        c1 = 1;
        c3 = 1;
        for l = 1:length(c)
            if c(l,1) == 1
                zx_smth_c(s+1:22*l) = z_x_smth_c_1b(c1,:);
                zx_prf_c(s+1:22*l) = z_x_prf_c_1b(c1,:);
                s = length(zx_smth_c);
                c1 = c1 + 1;
            else
                zx_smth_c(s+1:22*l) = z_x_smth_c_3b(c3,:);
                zx_prf_c(s+1:22*l) = z_x_prf_c_3b(c3,:);
                s = length(zx_smth_c);
                c3 = c3 + 1;
            end
        end

        tbl_c = table(zx_smth_c',zx_prf_c','VariableNames',{'Clmng-arsl','Clmng-prf'});

        lm_c = fitlm(tbl_c,'purequadratic','RobustOpts','on');
        c_coef_c = table2array(lm_c.Coefficients(1,1));
        b_coef_c = table2array(lm_c.Coefficients(2,1));
        a_coef_c = table2array(lm_c.Coefficients(3,1));
        zx_prf_pred_c = lm_c.Fitted;
        %plotSlice(lm_c)




        s = 0;
        v1 = 1;
        v3 = 1;
        for l = 1:length(v)
            if c(l,1) == 1
                zx_smth_v(s+1:22*l) = z_x_smth_v_1b(v1,:);
                zx_prf_v(s+1:22*l) = z_x_prf_v_1b(v1,:);
                s = length(zx_smth_v);
                v1 = v1 + 1;
            else
                zx_smth_v(s+1:22*l) = z_x_smth_v_3b(v3,:);
                zx_prf_v(s+1:22*l) = z_x_prf_v_3b(v3,:);
                s = length(zx_smth_v);
                v3 = v3 + 1;
            end
        end

        tbl_v = table(zx_smth_v',zx_prf_v','VariableNames',{'Vxng-arsl','Vxng-prf'});

        lm_v = fitlm(tbl_v,'purequadratic','RobustOpts','on');
        c_coef_v = table2array(lm_v.Coefficients(1,1));
        b_coef_v = table2array(lm_v.Coefficients(2,1));
        a_coef_v = table2array(lm_v.Coefficients(3,1));
        zx_prf_pred_v = lm_v.Fitted;
        %plotSlice(lm_v)

        % only 1 backs regression analysis

        zx_smth_1b = [zx_smth_c_1b;zx_smth_v_1b];
        zx_prf_1b = [zx_prf_c_1b;zx_prf_v_1b];

        tbl_1b = table(zx_smth_1b,zx_prf_1b,'VariableNames',{'1b_arsl','1b_prf'});

        lm_1b = fitlm(tbl_1b,'purequadratic','RobustOpts','on');
        c_coef_1b = table2array(lm_1b.Coefficients(1,1));
        b_coef_1b = table2array(lm_1b.Coefficients(2,1));
        a_coef_1b = table2array(lm_1b.Coefficients(3,1));
        zx_prf_pred_1b = lm_1b.Fitted;
        %plotSlice(lm_1b)

        % only 3 backs regression analysis

        zx_smth_3b = [zx_smth_c_3b;zx_smth_v_3b];
        zx_prf_3b = [zx_prf_c_3b;zx_prf_v_3b];

        tbl_3b = table(zx_smth_3b,zx_prf_3b,'VariableNames',{'3b_arsl','3b_prf'});

        lm_3b = fitlm(tbl_3b,'purequadratic','RobustOpts','on');
        c_coef_3b = table2array(lm_3b.Coefficients(1,1));
        b_coef_3b = table2array(lm_3b.Coefficients(2,1));
        a_coef_3b = table2array(lm_3b.Coefficients(3,1));
        zx_prf_pred_3b = lm_3b.Fitted;
        %plotSlice(lm_3b)

        % full data regression analysis
        zx_smth_full = [zx_smth_c';zx_smth_v'];
        zx_prf_full = [zx_prf_c';zx_prf_v'];
        tbl_full = table(zx_smth_full,zx_prf_full,'VariableNames',{'Full-arsl','Full-prf'});

        lm_full = fitlm(tbl_full,'purequadratic','RobustOpts','on')
        c_coef_full = table2array(lm_full.Coefficients(1,1));
        b_coef_full = table2array(lm_full.Coefficients(2,1));
        a_coef_full = table2array(lm_full.Coefficients(3,1));
        zx_prf_pred_full = lm_full.Fitted;
        plotSlice(lm_full)

        ci = coefCI(lm_full);
        eps = zx_prf_full-zx_prf_pred_full;



        %chi2 = chi_squared(zx_prf_full,zx_prf_pred_full,3,eps)

        % data c1 (calmjing 1-back task)
        IUL_c1(subj).a_coef = a_coef_c_1b;
        IUL_c1(subj).b_coef = b_coef_c_1b;
        IUL_c1(subj).c_coef = c_coef_c_1b;
        IUL_c1(subj).zx_smth = zx_smth_c_1b;
        IUL_c1(subj).zx_prf = zx_prf_c_1b;
        IUL_c1(subj).zx_prf_pred = zx_prf_pred_c_1b;


        % data c3 (calming 3-back task)
        IUL_c3(subj).a_coef = a_coef_c_3b;
        IUL_c3(subj).b_coef = b_coef_c_3b;
        IUL_c3(subj).c_coef = c_coef_c_3b;
        IUL_c3(subj).zx_smth = zx_smth_c_3b;
        IUL_c3(subj).zx_prf = zx_prf_c_3b;
        IUL_c3(subj).zx_prf_pred = zx_prf_pred_c_3b;


        % data v1 (vexing 1-back task)
        IUL_v1(subj).a_coef = a_coef_v_1b;
        IUL_v1(subj).b_coef = b_coef_v_1b;
        IUL_v1(subj).c_coef = c_coef_v_1b;
        IUL_v1(subj).zx_smth = zx_smth_v_1b;
        IUL_v1(subj).zx_prf = zx_prf_v_1b;
        IUL_v1(subj).zx_prf_pred = zx_prf_pred_v_1b;


        % data v3 (vexing 3-back task)
        IUL_v3(subj).a_coef = a_coef_v_3b;
        IUL_v3(subj).b_coef = b_coef_v_3b;
        IUL_v3(subj).c_coef = c_coef_v_3b;
        IUL_v3(subj).zx_smth = zx_smth_v_3b;
        IUL_v3(subj).zx_prf = zx_prf_v_3b;
        IUL_v3(subj).zx_prf_pred = zx_prf_pred_v_3b;

        % data calming
        IUL_c(subj).a_coef = a_coef_c;
        IUL_c(subj).b_coef = b_coef_c;
        IUL_c(subj).c_coef = c_coef_c;
        IUL_c(subj).zx_smth = zx_smth_c;
        IUL_c(subj).zx_prf = zx_prf_c;
        IUL_c(subj).zx_prf_pred = zx_prf_pred_c;

        % data vexing
        IUL_v(subj).a_coef = a_coef_v;
        IUL_v(subj).b_coef = b_coef_v;
        IUL_v(subj).c_coef = c_coef_v;
        IUL_v(subj).zx_smth = zx_smth_v;
        IUL_v(subj).zx_prf = zx_prf_v;
        IUL_v(subj).zx_prf_pred = zx_prf_pred_v;


        % data 1 back
        IUL_1b(subj).a_coef = a_coef_1b;
        IUL_1b(subj).b_coef = b_coef_1b;
        IUL_1b(subj).c_coef = c_coef_1b;
        IUL_1b(subj).zx_smth = zx_smth_1b;
        IUL_1b(subj).zx_prf = zx_prf_1b;
        IUL_1b(subj).zx_prf_pred = zx_prf_pred_1b;


        % data 3 back
        IUL_3b(subj).a_coef = a_coef_3b;
        IUL_3b(subj).b_coef = b_coef_3b;
        IUL_3b(subj).c_coef = c_coef_3b;
        IUL_3b(subj).zx_smth = zx_smth_3b;
        IUL_3b(subj).zx_prf = zx_prf_3b;
        IUL_3b(subj).zx_prf_pred = zx_prf_pred_3b;

        % data full experiment
        IUL_full(subj).a_coef = a_coef_full;
        IUL_full(subj).b_coef = b_coef_full;
        IUL_full(subj).c_coef = c_coef_full;
        IUL_full(subj).zx_smth = zx_smth_full;
        IUL_full(subj).zx_prf = zx_prf_full;
        IUL_full(subj).zx_prf_pred = zx_prf_pred_full;

        save('IUL_data.mat','IUL_c1', 'IUL_c3', 'IUL_v1' ,'IUL_v3', 'IUL_c', 'IUL_v','IUL_1b', 'IUL_3b', 'IUL_full')

      
        IUL_DATA(subj).C = IUL_c;
        IUL_DATA(subj).V = IUL_v;

        IUL_DATA(subj).C1 = IUL_c1;
        IUL_DATA(subj).C3 = IUL_c3;

        IUL_DATA(subj).V1 = IUL_v1;
        IUL_DATA(subj).V3 = IUL_v3;

        IUL_DATA(subj).b1 = IUL_1b;
        IUL_DATA(subj).b3 = IUL_3b;

        IUL_DATA(subj).Full = IUL_full;

end

% Main Figure

figure('position', [0 0 315 315]);
for l = 1:length(subj)
        
    x(:,l) = IUL_DATA(subj(l)).Full(subj(l)).zx_smth;
    z(:,l) = IUL_DATA(subj(l)).Full(subj(l)).zx_prf_pred;
    
    z_prf_act(:,l) = IUL_DATA(subj(l)).Full(subj(l)).zx_prf;
 
    [sortedX(:,l), sortIndex(:,l)] = sort(x(:,l));
    sortedX_prf(:,l) = z(sortIndex(:,l),l);
    sortedX_prf_act(:,l) = z_prf_act(sortIndex(:,l),l);

 
    %subplot(2,length(subj)/2,l)
    subplot(1,1,1)

    plot(sortedX(:,l),sortedX_prf(:,l),'b', 'linewidth', 1)
    hold on
    scatter(IUL_DATA(subj(l)).Full(subj(l)).zx_smth,IUL_DATA(subj(l)).Full(subj(l)).zx_prf,'r','MarkerEdgeAlpha',.49)
  
    title(['Participant ' num2str(l)],'FontSize', 12);
    
    %plot(sortedX(:,l),sortedX_prf(:,l),'b', 'linewidth', 1)
    ylim([-3 3]);
    
    xlim([-3 3])
    
    xlabel({'arousal'},'FontSize', 9);
    ylabel('performance','FontSize', 9);
    %legend('predicted','actuall')
    
    
    set(gca,'LineWidth',1);
 
end
