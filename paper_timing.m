% This code is only for processing experimental data published in:

% [Khazaei, Saman, Md Rafiul Amin, and Rose T. Faghih. "Decoding a Neurofeedback-Modulated Cognitive Arousal 
% State to Investigate Performance Regulation by the Yerkes-Dodson Law." 
% 2021 43rd Annual International Conference of the IEEE Engineering in
% Medicine & Biology Society (EMBC). IEEE, 2021.]

% And 

%[Parshi, Srinidhi. A functional-near infrared spectroscopy investigation of mental workload. Diss. 2020.]




clear;
close all;
clc

for subj = [3,4,6,7,8,11] % subject number
    
        load(['Timing_Information',num2str(subj),'.mat']);

        % experiment time index 

        % calming 

        bstart_1_c = Timing_Information.trial_block_start_times_1_back_calming';
        bend_1_c = Timing_Information.trial_block_end_times_1_back_calming';

        bstart_3_c = Timing_Information.trial_block_start_times_3_back_calming';
        bend_3_c = Timing_Information.trial_block_end_times_3_back_calming';


        rstarts = Timing_Information.trial_start_times_resting';
        rends = Timing_Information.trial_end_times_resting;

        start_c = sort([bstart_1_c;bstart_3_c]);
        end_c = sort([bend_1_c;bend_3_c]);

        c = zeros(length(start_c), 3);

        calming = cell(length(start_c), 3);

        for idx = 1:length(start_c)
            calming(idx, 2) = {[start_c(idx), end_c(idx)]};
            if (ismember(start_c(idx), bstart_1_c)) == 1
                c(idx, 1) = 1;   % as 1 back task
                calming(idx, 1) = {'1 back'};
            else
                c(idx, 1) = 3;   % as 3 back task
                calming(idx, 1) = {'3 back'};
            end
        end


        c = [c(:, 1), start_c, end_c];

        % trials

        tstarts_1_c = Timing_Information.trial_start_times_1_back_calming';
        tends_1_c = Timing_Information.trial_end_times_1_back_calming';

        tstarts_3_c = Timing_Information.trial_start_times_3_back_calming';
        tends_3_c = Timing_Information.trial_end_times_3_back_calming';



        trials_starts_c = sort([tstarts_1_c;tstarts_3_c]);
        trials_ends_c = sort([tends_1_c;tends_3_c]);


        for idx = 1:length(start_c)
            o = 1;
            for t = 1:length(trials_starts_c)
                if (trials_starts_c(t) <= c(idx, end)) && (trials_starts_c(t) >= c(idx, end - 1))
                    tri_starts_c_order(idx, o) = trials_starts_c(t);
                    tri_ends_c_order(idx, o) = trials_ends_c(t);
                    o = o + 1;
                end
            end
        end
        rends_c =  rends(1:length(rends)/2);         
        rends_c = reshape(rends_c',22,16);
        rends_c = rends_c';
        for idx = 1:length(start_c)
            rends_c(idx,22) = rends_c(idx,21) + 2;
            calming(idx, 3) = {[tri_starts_c_order(idx, :)', rends_c(idx, :)']};   % start and end of each trials

        end

        % vexing

        bstart_1_v = Timing_Information.trial_block_start_times_1_back_vexing';
        bend_1_v = Timing_Information.trial_block_end_times_1_back_vexing';

        bstart_3_v = Timing_Information.trial_block_start_times_3_back_vexing';
        bend_3_v = Timing_Information.trial_block_end_times_3_back_vexing';


        start_v = sort([bstart_1_v;bstart_3_v]);
        end_v = sort([bend_1_v;bend_3_v]);
        v = zeros(length(start_v), 3);

        vexing = cell(length(start_v), 3);

        for idx = 1:length(start_v)
            vexing(idx, 2) = {[start_v(idx), end_v(idx)]};
            if (ismember(start_v(idx), bstart_1_v)) == 1
                v(idx, 1) = 1;      % as 1 back task
                vexing(idx, 1) = {'1 back'};
            else
                v(idx, 1) = 3;      % as 3 back task
                vexing(idx, 1) = {'3 back'};
            end
        end

        end_v = sort([bend_1_v;bend_3_v]);
        v = [v(:, 1), start_v, end_v];

        % trialas

        tstarts_1_v = Timing_Information.trial_start_times_1_back_vexing';
        tends_1_v = Timing_Information.trial_end_times_1_back_vexing';

        tstarts_3_v = Timing_Information.trial_start_times_3_back_vexing';
        tends_3_v = Timing_Information.trial_end_times_3_back_vexing';

        trials_starts_v = sort([tstarts_1_v;tstarts_3_v]);
        trials_ends_v = sort([tends_1_v;tends_3_v]);


        for idx = 1:length(start_v)
            o = 1;
            for t = 1:length(trials_starts_v)
                if (trials_starts_v(t) <= v(idx, end)) && (trials_starts_v(t) >= v(idx, end - 1))
                    tri_starts_v_order(idx, o) = trials_starts_v(t);
                    tri_ends_v_order(idx, o) = trials_ends_v(t);
                    o = o + 1;
                end
            end
        end

        rends_v = rends(length(rends)/2 + 1:length(rends));         
        rends_v = reshape(rends_v',22,16);
        rends_v = rends_v';
        for idx = 1:length(start_v)
            rends_v(idx,22) = rends_v(idx,21) + 2;
            vexing(idx, 3) = {[tri_starts_v_order(idx, :)', rends_v(idx, :)']};   % start and end of each trials

        end
        first_stimuli_start = min(min(cell2mat(calming(1,3))));
        last_stimuli_end = max(max(cell2mat(vexing(end,end))));

        first_calming = first_stimuli_start;
        last_calming = max(max(cell2mat(calming(end,end))));

        first_vexing = min(min(cell2mat(vexing(1,3))));
        last_vexing = max(max(cell2mat(vexing(end,end))));

        clear idx;

        %%

        % Signal deconvolution has been done by MD RAFIUL AMIN.
        % Based on his published work:
        % [Amin, Md Rafiul, and Rose T. Faghih. "Sparse deconvolution of electrodermal activity via 
        % continuous-time system identification." IEEE Transactions on Biomedical Engineering 66.9 (2019): 2585-2595.]

        load('EDA_deconvolution.mat');


        u = (DECON_EDA_RES(subj).u)';

        signal = (DECON_EDA_RES(subj).y);

        tu = (DECON_EDA_RES(subj).tu)';

        tonic = (DECON_EDA_RES(subj).tonic)';
        phasic = (DECON_EDA_RES(subj).phasic)';

        % finding the pick index for tu based on the exp time and same frequency
        % aproximation ( might have 0.02 sec loss ) which is negligible;

        trials_starts = sort(reshape([tri_starts_c_order;tri_starts_v_order],1,32*22));
        trials_ends = sort(reshape([rends_c;rends_v],1,32*22));

        for j = 1:length(trials_starts)

            ind_trials_start(j) = find ((tu >= trials_starts(j)) & (tu < trials_starts(j) + 0.25));
            ind_trials_end(j) = find ((tu > trials_ends(j) - 0.25) & (tu <= trials_ends(j)));
            stop_calming_idx = find ((tu > rends_c(end) - 0.25) & (tu <= rends_c(end)));
            start_vexing_idx = find ((tu >= tri_starts_v_order(1,1)) & (tu < tri_starts_v_order(1, 1) + 0.25));
        end

        start_calming_idx = ind_trials_start(1); 
        stop_vexing_idx = ind_trials_end(end); 

        tri_ind_start_stop = [ind_trials_start', ind_trials_end'];

        dif = tri_ind_start_stop(:,2) - tri_ind_start_stop(:,1);

        % main scope consideration plus 5 min before the start
        % 5 min + trials                                                    call tu_5exp
        % 5 min idx with f = 4 hz --> 5 * 60 = 300 s, 300s*4hz = 1200;

        tu_5exp = tu(start_calming_idx-1200:stop_vexing_idx);
        u_5exp = u(start_calming_idx-1200:stop_vexing_idx);
        signal_5exp = signal(start_calming_idx-1200:stop_vexing_idx);

        phasic_5exp = phasic(start_calming_idx-1200:stop_vexing_idx);

        tonic_5exp = tonic(start_calming_idx-1200:stop_vexing_idx);



        tri_ind_start_stop_5exp = tri_ind_start_stop - length(tu(1:start_calming_idx - 1201)) * ones(length(tri_ind_start_stop),2);


        % calming & vexing consideration
        % tu_calming = tu(start_calming_idx:stop_calming_idx);
        % tu_vexing = tu(start_vexing_idx:stop_vexing_idx);

        % n back tasks

        calming_trials_nbacks = zeros(1, 3);
        for i = 1:length(c)
            calming_trials_nbacks(i,:) = [c(i), min(min(cell2mat(calming(i,3)))), max(max(cell2mat(calming(i,3))))];
        end


        vexing_trials_nbacks = zeros(1, 3);
        for i = 1:length(v)
            vexing_trials_nbacks(i,:) = [v(i), min(min(cell2mat(vexing(i,3)))), max(max(cell2mat(vexing(i,3))))];
        end


        pt = find(u > 0);
        u_plot = NaN * ones(1, length(u));
        u_plot(pt) = u(pt);



%       plots:
%{
        figure('position', [0 0 515 660]);

        subplot(211)
        plot(tu_5exp, signal_5exp,'r','linewidth', 2.25);grid;
        hold on; 
        plot(tu, signal,'b','linewidth', 0.8)
        xlabel({'time (s)'}); 
        ylabel({' skin cond.', '(\mu S)'});
        title(['Participant ' num2str(subj)]);
        ylim;
        yl = ylim;
        patch([first_calming, last_calming, last_calming, first_calming], [yl(1) yl(1) yl(2) yl(2)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        patch([first_vexing, last_vexing, last_vexing, first_vexing],[yl(1) yl(1) yl(2) yl(2)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        legend('main scope','whole experiment');

        subplot(212)
        stem(tu_5exp, u_plot(start_calming_idx - 1200:stop_vexing_idx), 'filled', 'r', 'markersize', 5);  % important (apprx)
        grid;
        hold on;
        stem(tu, u_plot, 'filled', 'b', 'markersize', 2.2);
        ylim;
        yl = ylim;
        patch([first_calming, last_calming, last_calming, first_calming], [yl(1) yl(1) yl(2) yl(2)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        patch([first_vexing, last_vexing, last_vexing,first_vexing],[yl(1) yl(1) yl(2) yl(2)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        xlabel({'time (s)'}); 
        ylabel({'Amplitude'});
        legend('main scope','whole experiment');
%}

        pre_MPP(subj).calming = calming;
        pre_MPP(subj).vexing = vexing;
        pre_MPP(subj).c = c;
        pre_MPP(subj).v = v;

        pre_MPP(subj).u_5exp = u_5exp;
        pre_MPP(subj).signal_5exp = signal_5exp;
        pre_MPP(subj).tu_5exp = tu_5exp;

        pre_MPP(subj).calming_period = [first_calming, last_calming];
        pre_MPP(subj).vexing_period = [first_vexing, last_vexing];

        pre_MPP(subj).tri_ind_start_stop = tri_ind_start_stop;
        pre_MPP(subj).tri_ind_start_stop_5exp = tri_ind_start_stop_5exp;


        pre_MPP(subj).deconvolution_res = DECON_EDA_RES(subj);
        pre_MPP(subj).Timing_information_source_file = Timing_Information;

        pre_MPP(subj).calming_trials_nbacks_period = calming_trials_nbacks;
        pre_MPP(subj).vexing_trials_nbacks_period = vexing_trials_nbacks;
        
        pre_MPP(subj).phasic_tonic_exp.phasic_5_exp = phasic_5exp;
        pre_MPP(subj).phasic_tonic_exp.tonic_5_exp = tonic_5exp;
        
end

save('pre_MPP.mat','pre_MPP')
