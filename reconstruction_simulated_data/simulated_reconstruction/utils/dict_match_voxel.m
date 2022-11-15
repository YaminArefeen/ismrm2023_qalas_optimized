function [t2est_us,t2est_fs,t1est_us,t1est_fs,dsig_us,dsig_fs] = dict_match_voxel(dictionary_old,t1t2_lut_prune,...
    x,y,rec_old_fs,rec_old_us,E)
%-normalize dictionary + get parameters to compare
dict_old = abs(dictionary_old) ./ sqrt(sum(dictionary_old.^2,1));
parameters_dict = t1t2_lut_prune;

%-get signals to compare
S = length(x);
voxel_signals_fs  = zeros(E,S);
voxel_signals_us  = zeros(E,S);

for ss = 1:S
    voxel_signals_fs(:,ss)   = squeeze(rec_old_fs(x(ss),y(ss),:));
    voxel_signals_us(:,ss)   = squeeze(rec_old_us(x(ss),y(ss),:));
end

t1est_fs = zeros(S,1);
t2est_fs = zeros(S,1);
dsig_fs  = zeros(E,S);

t1est_us = zeros(S,1);
t2est_us = zeros(S,1);
dsig_us  = zeros(E,S);

%t1 / t2 mapping
for ss = 1:S
    %-mapping fs
    signal     = voxel_signals_fs(:,ss) ./ sqrt(sum(voxel_signals_fs(:,ss).^2,1));
    difference = sqrt(sum((abs(dict_old) - abs(signal)).^2,1))./...
        sqrt(sum(abs(dict_old).^2,1));

    [~,minidcs] = min(difference,[],2);

    minidcs = squeeze(minidcs);

    t2est_fs(ss)  = parameters_dict(minidcs,2);
    t1est_fs(ss)  = parameters_dict(minidcs,1);
    dsig_fs(:,ss) = abs(dictionary_old(:,minidcs).');

    %-mapping us
    signal     = voxel_signals_us(:,ss) ./ sqrt(sum(voxel_signals_us(:,ss).^2,1));
    difference = sqrt(sum((abs(dict_old) - abs(signal)).^2,1))./...
        sqrt(sum(abs(dict_old).^2,1));

    [~,minidcs] = min(difference,[],2);

    minidcs = squeeze(minidcs);

    t2est_us(ss)  = parameters_dict(minidcs,2);
    t1est_us(ss)  = parameters_dict(minidcs,1);
    dsig_us(:,ss) = abs(dictionary_old(:,minidcs).');
end
