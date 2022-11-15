function t1t2_lut_prune = generate_t1t2_lut(t1_entries,t2_entries)

T1_entries = repmat(t1_entries.', [1,length(t2_entries)]).';
T1_entries = T1_entries(:);

T2_entries = repmat(t2_entries.', [1,length(t1_entries)]);
T2_entries = T2_entries(:);


t1t2_lut = cat(2, T1_entries, T2_entries);

% remove cases where T2>T1
idx = 0;
%for t = 1:length(t1t2_lut)
for t = 1:size(t1t2_lut,1)
    if t1t2_lut(t,1) < t1t2_lut(t,2)
        idx = idx+1;
    end
end

% t1t2_lut_prune = zeross( [length(t1t2_lut) - idx, 2] );
t1t2_lut_prune = zeross( [size(t1t2_lut,1) - idx, 2] );

idx = 0;
% for t = 1:length(t1t2_lut)
for t = 1:size(t1t2_lut,1)
    if t1t2_lut(t,1) >= t1t2_lut(t,2)
        idx = idx+1;
        t1t2_lut_prune(idx,:) = t1t2_lut(t,:);
    end
end

disp(['dictionary entries: ', num2str(size(t1t2_lut_prune,1))])


end