addpath simulation_functions/ utils/

%% loading sequences and setting sequence simulation parameters
load data/sequence_flip_angles.mat
NS = length(sequences);

%-sequence parameters
TR         = 4500e-3;   % excluding dead time at the end of the sequence 
alpha_deg  = 4;
num_reps   = 5;       % number of repetitions to simulate to reach steady state
echo2use   = 1;
esp        = 5.74 * 1e-3;     % echo spacing in sec
turbo_fact = 128;  % ETL

gap_between_readouts  = 900e-3;
time2relax_at_the_end = 0;  % actual TR on the console = TR + time2relax_at_the_end
 
CC = 18; 

%-dictionary simulation parameters
t1_entries = [300:10:2000, 2020:20:2800];
t2_entries = [40:1:80, 82:2:100, 104:2:350];
b1_val     = 1;
inv_eff    = 1;

%-reconstruction parameters
K = 3;

noise_levels = 4e-4;
accels       = [2,2];
lambdas      = .00001;
types        = 'wavelets';
iters        = 800;

sampling_flag = 1; %kind of sampling pattern (only have one for now)
gpu_flag      = 1; %perform dictionary matching on GPU?
repeats       = 1; %for now, assume repeats = 1
%% setting up dictionary entries 
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

%% simulating dictionaries

dictionaries_norm = {};
dictionaries      = {};

for ns = 1:NS
    if(length(sequences{ns}.alpha) <= 5)
        alpha_train = repelem(sequences{ns}.alpha,turbo_fact);
    else
        alpha_train = sequences{ns}.alpha;
    end
    
    num_acq = sequences{ns}.acqs;
    
    [Mz_all,Mxy_all] = sim_qalas_allalpha_acqs(TR, alpha_train, esp, turbo_fact, t1t2_lut_prune(:,1)*1e-3,...
    t1t2_lut_prune(:,2)*1e-3, num_reps, num_acq, gap_between_readouts, time2relax_at_the_end,...
    b1_val,inv_eff);

    dictionary = squeeze(Mxy_all(:,:,end));
    
    dictionary_norm = abs(dictionary ./ sqrt(sum(abs(dictionary).^2,1)));
    
    dictionaries_norm{ns} = dictionary_norm;
    dictionaries{ns}      = dictionary;
end

%% generating quantitative phantom
load('data/phantom.mat')
t2map = phantom.t2map;
t1map = phantom.t1map;
pdmap = phantom.pdmap;

[M,N] = size(t2map);

%% simulating signal

signals = {};

for ns = 1:NS
    if(length(sequences{ns}.alpha) <= 5)
        alpha_train = repelem(sequences{ns}.alpha,turbo_fact);
    else
        alpha_train = sequences{ns}.alpha;
    end
    
    num_acq = sequences{ns}.acqs;
    
    [Mz,Mxy] = sim_qalas_allalpha_acqs(TR, alpha_train, esp, turbo_fact, t1map(:),...
    t2map(:), num_reps, num_acq, gap_between_readouts, time2relax_at_the_end, b1_val,inv_eff);

    E = length(alpha_train);
    
    signal = reshape(Mxy(:,:,end).',M,N,E);
    signal(isnan(signal)) = 0;
    
    signals{ns} = signal .* pdmap;
end

%% loading coils
coils = readcfl('data/coils');

C = size(coils,3);
coils_rsh = reshape(coils,M*N,C).';

[U,S,V] = svd(coils_rsh,'econ');

coils = reshape((U(:,1:CC)'*coils_rsh).',M,N,CC);

%% reconstructions  
[M,N,C] = size(coils);

recs = {};

for ns = 1:NS
	coils_bart = reshape(coils,M,N,1,C);
    nA = sequences{ns}.acqs; 
    
    %-generating k-space
    fprintf('generating k-space and noise... ') 
    
    signal = signals{ns}; 
    E = size(signal,3);
    
    noise  = noise_levels*(randn(M,N,C,E) + 1i*randn(M,N,C,E));
    kspace = mfft2(coils.*reshape(signal,M,N,1,E));

    mask = generate_sampling_mask(sampling_flag,turbo_fact,nA,[M,N,E],accels(1),accels(2),repeats);

    ksp = reshape((kspace+noise).*mask,M,N,1,C,1,E);
    
    fprintf('done\n')
    
    %-generating subspace
    [u,s,v] = svd(dictionaries{ns},'econ'); 
    phi     = reshape(u(:,1:K,:),1,1,1,1,1,E,K);
    
    writecfl('data/phi',phi)
    
    if(strcmp(types, 'llr'))
        bartstr = sprintf('-R L:3:0:%f -i %d',lambdas,iters);
    else
        bartstr = sprintf('-l1 -r %f -i %d',lambdas,iters);
    end
    
    coeff =  bart(sprintf('pics -g -S -w 1 -B data/phi %s',bartstr), ksp,coils_bart);
    recs{ns} = reshape((squeeze(phi)*reshape(squeeze(coeff),M*N,K).').',M,N,E);
end

%% creating parallel processing for dictionary matching
delete(gcp('nocreate'))
c = parcluster('local');    % build the 'local' cluster object

total_cores = c.NumWorkers;
parpool(total_cores)

%% dictionary matching loop
t2estimates = zeros(M,N,NS); 
t1estimates = zeros(M,N,NS);

p_rsh   = reshape(permute(pdmap,[3,1,2]),1,1,M*N);

for ns = 1:NS 
    E = size(dictionaries_norm{ns},1);
    
    rec_rsh = reshape(permute(squeeze(recs{ns}),[3,1,2,4]),E,1,M*N);
    
    %-transferring to gpu
    if(gpu_flag)
        cur_dict = gpuArray(dictionaries_norm{ns});
        cur_rsh  = gpuArray(rec_rsh);
    end
    
    nvox = size(rec_rsh,3);
    
    t2est = zeros(nvox,1);
    t1est = zeros(nvox,1);
    
    fprintf('matching...\n')
    tic

    parfor ss = 1:nvox
        if(p_rsh(1,1,ss) == 0)
            continue
        end

        time_to_compare = cur_rsh(:,:,ss) ./ sqrt(sum(cur_rsh(:,:,ss).^2,1));
        differences     = sum((cur_dict - abs(time_to_compare)).^2,1);
        [~,minidcs]     = min(differences,[],2);

        minidcs = squeeze(minidcs);

        t2est(ss,1)  = t1t2_lut_prune(minidcs,2);
        t1est(ss,1)  = t1t2_lut_prune(minidcs,1);
    end
    toc
    
    
    t2estimates(:,:,ns) = reshape(t2est,M,N);
    t1estimates(:,:,ns) = reshape(t1est,M,N);
end
delete(gcp('nocreate'))

%% computing errors
cro = 10;
cpe = 7;
ro  = 1+cro:M-cro; R = length(ro);
pe  = 1+cpe:N-cpe; P = length(pe);

n  = @(x) flipud(x(ro,pe));
er = @(tr,x) abs(tr-x);

errorst2 = zeros(R,P,NS);
errorst1 = zeros(R,P,NS);
t2disp = zeros(R,P,NS);
t1disp = zeros(R,P,NS);

mask = zeros(size(t2map));
mask(t2map > 0) = 1;

for ns = 1:NS
    errorst2(:,:,ns) = n(er(t2map*1000,t2estimates(:,:,ns)));
    errorst1(:,:,ns) = n(er(t1map*1000,t1estimates(:,:,ns)));
    
    t2disp(:,:,ns) = n(t2estimates(:,:,ns));
    t1disp(:,:,ns) = n(t1estimates(:,:,ns));
end

figure; imshow3(t2disp)
figure; imshow3(errorst2,[0,80])

figure; imshow3(t1disp)
figure; imshow3(errorst1,[0,2000])
