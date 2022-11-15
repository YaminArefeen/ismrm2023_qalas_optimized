function mask = generate_sampling_mask(sampling_flag,tf,nA,dims,accel_Ry,accel_Rz,repeats)

if(nargin < 7)
    repeats = 1;
end

M = dims(1);
N = dims(2);
E = dims(3);

mask = zeros(M,N,1,E);

switch(sampling_flag)
    %-generate random under-sampling mask
    case 1
        mask_notemp = zeros(M,N);
        mask_notemp(1:accel_Ry:end,1:accel_Rz:end) = 1;
        
        total_samples_per_acq   = sum(vec(mask_notemp));
        samples_per_timepoint   = floor(total_samples_per_acq / tf); 
    
        %-generating random under-sampling mask for each of the 5 QALAS acquisitions
        for rr = 1:repeats
            for aa = 1:nA
                nonzero_indices         = find(vec(mask_notemp));
                nonzero_indices_shuffle = nonzero_indices(randperm(length(nonzero_indices)));

                for ee = 1:tf
                    cur_mask = zeros(M*N,1);

                    cur_mask(nonzero_indices_shuffle((ee-1)*samples_per_timepoint + 1:ee*samples_per_timepoint)) = 1;
                    cur_mask = reshape(cur_mask,M,N);

                    mask(:,:,:,ee + 128*(aa-1)) = cur_mask + mask(:,:,:,ee + 128*(aa-1));
                    %enables accumulation of mask for repeats > 1
                end
            end
        end
end

mask(mask > 0) = 1;

end