function D = create_blur_matrix(mn, blur)
    mindex = 1:mn;
    nindex = 1:mn;
    for i = 1:blur
%                              vecini                           
%                            stg     dr
        mindex = [mindex, i+1:mn, 1:mn-i]; % pentru liniile din D
        nindex = [nindex, 1:mn-i, i+1:mn]; % pentru coloanele din D
    end
    D = sparse(mindex, nindex, 1 / (2*blur + 1), mn, mn); % media vec. stg + element curent + 
                                                            % + vec. dr
end

% blur - cati vecini la stanga si la dreapta consider pentru mediere