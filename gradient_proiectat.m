function [x, iter_vec, diff_vec] = gradient_proiectat(D, y, max_iter, c, epsilon)

    mn = length(y);
    x = zeros(mn, 1);  % punctul initial
    iter = 0;
    diff = Inf;

    % vectorii pentru plot ulterior
    iter_vec = [];
    diff_vec = [];

    while iter < max_iter && diff > epsilon
        x_prev = x;
        grad = 2 * D' * (D * x - y);
        alpha = c / (iter + 1);
        x = x - alpha * grad;
        x = min(max(x, 0), 1); % proiectie pe [0,1]

        % Diferenta intre iteratii (norma euclidiana)
        diff = norm(x - x_prev);
        
        iter = iter + 1;
        iter_vec(end+1) = iter;
        diff_vec(end+1) = diff;
    end

end