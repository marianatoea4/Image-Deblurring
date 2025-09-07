function [x, iter_vec, diff_vec] = gradient_conditional(D, y, max_iter, epsilon)
   
    mn = length(y);
    x = zeros(mn, 1); % punctul initial
    iter = 0;
    diff = Inf;

    % vectorii pentru plot ulterior
    iter_vec = [];
    diff_vec = [];

    while iter < max_iter && diff > epsilon
        grad = 2 * D' * (D * x - y);

        % Minimizez prod. scalar grad'*s sub constrangerea s apartine lui
        % [0,1]^n
        % s(i) = 0 daca grad(i) > 0, s(i) = 1 daca grad(i) < 0
        s = double(grad < 0);

        % Pas adaptiv: gamma = 2 / (k + 2)
        gamma = 2 / (iter + 2); % iter + 1 + 1, incep de la 0

        x_new = x + gamma * (s-x);
        
        % Calcul criteriu de oprire
        diff = norm(x_new - x);

        iter = iter + 1;
        iter_vec(end+1) = iter;
        diff_vec(end+1) = diff;

        x = x_new;
    end

end