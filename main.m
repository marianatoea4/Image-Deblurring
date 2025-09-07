clc; clear; close all;

% imagine - matrice - culori alb-negru, 0<=pixel<=1
Image = im2double(rgb2gray(imread('original.jpg')));

% redimensionare imagine, sa aiba maxim 100 pe linie/coloana
max_size = 100;
[m, n] = size(Image);
scale = max_size / max(m, n);
Image_resized = imresize(Image, scale); % imagine originala - matrice
[m, n] = size(Image_resized);

% numarul de linii din vectorii coloana
mn = m * n;

% imaginea - matrice de pixeli(valori)
x_true = Image_resized(:); % transform imaginea matrice in imagine vector, imaginea originala 

% valoarea blur-ului, valoare mai mare blur mai mare, valoare mai mica blur mai mic
blur = 10;

% creez matricea de blur(matrice de mediere) 
D = create_blur_matrix(mn, blur);

y = D * x_true; % creez imaginea blurata - vector
Y = reshape(y, m, n); % transform imaginea blurata vector in imagine blurata matrice

% figure;
% subplot(1,2,1); imshow(Image_resized); title('Originala');
% subplot(1,2,2); imshow(Y); title('Blurata');

% Image_resized - imaginea originala sub forma de matrice
% x_true - imaginea originala sub forma de vector
% Y - imaginea blurata sub forma de matrice
% y - imaginea blurata sub forma de vector


% ------------------------------------------------------------------------
%                       CVX
% Rezolvarea problemei de optimizare folosind CVX
tic;
cvx_begin
    variable x(mn)    % variabila de decizie
    minimize (square_pos(norm(D*x-y)));
    subject to
        0 <= x <= 1     % constrangerile
cvx_end
t_cvx = toc;

X_cvx = reshape(x, m, n);  % transform imaginea obtinuta din vector in matrice
% figure;
% subplot(1,3,1); imshow(Image_resized); title('Originala');
% subplot(1,3,2); imshow(Y); title('Blurata');
% subplot(1,3,3); imshow(X_cvx); title('Deblurata cu CVX');

% ------------------------------------------------------------------------
%                   Cu fct. MatLab: fmincon

% Fct obiectiv: f(x) = ||Dx - y||^2
fun = @(x) norm(D * x - y)^2;

% Dimensiune var.
mn = length(y);

% Cstr.: 0 <= x <= 1
lb = zeros(mn,1);
ub = ones(mn,1);

% Punct initial
x0 = 0.5 * zeros(mn,1);

% Optiuni (pentru a urmari progresul si a seta limita de iteratii)
options = optimoptions('fmincon', 'Display', 'iter', ...
    'Algorithm', 'interior-point', ...
    'MaxIterations', 1000, ...
    'MaxFunctionEvaluations', 1e5, ...
    'OptimalityTolerance', 1e-6);

% Rezolvare
tic;
[x_fmincon, fval_fmincon, exitflag, output] = fmincon(fun, x0, [], [], [], [], lb, ub, [], options);
t_fmincon = toc;

% Reconstruire imagine
X_fmincon = reshape(x_fmincon, m, n);


% ------------------------------------------------------------------------
%                   METODA GRADIENT PROIECTAT 

max_iter = 1000;
c = 10;
epsilon = 1e-2;
tic;
[x_gp, iter_vec, diff_vec] = gradient_proiectat(D, y, max_iter, c, epsilon);
t_gp = toc;
X_gp = reshape(x_gp, m, n);


% figure;
% subplot(2,2,1); imshow(Image_resized); title('Originala');
% subplot(2,2,2); imshow(Y); title('Blurata');
% subplot(2,2,3); imshow(X_cvx); title('Deblurata cu CVX');
% subplot(2,2,4); imshow(X_gp); title('Deblurata cu MGP');
% 
% figure;
% semilogy(iter_vec, diff_vec, '-o');
% xlabel('Iteratii');
% ylabel('Norma ||X_{k+1} - X_k||');
% title('Criteriu de oprire pentru Gradient Proiectat');
% grid on;

% ------------------------------------------------------------------------
%                 METODA GRADIENT CONDITIONAL (FRANK-WOLFE)


max_iter1 = 1000;
epsilon1 = 1e-2;
tic;
[x_gc, iter_gc, diff_gc] = gradient_conditional(D, y, max_iter1, epsilon1);
t_gc = toc;
X_gc = reshape(x_gc, m, n);


% cu metodele de optimizare alese de mine
figure;
subplot(2,2,1); imshow(Image_resized); title('Originala');
subplot(2,2,2); imshow(Y); title('Blurata');
subplot(2,2,3); imshow(X_gp); title('Deblurata cu MGP');
subplot(2,2,4); imshow(X_gc); title('Deblurata cu MGC');

% cu cvx si functii matlab
figure;
subplot(2,2,1); imshow(Image_resized); title('Originala');
subplot(2,2,2); imshow(Y); title('Blurata');
subplot(2,2,3); imshow(X_cvx); title('Deblurata cu CVX');
subplot(2,2,4); imshow(X_fmincon); title('Deblurata cu fmincon');

% grafice pentru convergenta algoritmilor MGP si MGC
figure;
semilogy(iter_gc, diff_gc, '-o'); hold on;
semilogy(iter_vec, diff_vec, '-x');
legend('Gradient Conditional', 'Gradient Proiectat');
xlabel('Iteratii');
ylabel('Norma ||X_{k+1} - X_k||');
title('Convergenta algoritmilor');
grid on;

%      Comparare timpi de executie intre metode

% Vector cu timpi (in secunde) pentru toate metodele
times = [t_fmincon, t_gp, t_gc, t_cvx];

% Etichete metode
labels = {'fmincon', 'MGP', 'MGC', 'CVX'};

% Grafic bara
figure;
bar(times);
set(gca, 'xticklabel', labels);
ylabel('Timp executie (secunde)');
title('Comparatie timpi de executie pentru metodele de deblur');
grid on;

% Comparatie intre acuratetea rezultatelor
mse_cvx = immse(X_cvx, Image_resized);
mse_fmincon = immse(X_fmincon, Image_resized);
mse_mgp = immse(X_gp, Image_resized);
mse_mgc = immse(X_gc, Image_resized);

% Grafic MSE
mse_vals = [mse_cvx, mse_fmincon, mse_mgp, mse_mgc];
labels = {'CVX', 'fmincon', 'MGP', 'MGC'};

figure;
bar(mse_vals);
set(gca, 'xticklabel', labels);
ylabel('MSE');
title('Comparatie Mean Squared Error');
grid on;


% comparatie intre numarul de iteratii
num_iter_gp = length(iter_vec);
num_iter_gc = length(iter_gc);
num_iter_fmincon = output.iterations;
num_iter_cvx = 36;

num_iter = [num_iter_fmincon, num_iter_gp, num_iter_gc, num_iter_cvx];
labels = {'fmincon', 'MGP', 'MGC', 'CVX'};

figure;
bar(num_iter);
set(gca, 'xticklabel', labels);
ylabel('Numar de iterarii');
title('Comparatie intre numarul de iteratii');
grid on;