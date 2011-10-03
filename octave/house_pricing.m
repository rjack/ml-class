%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Gradient Descent algorithm for Cost Function optimization in Linear
% Regression.
%
%
% Lectures: II. Linear Regression With One Variable
%
%     http://www.ml-class.org/course/video/list
%
%
% First time I touch GNU Octave and first time I do numerical computations, so
% bear with me.
%
%
% For comments, corrections and improvements:
% 
%     https://github.com/rjack/ml-class/issues
%
%
% This work is released under the WTFPL:
%
%     http://sam.zoy.org/wtfpl/COPYING
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


m = 100;
x_min = 50;
x_max = 200;

alpha = 0.00001;


function [th0_step, th1_step] = gradient_descent_step(th0, th1, xs, ys, m, alpha)
	distances = (th0 + th1 * xs) - ys;

	th0_step = alpha * sum(distances) / m;
	th1_step = alpha * sum(distances .* xs) / m;
endfunction


function j = cost (th0, th1, xs, ys, m)
	j = sum((th0 + th1 * xs - ys)^2) / (2 * m);
endfunction


function res = converge (dx, dy, small_enough = 0)
	if (abs(dx) <= small_enough && abs(dy) <= small_enough)
		res = true;
	else
		res = false;
	endif
endfunction


function die_if_diverged (x)
	if (isnan(x))
		printf("Diverged!\n");
		exit(1);
	endif
endfunction


xs = x_min + rand(1, m) * (x_max - x_min);
ys = 3 * xs + -(xs/2) + xs .* rand(1, m);

th0 = 0;
th1 = 10;

figure;
plot(xs, ys, "x");
hold on;


tries = max_tries = 1000;
while (tries--)
	plot(x_min:x_max, th0 + th1 * (x_min:x_max))
	[th0_step, th1_step] = gradient_descent_step(th0, th1, xs, ys, m, alpha);
	if (converge(th0_step, th1_step, alpha))
		break;
	else
		th0 = th0 - th0_step;
		th1 = th1 - th1_step;

		die_if_diverged(th0);
		die_if_diverged(th1);
	endif
endwhile

printf("After %d iterations (out of %d)\n", max_tries - tries - 1, max_tries);
printf("th0 = %g\n", th0);
printf("th1 = %g\n", th1);
