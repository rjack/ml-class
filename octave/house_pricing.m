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

alpha = 0.0001;




function [th0_step, th1_step] = gradient_descent_step(th0, th1, xs, ys, m, alpha)

% usage: gradient_descent_step (th0, th1, xs, ys, m, alpha)
%
% This function computes the values th0_step and th1_step that must be
% subtracted respectively from theta zero and theta one in order to move them
% nearer the minimum of the cost function.

	distances = (th0 + th1 * xs) - ys;

	th0_step = alpha * sum(distances) / m;
	th1_step = alpha * sum(distances .* xs) / m;

endfunction


function j = cost (th0, th1, xs, ys, m)

% usage: cost (th0, th1, xs, ys, m)
%
% Compute the cost function J.

	j = sum((th0 + th1 * xs - ys)^2) / (2 * m);

endfunction




function res = converge (dx, dy, small_enough = 0)

% usage:  converge (dx, dy)
%         converge (dx, dy, threshold)
%
% This function returns true if dx and dy are converging. Used to stop the
% gradient_descent_step iterations.
%
% It's useful to experiment for different values of threshold: small values
% of threshold mean more iterations in order to converge.
%
% If threshold is not specified, zero is assumed.

	if (abs(dx) <= small_enough && abs(dy) <= small_enough)
		res = true;
	else
		res = false;
	endif
endfunction




%
% Examples are randomly generated.
%

xs = x_min + rand(1, m) * (x_max - x_min);
ys = 3 * xs + -(xs/2) + xs .* rand(1, m);


%
% Thetas. TODO: experiment with different values.
%
th0 = 1;
th1 = 1;


%
% Scatterplot of data.
%
figure;
plot(xs, ys, "x");
hold on;


%
% Gradient Descent.
% Call gradient_descent_step until values converge.
%
tries = max_tries = 1000;
while (tries--)
	% Uncomment this to plot each iteration. Beware, it's SLOW.
	% plot(x_min:x_max, th0 + th1 * (x_min:x_max), "g");

	[th0_step, th1_step] = gradient_descent_step(th0, th1, xs, ys, m, alpha);
	% TODO: try converge with different thresholds.
	if (converge(th0_step, th1_step, alpha))
		break;
	else
		th0 = th0 - th0_step;
		th1 = th1 - th1_step;

		if (isnan(th0) || isnan(th1))
			printf("Diverged\n");
			break;
		endif
	endif
endwhile


%
% Plot the hypothesis that minimizes the cost function
%
plot(x_min:x_max, th0 + th1 * (x_min:x_max), "r");


printf("After %d iterations (out of %d)\n", max_tries - tries - 1, max_tries);
printf("th0 = %g\n", th0);
printf("th1 = %g\n", th1);
