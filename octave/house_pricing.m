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

conf.sensibility = alpha;

%
% Examples are randomly generated.
%
xs = x_min + rand(1, m) * (x_max - x_min);
fuzzyness = 1000 .* rand(1, m);
ys = 1200 + xs / 2 + fuzzyness;


%
% Thetas. TODO: experiment with different values.
%
th0 = 0;
th1 = 0;




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

	distances = (th0 + th1 * xs) - ys;

	j = sum(distances.^2) / (2 * m);

endfunction




function res = converge (dx, dy, sensibility = 0)

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

	if (abs(dx) <= sensibility && abs(dy) <= sensibility)
		res = true;
	else
		res = false;
	endif

endfunction




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
tries_left = max_tries = 1000000;
th0_steps = [];
th1_steps = [];
while (tries_left--)

	[th0_step, th1_step] = gradient_descent_step(th0, th1, xs, ys, m, alpha);
	% TODO: try converge with different alpha thresholds.
	if (converge(th0_step, th1_step, conf.sensibility))
		break;
	else
		th0 = th0 - th0_step;
		th1 = th1 - th1_step;

		% save the step values, so we can plot them in the end.
		th0_steps(end + 1) = th0_step;
		th1_steps(end + 1) = th1_step;

		if (isnan(th0) || isnan(th1))
			printf("Diverged\n");
			break;
		endif
	endif
endwhile

number_of_tries = length(th0_steps);

%
% Plot the hypothesis that minimizes the cost function
%
plot(x_min:x_max, th0 + th1 * (x_min:x_max), "r");
title("Hypothesis");
xlabel("feet^2");
ylabel("K dollars");

%
% Plot the increments so we can see how they converge / diverge
%
figure;
plot(1:number_of_tries, th0_steps, th1_steps);
title("Increments");
xlabel("algorithm iterations");
ylabel("step length");

%
% Plot the cost function for the given set of examples
%
% From: http://www.ml-class.org/course/qna/view?id=68
figure;
tx = ty = -10:0.1:10;
[xx, yy] = meshgrid(tx, ty);
h = zeros(size(xx)(1), size(xx)(2), m);
tz = zeros(size(xx));

for i = 1:m
	h(:,:,i) = xx .+ yy * xs(i);
	tz = tz + (h(:,:,i) - ys(i)) .^ 2;
endfor
tz = tz / (2 * m);

meshc(tx, ty, tz);
title("Cost function");
xlabel("th0");
ylabel("th1");
zlabel("J(th0,th1)");

figure;
contour(tx, ty, tz, 1:5);
%title("Cost function");
xlabel("th0");
ylabel("th1");
%zlabel("J(th0,th1)");

printf("After %d iterations (out of %d)\n", number_of_tries, max_tries);
printf("th0 = %g\n", th0);
printf("th1 = %g\n", th1);
