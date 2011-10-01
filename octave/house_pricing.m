m = 100;
x_min = 50;
x_max = 200;

alpha = 0.000001;


function [th0_step, th1_step] = gradient_descent_step(th0, th1, xs, ys, m, alpha)
	distances = (th0 + th1 * xs) - ys;

	th0_step = alpha * sum(distances) / m;
	th1_step = alpha * sum(distances .* xs) / m;
endfunction


function j = cost(th0, th1, xs, ys, m)
	sum(th0 + th1 * xs - ys)/m;
endfunction

xs = x_min + rand(1, m) * (x_max - x_min);

ys = [];
for x = xs
	ys(end + 1) = x + (-(x/2) + (rand(1,1) * x));
endfor

th0 = 0;
th1 = 1;

figure;
plot(xs, ys, "x");
hold on;


for foo = 1:200
	plot(x_min:x_max, th0 + th1 * (x_min:x_max))
	[th0_step, th1_step] = gradient_descent_step(th0, th1, xs, ys, m, alpha);
	if (th0_step == 0 && th1_step == 0)
		break;
	else
		th0 = th0 - th0_step;
		th1 = th1 - th1_step;
	endif
endfor

disp(th0);
disp(th1);
