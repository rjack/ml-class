examples_in_dataset = 100;

min_house_size = 50;
max_house_size = 200;

sizes = min_house_size + rand(1, examples_in_dataset) * (max_house_size - min_house_size);

prices = []
for s = sizes
	prices(end + 1) = s + (-(s/2) + (rand(1,1) * s));
endfor

figure;
plot(sizes, prices, "x");
hold on;
