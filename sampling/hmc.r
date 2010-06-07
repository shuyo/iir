# Hybrid Monte Carlo sampling

N <- 10000; # number of sampling

hmc_sampling <- function(N, E, partial_E, leapforg_count=100, leapforg_epsilon=0.01) {
	r <- rnorm(1,0,1);
	z <- 1;
	zlist <- numeric(N);

	for(i in 1:N) {
		H <- E(z) + r^2/2;
		# leapfrog
		e <- sample(c(-1,1), 1) * leapforg_epsilon;
		z2 <- z;
		r2 <- r - e * partial_E(z2) / 2;
		for(j in 1:leapforg_count) {
			z2 <- z2 + e * r2;
			r2 <- r2 - e * partial_E(z2);
		}
		r2 <- r2 - e * partial_E(z2) / 2;
		dH <- H - (E(z2) + r2^2/2);
		if (dH > 0 || runif(1) < exp(dH)) {
			z <- z2;
			zlist[i] <- z;

			# resampling of r from p(r|z) = p(r) = N(0,1)
			r <- rnorm(1,0,1);
		} else {
			cat(sprintf("%d: rejected\n", i));
		}
	}
	zlist;
}

# p(z) = N(0,1) = exp(-z^2/2)/sqrt(2pi)
# E(z) = z^2/2, Zp = sqrt(2pi), dE/dz = z
# Hamiltonian: H(z,r) = E(z) + K(r) = z^2/2 + r^2/2

zlist <- hmc_sampling(N, function(z)z**2/2, function(z)z);
hist(zlist, breaks=20, main=sprintf("N(1,0), mean=%.3f, var=%.3f", mean(zlist), var(zlist)));
acf(zlist);

# p(z) = Gamma(a,b) = 1/Zp * exp((a-1)ln z - bz)
# E(z) = -(a-1)ln z + bz, dE/dz = b - (a-1)/z

a <- 3;
b <- 2;
zlist <- hmc_sampling(N, function(z)b*z-(a-1)*log(z), function(z)b-(a-1)/z);
hist(zlist, breaks=20, main=sprintf("Gamma(%d,%d), mean=%.3f, var=%.3f", a, b, mean(zlist), var(zlist)));
acf(zlist);



