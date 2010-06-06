# Hybrid Monte Carlo sampling

N <- 1000; # number of sampling

hmc_sampling <- function(N, dif_E, leapforg_count=100, leapforg_epsilon=0.01) {
	r <- rnorm(1,0,1);
	z <- 1;
	zlist <- numeric(N);

	for(i in 1:N) {
		# leapfrog
		e <- sample(c(-1,1), 1) * leapforg_epsilon;
		r <- r - e * dif_E(z) / 2;
		for(j in 1:leapforg_count) {
			z <- z + e * r;
			r <- r - e * dif_E(z);
		}
		r <- r - e * dif_E(z) / 2;

		zlist[i] <- z;

		# resampling of r
		# p(r|z) = p(r) = N(0,1);
		r <- rnorm(1,0,1);
	}
	zlist;
}

N <- 1000;

# p(z) = N(0,1) = exp(-z^2/2)/sqrt(2pi)
# E(z) = z^2/2, Zp = sqrt(2pi), dE/dz = z
# Hamiltonian: H(z,r) = E(z) + K(r) = z^2/2 + r^2/2

zlist <- hmc_sampling(N, function(z)z);
hist(zlist, breaks=20, main="N(1,0)");

# p(z) = Gamma(a,b) = 1/Zp * exp((a-1)ln z - bz)
# E(z) = -(a-1)ln z + bz, dE/dz = b - (a-1)/z

a <- 3;
b <- 1;
zlist <- hmc_sampling(N, function(z)b-(a-1)/z);
hist(zlist, breaks=20, main="Gamma(3,1)");



