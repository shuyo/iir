
polynomial_basis_func <- function(M) {
	lapply(1:M, function(u){ u1=u; function(x) x^(u1-1) })
}
gaussian_basis_func <- function(M, has_bias=T, s=0.1) {
	phi <- c()
	if (has_bias) phi <- function(x) 0*x+1 # bias
	u_i <- seq(0,1,length=ifelse(has_bias, M-1, M))
	append(phi, lapply(u_i, function(u){ u1=u; function(x) exp(-(x-u1)^2/(2*s*s)) }))
}
sigmoid_basis_func <- function(M, has_bias=T, s=0.1) {
	phi <- c()
	if (has_bias) phi <- function(x) 0*x+1 # bias rep(1, length(x))
	u_i <- seq(0,1,length=ifelse(has_bias, M-1, M))
	append(phi, lapply(u_i, function(u){ u1=u; function(x) 1/(1+exp(-(x-u1)/s)) } ))
}

xlist <- seq(0, 1, length=250)
tlist <- sin(2*pi*xlist)+rnorm(length(xlist), sd=0.2)
D <- data.frame(x=xlist, t=tlist)

# PRML's synthetic data set
curve_fitting <- data.frame(
	x=c(0.000000,0.111111,0.222222,0.333333,0.444444,0.555556,0.666667,0.777778,0.888889,1.000000),
	t=c(0.349486,0.830839,1.007332,0.971507,0.133066,0.166823,-0.848307,-0.445686,-0.563567,0.261502)
)

calc_evidence <- function(phi, D, alpha=2, beta=25, graph=NULL) {
	M <- length(phi)
	N <- length(D$x)
	PHI <- sapply(phi, function(f)f(D$x))

	if (!is.null(graph)) {
		plot(graph, lty=2, col="blue", xlim=c(0,1), ylim=c(-1.1,1.1), ylab="")
		par(new=T)
		plot(D, xlim=c(0,1), ylim=c(-1.1,1.1), xlab="", ylab="")
	}

	if (beta=="ml") {
		w_ML <- solve(t(PHI) %*% PHI) %*% t(PHI) %*% D$t
		loss_ML <- D$t - PHI %*% w_ML
		beta <- N / sum(loss^2)
		if (!is.null(graph)) {
			par(new=T)
			plot( function(x) sapply(phi, function(f)f(x)) %*% w_ML , col="red", xlim=c(0,1), ylim=c(-1.1,1.1), ylab="")
		}
	}

	A <- alpha * diag(M) + beta * t(PHI) %*% PHI  # equal to S_N(PRML 3.54)
	m_N <- beta * solve(A) %*% t(PHI) %*% D$t
	loss_m_N <- D$t - PHI %*% m_N
	E_m_N <- beta / 2 * sum(loss_m_N^2) + alpha / 2 * sum(m_N^2)

	if (!is.null(graph)) {
		par(new=T)
		plot( function(x) sapply(phi, function(f)f(x)) %*% m_N, xlim=c(0,1), ylim=c(-1.1,1.1), ylab="")
	}

	# model evidence
	c(M/2*log(alpha) + N/2*log(beta) - E_m_N - 1/2*log(det(A)) - N/2*log(2*pi), beta)
}


a<-sapply(1:9, function(n) calc_evidence(polynomial_basis_func(n), curve_fitting, alpha=5e-3))

orig_func <- function(x)sin(2*pi*x)
calc_evidence(gaussian_basis_func(9, F, s=0.37), D, beta="ml", alpha=2, graph=orig_func)
calc_evidence(polynomial_basis_func(4), curve_fitting, alpha=5e-3, beta="ml", graph=orig_func)

calc_evidence(gaussian_basis_func(6, F), D0, alpha=2, beta="ml", graph=orig_func)

# ----

> a<-sapply(1:9, function(n) calc_evidence(polynomial_basis_func(n), curve_fitting, alpha=5e-3, beta="ml"))
> data.frame(M=0:8, evidence=a[1,], beta_ML=a[2,])
  M  evidence   beta_ML
1 0 -13.60463  2.649926
2 1 -14.48098  4.680463
3 2 -16.60761  4.752649
4 3 -14.38654 28.600038
5 4 -14.20562 28.651286
6 5 -15.12706 29.206330
7 6 -15.86874 30.294868
8 7 -16.43925 30.954700
9 8 -17.37590 35.353486
> a<-sapply(1:9, function(n) calc_evidence(polynomial_basis_func(n), curve_fitting, alpha=5e-3, beta=11.1))
> data.frame(M=0:8, evidence=a[1,], beta_ML=a[2,])
  M  evidence beta_ML
1 0 -23.10268    11.1
2 1 -17.88419    11.1
3 2 -20.30879    11.1
4 3 -13.93411    11.1
5 4 -13.71294    11.1
6 5 -14.35868    11.1
7 6 -14.98120    11.1
8 7 -15.48112    11.1
9 8 -15.90587    11.1

# ----

a<-sapply(1:9, function(n) calc_evidence(gaussian_basis_func(n), D0, alpha=2, beta="ml"))
data.frame(n=1:9, evidence=a[1,], beta_ML=a[2,])

  n   evidence   beta_ML
1 1  -24.15372  1.991818
2 2  -26.00534  2.080833
3 3  -28.32203  2.199902
4 4  -28.98627  2.204309
5 5 -141.40382 11.622494
6 6 -262.51315 20.230023
7 7 -531.38789 39.600405
8 8 -558.69096 41.400126
9 9 -566.97144 41.904756

a<-sapply(1:9, function(n) calc_evidence(gaussian_basis_func(n, F), D0, alpha=2, beta="ml"))
data.frame(n=1:9, evidence=a[1,], beta_ML=a[2,])
  n   evidence   beta_ML
1 1  -23.54826  2.069294
2 2  -27.36097  2.194946
3 3  -28.18806  2.194992
4 4 -139.53185 11.503610
5 5 -260.09231 20.131365
6 6 -523.95049 39.104824
7 7 -555.47110 41.216737
8 8 -564.31398 41.761091
9 9 -587.98582 43.413693

a<-sapply(1:9, function(n) calc_evidence(gaussian_basis_func(n, F), curve_fitting, alpha=2, beta="ml"))
data.frame(n=1:9, evidence=a[1,], beta_ML=a[2,])
  n    evidence   beta_ML
1 1   -9.947638  2.844855
2 2  -10.661711  2.849207
3 3  -10.915992  2.895102
4 4  -32.215789  9.113815
5 5  -84.841899 21.990299
6 6  -84.348342 21.365737
7 7 -110.852386 27.425745
8 8 -119.979257 29.427823
9 9 -131.627930 32.218532

a<-sapply(1:9, function(n) calc_evidence(gaussian_basis_func(n), curve_fitting, alpha=2, beta="ml"))
data.frame(n=1:9, evidence=a[1,], beta_ML=a[2,])
  n    evidence   beta_ML
1 1   -8.439416  2.649926
2 2   -9.732442  2.903392
3 3  -10.598375  2.953347
4 4  -11.048473  2.954229
5 5  -32.992820  9.370027
6 6  -93.909226 24.235243
7 7  -85.088391 21.481192
8 8 -132.661228 32.884408
9 9 -145.904973 35.879891

