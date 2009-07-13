
polynomial_basis_func <- function(M) {
	lapply(1:M, function(u){ u1=u; function(x) x^(u1-1) })
}
gaussian_basis_func <- function(M, has_bias=T, s=0.1) {
	phi <- c()
	if (has_bias) phi <- function(x) rep(1, length(x)) # bias
	u_i <- seq(0,1,length=ifelse(has_bias, M-1, M))
	append(phi, lapply(u_i, function(u){ u1=u; function(x) exp(-(x-u1)^2/(2*s*s)) }))
}
sigmoid_basis_func <- function(M, has_bias=T, s=0.1) {
	phi <- c()
	if (has_bias) phi <- function(x) rep(1, length(x)) # bias
	u_i <- seq(0,1,length=ifelse(has_bias, M-1, M))
	append(phi, lapply(u_i, function(u){ u1=u; function(x) 1/(1+exp(-(x-u1)/s)) } ))
}

# PRML's synthetic data set
curve_fitting <- data.frame(
	x=c(0.000000,0.111111,0.222222,0.333333,0.444444,0.555556,0.666667,0.777778,0.888889,1.000000),
	t=c(0.349486,0.830839,1.007332,0.971507,0.133066,0.166823,-0.848307,-0.445686,-0.563567,0.261502)
)

calc_evidence <- function(phi, D, alpha=2, beta=25) {
	M <- length(phi)
	N <- length(D$x)

	PHI <- sapply(phi, function(f)f(D$x))
	A <- alpha * diag(M) + beta * t(PHI) %*% PHI  # equal to S_N(PRML 3.54)
	m_N <- beta * solve(A) %*% t(PHI) %*% D$t
	E_m_N <- beta / 2 * sum((D$x - PHI %*% m_N)^2) + alpha / 2 * sum(m_N^2)

	# model evidence
	M/2*log(alpha) + N/2*log(beta) - E_m_N - 1/2*log(det(A)) - N/2*log(2*pi)
}

calc_evidence(gaussian_basis_func(9), D)
calc_evidence(polynomial_basis_func(1), curve_fitting, alpha=5e-3)

