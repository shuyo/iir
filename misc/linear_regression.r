xlist <- runif(25)
D <- data.frame(x=xlist, t=sin(2*pi*xlist))

by(D, D$x, phi)
phi(D$x)


> z <- c( function(x)(x+1), function(x)(x+2) )
> z
[[1]]
function(x)(x+1)

[[2]]
function(x)(x+2)

> z[1](3)
 エラー：  関数でないものを適用しようとしました 
> z[[1]](3)
[1] 4
> lapply(z, function(f)f(1:3))
[[1]]
[1] 2 3 4

[[2]]
[1] 3 4 5

> z1 <- lapply(z, function(f)f(1:3))
> as.vector(z1)
[[1]]
[1] 2 3 4

[[2]]
[1] 3 4 5

> as.matrix(z1)
     [,1]     
[1,] Numeric,3
[2,] Numeric,3
> rbind(z1)
   [,1]      [,2]     
z1 Numeric,3 Numeric,3
> cbind(z1)
     z1       
[1,] Numeric,3
[2,] Numeric,3
> cbind(lapply(z1, as.vector))
     [,1]     
[1,] Numeric,3
[2,] Numeric,3
> lapply(z1, as.vector)
[[1]]
[1] 2 3 4

[[2]]
[1] 3 4 5

> z1 <- sapply(z, function(f)f(1:3))
> z1
     [,1] [,2]
[1,]    2    3
[2,]    3    4
[3,]    4    5









    (p <- function(x, t){ 
        mapply(function(x, t)dnorm(t, m=(t(m_N) %*% phi(x))[1], s=var_N(x), log=T), x, t)
    })

    p1 <- function(x, t){ 
        dnorm(t, m=(t(m_N) %*% phi(x))[1], s=var_N(x), log=T)
    }


M <- 9
phi <- c(function(x)rep(1,length(x)))
phi <- append(phi, lapply(seq(0,1,length=M-1), function(u){u1=u; function(x)exp(-(x-u1)^2/(2*s*s))}))


gaussian_basis_func <- function(M, has_bias=T, s=0.1) {
	phi <- c()
	if (has_bias) phi <- function(x) rep(1, length(x)) # bias
	u_i <- seq(0,1,length=ifelse(has_bias, M-1, M))
	append(phi, lapply(u_i, function(u){ u1=u; function(x) exp(-(x-u1)^2/(2*s*s)) }))
}
polynomial_basis_func <- function(M) {
	lapply(1:M, function(u){ u1=u; function(x) x^(u1-1) })
}

sapply(polynomial_basis_func(3), function(f)f(1:3))

curve_fitting <- data.frame(
	x=c(0.000000,0.111111,0.222222,0.333333,0.444444,0.555556,0.666667,0.777778,0.888889,1.000000),
	t=c(0.349486,0.830839,1.007332,0.971507,0.133066,0.166823,-0.848307,-0.445686,-0.563567,0.261502)
)

calc_evidence <- function(phi, D, alpha=2, beta=25) {
	M <- length(phi)
	N <- length(D$x)
	PHI <- sapply(phi, function(f)f(D$x))

	if (beta=="ml") {
		w_ML <- solve(t(PHI) %*% PHI) %*% t(PHI) %*% D$t
		beta_ML_inv <- 1/N * sum(lapply(1:N), function(n)D[n,]$x lapply(phi, function(f)
	}

	A <- alpha * diag(M) + beta * t(PHI) %*% PHI  # equal to S_N(PRML 3.54)
	m_N <- beta * solve(A) %*% t(PHI) %*% D$t
	E_m_N <- beta / 2 * sum((D$x - PHI %*% m_N)^2) + alpha / 2 * sum(m_N^2)

	# model evidence
	M/2*log(alpha) + N/2*log(beta) - E_m_N - 1/2*log(det(A)) - N/2*log(2*pi)
}

calc_evidence(gaussian_basis_func(9), D)
calc_evidence(polynomial_basis_func(1), curve_fitting, alpha=5e-3)


# expressions vector?




E(\vec w) = E(\vec m_N) + \frac 1 2 (\vec w - \vec m_N)^T A (\vec w - \vec m_N)

\integral exp \left\{ -E(\vec w) \right\} d\vec w



