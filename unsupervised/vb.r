# Variational Bayes (c)2010 Nakatani Shuyo

# Old Faithful dataset を取得して正規化
data("faithful");
xx <- scale(faithful, apply(faithful, 2, mean), apply(faithful, 2, sd));

# common
nan2zero <- function(x) ifelse(is.nan(x), 0, x);
drawGraph <- function(resp, m) {
	plot(xx, xlab=paste(sprintf(" %1.3f",N_k),collapse=","), ylab="",
		col=rgb(rowSums(resp[,1:3])*0.9, rowSums(resp[,3:5])*0.8, rowSums(resp[,c(2,4,6)])*0.9));
	points(m, pch = 8);
}

# calcurate normalization factor of Wishart distribution (except pi^D(D-1)/4)
calc_lnB <- function(W, nu) {
	D <- ncol(W);
	- nu / 2 * (log(det(W)) + D * log(2)) - sum(lgamma((nu + 1 - 1:D) / 2));
}

# 1. パラメータを初期化する。
first_param <- function(xx, init_param, K) {
	D <- ncol(xx);
	N <- nrow(xx)
	param <- list(
	    alpha = numeric(K) + init_param$alpha + N / K,
	    beta  = numeric(K) + init_param$beta + N / K,
	    nu    = numeric(K) + init_param$nu + N / K,
	    W     = list(),
	    m     = matrix(rnorm(K * D), nrow=K)
	);
	for(k in 1:K) param$W[[k]] <- init_param$W;
	param;
}

# 2. (10.65)～(10.67) により負担率 r_nk を得る
VB_Estep <- function(xx, param) {
	K <- length(param$alpha);
	D <- ncol(xx);

	# (10.65)
	ln_lambda <- sapply(1:K, function(k) {
		sum(digamma((param$nu[k] + 1 - 1:D) / 2)) + D * log(2) + log(det(param$W[[k]]));
	});

	# (10.66)
	ln_pi <- digamma(param$alpha) - digamma(sum(param$alpha));

	# (10.67)
	t(apply(xx, 1, function(x){
		quad <- sapply(1:K, function(k) {
			xm <- x - param$m[k,];
			t(xm) %*% param$W[[k]] %*% xm;
		});
		ln_rho <- ln_pi + ln_lambda / 2 - D / 2 / param$beta - param$nu / 2 * quad;
		rho <- exp(ln_rho - max(ln_rho));   # exp を Inf にさせないよう max を引く
		rho / sum(rho);
	}));
}

# 3. r_nk を用いて、(10.51)～(10.53) により統計量 N_k, x_k, S_k を求め、
# それらを用いて、(10.58), (10.60)～(10.63) によりパラメータ α_k, m_k, β_k, ν_k, W_k を更新する。
VB_Mstep <- function(xx, init_param, resp) {
	K <- ncol(resp);
	D <- ncol(xx);
	N <- nrow(xx);

	# (10.51) N_k は 0 になる可能性あり
	N_k <- colSums(resp);

	# (10.52)
	x_k <- nan2zero((t(resp) %*% xx) / N_k);

	# (10.53)
	S_k <- list();
	for(k in 1:K) {
		S <- matrix(numeric(D * D), D);
		for(n in 1:N) {
			x <- xx[n,] - x_k[k,];
			S <- S + resp[n,k] * ( x %*% t(x) );
		}
		S_k[[k]] <- nan2zero(S / N_k[k]);
	}

	param <- list(
	  alpha = init_param$alpha + N_k,    # (10.58)
	  beta  = init_param$beta + N_k,     # (10.60)
	  nu    = init_param$nu + N_k,      # (10.63)
	  W     = list()
	);

	# (10.61)
	param$m <- (init_param$beta * init_param$m + N_k * x_k) / param$beta;

	# (10.62)
	W0_inv <- solve(init_param$W);
	for(k in 1:K) {
		x <- x_k[k,] - init_param$m[k,];
		tryCatch({
			Wk_inv <- W0_inv + N_k[k] * S_k[[k]] + init_param$beta * N_k[k] * ( x %*% t(x)) / param$beta[k];
			param$W[[k]] <- solve(Wk_inv);
		}, error=function(e){
			print("Wk_inv error");
			print(N_k);
			print(S_k[[k]]);
			print(param$beta[k]);
			param$W[[k]] <<- diag(D);
		})
	}

	param;
}

# Variational Lower Bound
VB_LowerBound <- function(init_param, param, resp, N) {
	D <- ncol(param$m);
	K <- nrow(param$m);

	a <- lgamma(K * init_param$alpha) - K * lgamma(init_param$alpha) - lgamma(sum(param$alpha)) + sum(lgamma(param$alpha));
	b <- D / 2 * (K * log(init_param$beta) - sum(log(param$beta)));
	r <- sum(nan2zero(resp * log(resp)));
	L <- a + b - r - D * N / 2 * log(2 * pi);

	L <- L + K * calc_lnB(init_param$W, init_param$nu);
	for(k in 1:K) L <- L - calc_lnB(param$W[[k]], param$nu[k]);
	L;
}

# Variational Bayesian Inference
VB_inference <- function(xx, K, alpha_0, beta_0, nu_0) {
	init_param <- list(
		alpha = alpha_0,  
		beta  = beta_0, 
		nu    = nu_0, 
		m     = matrix(numeric(K * ncol(xx)), nrow=K),
		W     = diag(ncol(xx))
	);
	param <- first_param(xx, init_param, K);

	# 以降、収束するまで繰り返し
	pre_L <- -1e99;
	for(j in 1:999) {
		resp <- VB_Estep(xx, param);
		new_param <- VB_Mstep(xx, init_param, resp);
		if (is.null(new_param)) break;
		param <- new_param;
		L <- VB_LowerBound(init_param, param, resp, nrow(xx));
		#print(L);
		if (L - pre_L < 0) cat(sprintf("DECREASED LOWER BOUND! %f => %f\n", pre_L, L));
		if (L - pre_L < 0.0001) break;
		pre_L <- L;
	}

	param$resp <- resp;
	param$init <- init_param;
	param$convergence <- j;
	param$L <- L;
	param;
}


# command line
alpha_0 <- 1;
argv <- commandArgs(T);
if (length(argv)>0) alpha_0 <- as.numeric(commandArgs(T))[1];

# main
sink(format(Sys.time(), "vb%m%d%H%M.txt"));

I <- 5;
count <- 1;
for(K in 2:6) for(beta_0 in 1:20/20) for(nu_0 in ncol(xx)-1+1:20/2) {
	#cat(sprintf("#%d: K=%d, alpha=%f, beta=%f, nu=%f\n", count, K, alpha_0, beta_0, nu_0));

	max_L <- -1e99;
	for(i in 1:I) {
		param <- VB_inference(xx, K, alpha_0, beta_0, nu_0);

		N_k <- colSums(param$resp);
		n <- sum(N_k >= 0.5);
		#nonzeros[n] <- nonzeros[n] + 1;
		if (max_L < param$L) max_L <- param$L;

		cat(sprintf("#%d-%d: convergence=%d, L=%.3f, N_k = %d(%s)\n",
			count, i, param$convergence, param$L, n, paste(sprintf("%.3f",N_k), collapse=", ")));
		#drawGraph(param$resp, param$m);
	}
	cat(sprintf("#%d: K=%d, alpha=%f, beta=%f, nu=%f, max_L=%.3f, p(D|K)=%.3f\n", count, K, alpha_0, beta_0, nu_0, max_L, max_L+lfactorial(K)));

	count <- count + 1;
}
#cat(sprintf("non-zero components: %s\n", paste(sprintf("%d:%d", 1:K, nonzeros), collapse=", ")));

sink();

