# Old Faithful dataset を取得して正規化
data("faithful");
xx <- scale(faithful, apply(faithful, 2, mean), apply(faithful, 2, sd));

# 1. パラメータを初期化する。
first_param <- function(xx, init_param, K) {
	D <- ncol(xx);
	N <- nrow(xx)
	param <- list(
	    alpha = numeric(K) + init_param$alpha + N / K,
	    beta  = numeric(K) + init_param$beta + N / K,
	    nyu   = numeric(K) + init_param$nyu + N / K,
	    W     = list(),
	    m     = matrix(rnorm(K * D), nrow=K)
	);
	for(k in 1:K) param$W[[k]] <- diag(D);
	param;
}


# 2. (10.65)～(10.67) により負担率 r_nk を得る
VB_Estep <- function(xx, param) {
	K <- length(param$alpha);
	D <- ncol(xx);

	# (10.65)
	ln_lambda <- sapply(1:K, function(k) {
		sum(digamma((param$nyu[k] + 1 - 1:D) / 2)) + D * log(2) + log(det(param$W[[k]]));
	});

	# (10.66)
	ln_pi <- exp(digamma(param$alpha) - digamma(sum(param$alpha)));

	# (10.67)
	t(apply(xx, 1, function(x){
		quad <- sapply(1:K, function(k) {
			xm <- x - param$m[k,];
			t(xm) %*% param$W[[k]] %*% xm;
		});
		ln_rho <- ln_pi + ln_lambda / 2 - D / 2 / param$beta - param$nyu / 2 * quad;
		ln_rho <- ln_rho - max(ln_rho);   # exp を Inf にさせないよう
		rho <- exp(ln_rho);
		rho / sum(rho);
	}));
}


# 3. r_nk を用いて、(10.51)～(10.53) により統計量 N_k, x_k, S_k を求め、
# それらを用いて、(10.58), (10.60)～(10.63) によりパラメータ α_k, m_k, β_k, ν_k, W_k を更新する。
VB_Mstep <- function(xx, init_param, resp) {
	K <- ncol(resp);
	D <- ncol(xx);
	N <- nrow(xx);

	# (10.51)
	N_k <- colSums(resp);

	# (10.52)
	x_k <- (t(resp) %*% xx) / N_k;

	# (10.53)
	S_k <- list();
	for(k in 1:K) {
		S <- matrix(numeric(D * D), D);
		for(n in 1:N) {
			x <- xx[n,] - x_k[k,];
			S <- S + resp[n,k] * ( x %*% t(x) );
		}
		S_k[[k]] <- S / N_k[k];
	}

	param <- list(
	  alpha = init_param$alpha + N_k,    # (10.58)
	  beta  = init_param$beta + N_k,     # (10.60)
	  nyu   = init_param$nyu + N_k,      # (10.63)
	  W     = list()
	);

	# (10.61)
	param$m <- (init_param$beta * init_param$m + N_k * x_k) / param$beta;

	# (10.62)
	W0_inv <- solve(init_param$W);
	for(k in 1:K) {
		x <- x_k[k,] - init_param$m[k,];
		Wk_inv <- W0_inv + N_k[k] * S_k[[k]] + init_param$beta * N_k[k] * ( x %*% t(x)) / param$beta[k];
		param$W[[k]] <- solve(Wk_inv);
	}

	param;
}

#sink(format(Sys.time(), "%m%d%H%M.txt"));
K <- 6;
nokori <- numeric(6);
for(i in 1:10000) {
	init_param <- list(
		i=i,
		alpha= 10^runif(1, min=-4, max=2),
		beta=runif(1, min=1, max=6)^2, 
		nyu=ncol(xx)+runif(1, min=-1, max=1)
	);
	param <- first_param(xx, init_param, K);
	init_param$m <- param$m;
	# print(init_param);
	init_param$W <- param$W[[1]];


	# 以降、収束するまで繰り返し
	for(j in 1:50) {
		resp <- VB_Estep(xx, param);
		#plot(xx, col=rgb(resp[,1],0,resp[,2]), xlab=paste(sprintf(" %1.3f",t(param$m)),collapse=","), ylab="");
		#points(param$m, pch = 8);
		param <- VB_Mstep(xx, init_param, resp);
	}

	# print("N_k");
	# print(colSums(resp), width=200);
	# print(param, width=200);
	
	N_k <- colSums(resp);
	n <- 0;
	for(k in 1:K) {
		if (N_k[k] >= 1) n <- n + 1;
	}
	nokori[n] <- nokori[n] + 1;
	if (i %% 10 == 0) print(nokori);
}
print(nokori);
#sink();

