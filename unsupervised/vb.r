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
		tryCatch({
			sum(digamma((param$nu[k] + 1 - 1:D) / 2)) + D * log(2) + log(det(param$W[[k]]));
		}, error=function(e){
			print(param$W);
			0;
		})
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

ALPHA <- 0.001;
argv <- commandArgs(T);
if (length(argv)>0) ALPHA <- as.numeric(commandArgs(T))[1];

sink(format(Sys.time(), "%m%d%H%M.txt"));
K <- 6;
nokori <- numeric(6);
for(i in 1:1000) {
	init_param <- list(
		alpha = ALPHA,  # 10^runif(1, min=-4, max=2),
		beta  = runif(1, min=1, max=6)^2, 
		nu    = ncol(xx)+runif(1, min=-1, max=1),
		m     = matrix(numeric(K * ncol(xx)), nrow=K),
		W     = diag(ncol(xx))
	);
	param <- first_param(xx, init_param, K);

	cat(sprintf("%d: alpha=%.5f, beta=%.3f, nu=%.3f\n", i, init_param$alpha, init_param$beta, init_param$nu));
	print(param$m);

	# 以降、収束するまで繰り返し
	for(j in 1:100) {
		resp <- VB_Estep(xx, param);
		new_param <- VB_Mstep(xx, init_param, resp);
		if (is.null(new_param)) break;
		param <- new_param;
	}

	N_k <- colSums(resp);

	print(paste("N_k:", sprintf(" %1.3f",N_k),collapse=","));
	#print(param, width=200);
	#drawGraph(resp, param$m);

	n <- 0;
	for(k in 1:K) {
		if (N_k[k] >= 0.1) n <- n + 1;
	}
	nokori[n] <- nokori[n] + 1;
}
print("remained components:");
print(nokori);
sink();

