# EM algorithm and Online EMA

# Old Faithful dataset を取得して正規化
data("faithful");
xx <- scale(faithful, apply(faithful, 2, mean), apply(faithful, 2, sd));

# パラメータの初期化(平均、共分散、混合率)
init_param <- function(K, D) {
	sig <- list();
	for(k in 1:K) sig[[k]] <- diag(K);
	list(mu = matrix(rnorm(K * D), D), mix = numeric(K)+1/K, sig = sig);
}

# 多次元正規分布密度関数
dmnorm <- function(x, mu, sig) {
	D <- length(mu);
	1/((2 * pi)^D * sqrt(det(sig))) * exp(- t(x-mu) %*% solve(sig) %*% (x-mu) / 2)[1];
}

# EM アルゴリズムの E ステップ
Estep <- function(xx, param) {
	K <- nrow(param$mu);
	t(apply(xx, 1, function(x){
		numer <- param$mix * sapply(1:K, function(k) {
			dmnorm(x, param$mu[k,], param$sig[[k]])
		});
		numer / sum(numer);
	}))
}

# EM アルゴリズムの M ステップ
Mstep <- function(xx, gamma_nk) {
	K <- ncol(gamma_nk);
	D <- ncol(xx);
	N <- nrow(xx);

	N_k <- colSums(gamma_nk);
	new_mix <- N_k / N;
	new_mu <- (t(gamma_nk) %*% xx) / N_k;

	new_sig <- list();
	for(k in 1:K) {
		sig <- matrix(numeric(D^2), D);
		for(n in 1:N) {
			x <- xx[n,] - new_mu[k,];
			sig <- sig + gamma_nk[n, k] * (x %*% t(x));
		}
		new_sig[[k]] <- sig / N_k[k]
	}

	list(mu=new_mu, sig=new_sig, mix=new_mix);
}

# 対数尤度関数
Likelihood <- function(xx, param) {
	K <- nrow(param$mu);
	sum(apply(xx, 1, function(x){
		log(sum(param$mix * sapply(1:K, function(k) dmnorm(x, param$mu[k,], param$sig[[k]]))));
	}))
}

OnlineEM <- function(xx, m, param) {
	N <- nrow(xx);
	K <- nrow(param$mu);
	#cat(sprintf("---- %d: (%1.4f, %1.4f)\n", m, xx[m, 1], xx[m, 2]));
	new_gamma <- param$mix * sapply(1:K, function(k) {
		#print(param$mix[k]);
		#print(param$mu[k,]);
		#print(param$sig[[k]]);
		dmnorm(xx[m, ], param$mu[k,], param$sig[[k]]);
	});
	new_gamma <- new_gamma / sum(new_gamma);
	delta <- new_gamma - param$gamma[m,];
	param$gamma[m,] <- new_gamma;

	#cat(sprintf("new_gamma: %1.3f %1.3f\n", new_gamma[1], new_gamma[2]));
	#cat(sprintf("delta: %1.3f %1.3f\n", delta[1], delta[2]));
	param$mix <- param$mix + delta / N;
	N_k <- param$mix * N;
	for(k in 1:K) {
		x <- xx[m,] - param$mu[k,];
		d <- delta[k] / N_k[k];
		param$mu[k,] <- param$mu[k,] + d * x;
		param$sig[[k]] <- (1 - d) * (param$sig[[k]] + d * x %*% t(x));
	}

	param;
}




N <- nrow(xx);
K <- 2;

for (n in 1:10) {
	# 初期値
	param0 <- init_param(K, ncol(xx));

	# normal EM
	timing <- system.time({
		param <- param0;

		# 収束するまで繰り返し
		likeli <- -999999;
		for (j in 1:100) {
			gamma_nk <- Estep(xx, param);
			param <- Mstep(xx, gamma_nk);

			cat(sprintf(" %d: %.3f\n", j, (l <- Likelihood(xx, param))));
			if (l - likeli < 0.001) break;
			likeli <- l;
		}
	});
	cat(sprintf("A%d:convergence=%d, likelihood=%.4f, %1.2fsec\n", n, j, likeli, timing[3]));

	# incremental EM
	timing <- system.time({
		param <- param0;

		# 最初の一周は通常の EM
		gamma_nk <- Estep(xx, param);
		param <- Mstep(xx, gamma_nk);
		param$gamma <- gamma_nk;

		# online EM
		likeli <- -999999;
		for (j in 2:100) {
			randomlist <- sample(1:N);
			for(m in randomlist) param <- OnlineEM(xx, m, param);

			cat(sprintf(" %d: %.3f\n", j, (l <- Likelihood(xx, param))));
			if (l - likeli < 0.001) break;
			likeli <- l;
		}
	});
	cat(sprintf("B%d:convergence=%d, likelihood=%.4f, %1.2fsec\n", n, j, likeli, timing[3]));
}

# plot(xx, col=rgb(gamma_nk[,1],0,gamma_nk[,2]), xlab=paste(sprintf("%1.3f",t(param$mu)),collapse=","), ylab="");
# points(param$mu, pch = 8);

