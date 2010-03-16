# Old Faithful dataset を取得して正規化
data("faithful");
xx <- scale(faithful, apply(faithful, 2, mean), apply(faithful, 2, sd));

# パラメータの初期化(平均、共分散、混合率)
init_param <- function(K, D) {
	sig <- list();
	for(k in 1:K) sig[[k]] <- diag(K);
	list(mu = matrix(rnorm(K * D), D), mix = numeric(K)+1/K, sig = sig);
}

# 多次元正規分布密度関数(パッケージ使えって？)
dmnorm <- function(x, mu, sig) {
	D <- length(mu);
	1/((2 * pi)^D * sqrt(det(sig))) * exp(- t(x-mu) %*% solve(sig) %*% (x-mu) / 2)[1];
}

# EM アルゴリズムの E ステップ
Estep <- function(xx, param) {
	K <- nrow(param$mu);
	t(apply(xx, 1, function(x){
		numer <- sapply(1:K, function(k) {
			param$mix[k] * dmnorm(x, param$mu[k,], param$sig[[k]])
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

Likelihood <- function(xx, param) {
	K <- nrow(param$mu);
	sum(apply(xx, 1, function(x){
		log(sum(param$mix * sapply(1:K, function(k) dmnorm(x, param$mu[k,], param$sig[[k]]))));
	}))
}



for (n in 1:100) {
	timing <- system.time({
		# 初期値
		param <- init_param(2, ncol(xx));

		# 収束するまで繰り返し
		likeli <- -999999;
		for (j in 1:100) {
			gamma_nk <- Estep(xx, param);
			param <- Mstep(xx, gamma_nk);
			l <- Likelihood(xx, param);
			if (l - likeli < 0.0001) break;
			likeli <- l;
		}
	});
	cat(sprintf("%d:convergence=%d, likelihood=%f, %1.2fsec\n", n, j, likeli, timing[3]));
}

# plot(xx, col=rgb(gamma_nk[,1],0,gamma_nk[,2]), xlab=paste(sprintf("%1.3f",t(param$mu)),collapse=","), ylab="");
# points(param$mu, pch = 8);

