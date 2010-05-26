# Bayesian PPCA for R

M <- 2;
I <- 200;
directory <- ".";

argv <- commandArgs(T);
if (length(argv)>0) directory <- commandArgs(T)[1];
if (length(argv)>1) M <- as.integer(commandArgs(T)[2]);
if (length(argv)>2) I <- as.integer(commandArgs(T)[3]);

oilflow <- as.matrix(read.table(sprintf("%s/DataTrn.txt", directory)));
oilflow.labels <- read.table(sprintf("%s/DataTrnLbls.txt", directory));
likelihood.pre <- -999999;

ppca_bayes <- function(oilflow, oilflow.labels, M, I) {
	D <- ncol(oilflow);
	N <- nrow(oilflow);
	col <- colSums(t(oilflow.labels) * c(4,3,2));
	pch <- colSums(t(oilflow.labels) * c(3,1,4));

	# initialize parameters
	W <- matrix(rnorm(M*D), D);
	sigma2 <- rnorm(1);
	alpha <- rep(1, M);

	# mu = mean x_bar
	mu <- colMeans(oilflow);
	xn_minus_x_bar <- t(oilflow) - mu;  # DxN-matrix
	S <- var(oilflow);

	# iteration
	for(i in 0:I) {
		# M = W^T W + sigma^2 I (PRML 12.41)
		M_inv <- solve(t(W) %*% W + sigma2 * diag(M));

		### E-step:

		# E[z_n] = M^-1 W^T (x_n - x^bar) (PRML 12.54)
		Ez <- t(M_inv %*% t(W) %*% xn_minus_x_bar);

		# E[z_n z_n^T] = sigma^2 M^-1 + E[z_n]E[z_n]^T (PRML 12.55)
		Ezz <- list();
		sum_Ezz <- matrix(numeric(M*M), M);
		for(n in 1:N) {
			ezz <- sigma2 * M_inv + Ez[n,] %*% t(Ez[n,]);
			Ezz[[n]] <- ezz;
			sum_Ezz <- sum_Ezz + ezz;
		}

		### M-step:

		# W_new = {sum (x_n - x^bar)E[z_n]^T}{sum E[z_n z_n^T] + sigma^2 A}^-1 (PRML 12.63)
		W <- xn_minus_x_bar %*% Ez %*% solve(sum_Ezz + diag(sigma2 * alpha));

		# sigma_new^2 = 1/ND sum{ |x_n-x^bar|^2 - 2E[z_n]^T W^T (x_n-x^bar) + Tr(E[z_n z_n^T] W^T W) } (PRML 12.57)
		sigma2 <- sum(xn_minus_x_bar^2) - 2 * sum(diag(Ez %*% t(W) %*% xn_minus_x_bar));
		for(n in 1:N) {
			sigma2 <- sigma2 + sum(diag(Ezz[[n]] %*% t(W) %*% W));
		}
		sigma2 <- sigma2 / N / D;

		# alpha_i = D / w_i^T w_i (PRML 12.62)
		alpha <- D / diag(t(W) %*% W);

		cat(sprintf("I=%d, alpha=(%s)\n", i, paste(sprintf(" %.2f",alpha),collapse=",")));
	}

	png();
	plot(Ez[,order(alpha, decreasing=T)[1:2]], col=col, pch=pch, xlim=c(-3,3),ylim=c(-3,3),ylab="",
			xlab=sprintf("M=%d, I=%d, alpha=(%s)\n", M, i, paste(sprintf(" %.2f",alpha),collapse=",")));
};

ppca_bayes(oilflow, oilflow.labels, M, I);

