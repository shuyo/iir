# Bayesian PPCA for R

M <- 2;
I <- 50;
splits <- 1;

directory <- ".";

argv <- commandArgs(T);
if (length(argv)>0) directory <- commandArgs(T)[1];
if (length(argv)>1) M <- as.integer(commandArgs(T)[2]);
if (length(argv)>2) I <- as.integer(commandArgs(T)[3]);
if (length(argv)>3) splits <- as.integer(commandArgs(T)[4]);

oilflow <- as.matrix(read.table(sprintf("%s/DataTrn.txt", directory)));
oilflow.labels <- read.table(sprintf("%s/DataTrnLbls.txt", directory));

# density function of multivariate Gaussian
dmnorm <- function(x, mu, sig) {
	D <- length(mu);
	1/((2 * pi)^D * sqrt(det(sig))) * exp(- t(x-mu) %*% solve(sig) %*% (x-mu) / 2)[1];
}

ppca_bayes <- function() {
	D <- ncol(oilflow);
	N <- nrow(oilflow);
	col <- colSums(t(oilflow.labels) * c(4,3,2));
	pch <- colSums(t(oilflow.labels) * c(3,1,4));

	# initialize parameters
	W <- matrix(rnorm(M*D), D);
	sigma2 <- rgamma(1,1);
	alpha <- c(1, rep(10000, M-1));

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
		sigma2 <- sum(xn_minus_x_bar^2) - 2 * sum(diag(t(W) %*% xn_minus_x_bar %*% Ez));
		for(n in 1:N) {
			sigma2 <- sigma2 + sum(diag(Ezz[[n]] %*% t(W) %*% W));
		}
		sigma2 <- sigma2 / N / D;

		# alpha_i = D / w_i^T w_i (PRML 12.62)
		if (i>0) alpha <- D / diag(t(W) %*% W);

		cat(sprintf("M=%d, I=%d, alpha=(%s)\n", M, i, paste(sprintf(" %.2f",alpha),collapse=",")));
		if (sum(alpha>1e6)){
			W <- W[,alpha<1e6];
			alpha <- alpha[alpha<1e6];
			M <- length(alpha);
		}
	}

	# draw chart
	draw_chart <- function(targets) {
		plot(Ez[,targets], col=col, pch=pch, xlim=c(-3,3), ylim=c(-3,3),
				#main=sprintf("M=%d, I=%d, alpha=(%s)\n", M, i, paste(sprintf(" %.2f",alpha), collapse=",")),
				xlab=sprintf("alpha=%.2f", alpha[targets[1]]),
				ylab=sprintf("alpha=%.2f", alpha[targets[2]])
		);
	};
	png(width=640, height=640);
	par(mfrow=c(splits, splits), mar=c(4, 4, 2, 2));
	#for(i in 1:(M-1)) for(j in (i+1):M) draw_chart(c(i, j));
	#for(angle in 10:80) scatterplot3d(Ez[,1], Ez[,2], Ez[,3], color=col, pch=pch, xlim=c(-3,3), ylim=c(-3,3), zlim=c(-3,3), angle=angle*2);
};

ppca_bayes()

#library(scatterplot3d);
#library(animation);
#saveMovie(ppca_bayes(), interval=0.05, moviename="ppca_bayes", movietype="gif", outdir=getwd(),width=480, height=480);

