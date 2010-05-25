# Probability Principal Component Analysis for R

M <- 2;
directory <- ".";

argv <- commandArgs(T);
if (length(argv)>0) directory <- commandArgs(T)[1];
if (length(argv)>1) M <- as.integer(commandArgs(T)[2]);

oilflow <- as.matrix(read.table(sprintf("%s/DataTrn.txt", directory)));
oilflow.labels <- read.table(sprintf("%s/DataTrnLbls.txt", directory));
D <- ncol(oilflow);

# mu = mean x_bar
mu <- colMeans(oilflow);

# eigenvalues and eigenvectors of covariance S
e <- eigen(var(oilflow))

# sigma^2 = sum(rest of eigenvalues) / (D - M)
sigma2 <- mean(e$values[-(1:M)]);

# W_ML = U_M(L_M - sigma^2 I)R, (now R = I)
W_ML <- e$vectors[,1:M] %*% diag(e$values[1:M] - sigma2) %*% diag(c(1,-1))

# M = W^T W + sigma^2 I
M_inv <- solve(t(W_ML) %*% W_ML + sigma2 * diag(M));

# projection into principal subspace
z <- t(M_inv %*% t(W_ML) %*% (t(oilflow) - mu))

# draw chart
col <- colSums(t(oilflow.labels) * c(4,3,2));  # ラベルごとに色を指定
pch <- colSums(t(oilflow.labels) * c(3,1,4));  # ラベルごとにマーカーを指定
plot(z, col=col, pch=pch, xlim=c(-2,4),ylim=c(-4,2))

