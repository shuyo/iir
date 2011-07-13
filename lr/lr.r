# Multi-class Logistic Regression + Stochastic Gradient Descent for R
# (c)2011 Nakatani Shuyo / Cybozu Labs, Inc.
# This code is available under the MIT Licence.

# コマンドライン処理
commandline <- commandArgs(TRUE)
chart <- "--chart" %in% commandline
i <- match("-i", commandline)
if (is.na(i)) {
	I <- 1
} else {
	I <- as.numeric(commandline[i + 1])
}

# iris dataset
xlist <- scale(iris[1:4])
tlist <- cbind(
	ifelse(iris[5]=="setosa",1,0),
	ifelse(iris[5]=="versicolor",1,0),
	ifelse(iris[5]=="virginica",1,0)
)
N <- nrow(xlist) # データ件数

# 事後確率
y <- function(phi, w) {
	y <- c(phi %*% w)
	y <- exp(y - max(y))  # exp の中身から、その最大値を引く(オーバーフロー対策)
	return(y / sum(y))
}

# 誤差関数＆勾配
En <- function(phi, t, w) -log(sum(y(phi, w) * t))
dEn <- function(phi, t, w) outer(phi, y(phi, w) - t)

inference <- function(title, xlist, tlist, phi) {
	PHI <- t(apply(xlist, 1, phi))  # NxM - design matrix
	M <- ncol(PHI)  # 特徴数(特徴空間の次元)
	K <- ncol(tlist) # クラス数

	for (i in 1:I) {
		# 重み初期化
		w <- matrix(rnorm(M * K), M)

		eta <- 0.1  # 学習率
		while (eta > 0.0001) {
			for(n in sample(N)) {
				w <- w - eta * dEn(PHI[n,], tlist[n,], w)  # 確率的勾配降下法
			}
			eta <- eta * 0.95
		}

		ylist <- t(apply(PHI, 1, function(phi) y(phi, w)))
		error <- sum(sapply(1:nrow(PHI), function(n) En(PHI[n,], tlist[n,], w)))
		cat(sprintf("%s: error=%.3f", title, error), "\n")

		# 可視化
		if (chart) {
			pairs(xlist, col=rgb(ylist), main=title)
			plot(xlist[,c(1,2)],
				col=rgb(ylist),
				pch=(tlist %*% c(17,16,22)),
				main=title,
				sub=sprintf("Negative Log Likelihood = %.3f", error)
			)
		}
	}

	return(w)
}

if (chart) png(width=640, height=640)

# 線形特徴関数
phi <- function(x) c(1, x[1], x[2], x[3], x[4])
w <- inference("Linear Features", xlist, tlist, phi)

# 二次特徴関数
phi <- function(x) c(1, x[1], x[2], x[3], x[4],
	x[1]*x[1], x[1]*x[2], x[1]*x[3], x[1]*x[4], x[2]*x[2],
	x[2]*x[3], x[2]*x[4], x[3]*x[3], x[3]*x[4], x[4]*x[4])
w <- inference("Quadratic Features", xlist, tlist, phi)

# RBF 特徴関数
for (s in 1:10) {
	phi <- function(x) {
		c <- seq(-2.5,2.5,by=1)
		d <- outer(c,x,"-")^2
		return(exp(-c(0, outer(c(outer(c(outer(d[,1],d[,2],"+")),d[,3],"+")),d[,4],"+"))/s))
	}
	w <- inference(sprintf("RBF Features (s=%d)", s), xlist, tlist, phi)
}

if (chart) dev.off()


