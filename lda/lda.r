
# Latent Dirichlet Allocation + collapsed Gibbs sampling
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

K <- 50;
I <- 200;

filename <- "../data/gift_of_magi.txt";
argv <- commandArgs(T);
if (length(argv)>0) filename <- commandArgs(T)[1];
text <- tolower(readLines(filename));
corpus <- strsplit(text, split="[[:blank:][:punct:]]", perl=T);

words <- c();
words_id <- list();
docs <- list();
M <- 0;
for(line in corpus) {
	doc <- c();
	for (term in line) {
		if (term == "") next;
		if (is.null(words_id[[term]])) {
			words <- append(words, term);
			words_id[[term]] <- length(words);
		}
		doc <- append(doc, words_id[[term]]);
	}
	if (length(doc)==0) next;
	M <- M + 1;
	docs[[M]] <- doc;
}
V <- length(words);

z_m_n <- list();   # M * N_m
n_m_z <- matrix(numeric(M*K),M);
n_z_t <- matrix(numeric(K*V),K);
n_z <- numeric(K);
n_terms <- 0;

for(m in 1:M) {
	doc <- docs[[m]];
	N_m <- length(doc);

	z_n <- sample(1:K, N_m, replace=T);
	z_m_n[[m]] <- z_n;
	for(n in 1:N_m) {
		z <- z_n[n];
		t <- doc[n];
		n_m_z[m,z] <- n_m_z[m,z] + 1;
		n_z_t[z,t] <- n_z_t[z,t] + 1;
		n_z[z] <- n_z[z] + 1;
	}
	n_terms <- n_terms + N_m;
}

alpha <- 0.001;
beta <- 0.001;

for(ita in 1:I) {
	#print("-------------------------------------------------------------------");
	#print(ita);
	
	changes <- 0;
	for(m in 1:M) {
		doc <- docs[[m]];
		N_m <- length(doc);
		for(n in 1:N_m) {
			t <- doc[n];
			z <- z_m_n[[m]][n]; # z_i

			# z_{-i} の状況を作る
			n_m_z[m,z] <- n_m_z[m,z] - 1;
			n_z_t[z,t] <- n_z_t[z,t] - 1;
			n_z[z] <- n_z[z] - 1;

			# p(z|z_{-i}) からサンプリング
			denom_a <- sum(n_m_z[m,]) + K * alpha;
			denom_b <- rowSums(n_z_t) + V * beta;
			p_z <- (n_z_t[,t] + beta) / denom_b * (n_m_z[m,] + alpha) / denom_a;
			z_i <- sample(1:K, 1, prob=p_z);

			z_m_n[[m]][n] <- z_i;
			#print(p_z);
			#cat(sprintf("%d,%d: %d => %d\n", m, n, z, z_i));
			if (z != z_i) changes <- changes + 1;

			n_m_z[m,z_i] <- n_m_z[m,z_i] + 1;
			n_z_t[z_i,t] <- n_z_t[z_i,t] + 1;
			n_z[z_i] <- n_z[z_i] + 1;
		}
	}
	cat(sprintf("%d: %d/%d\n", ita, changes, n_terms));
}

phi <- matrix(numeric(K*V), K);
theta <- matrix(numeric(M*K), M);
for(m in 1:M) {
	theta_m <- n_m_z[m,] + alpha;
	theta[m,] <- theta_m / sum(theta_m);
}
for(z in 1:K) {
	phi_z <- n_z_t[z,] + beta;
	phi[z,] <- phi_z / sum(phi_z);
}
colnames(phi) <- words;

options(digits=5, scipen=1, width=100);
sink(format(Sys.time(), "lda%m%d%H%M.txt"));

for(m in 1:M) {
	doc <- docs[[m]];
	N_m <- length(doc);
	cat(sprintf("\n[corpus %d]-------------------------------------\n", m));
	print(theta[m,]);
	for(n in 1:N_m) {
		cat(sprintf("%s : %d\n", words[[doc[n]]], z_m_n[[m]][n]));
	}
}

print(phi);
sink();

