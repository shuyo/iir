/**
@file
@brief LDA CVB0

Copyright (C) 2013 Nakatani Shuyo / Cybozu Labs, Inc., all rights reserved.
*/

#ifndef LDACVB0_HPP
#define LDACVB0_HPP

#ifdef __linux__
#include <boost/regex.hpp>
namespace std {
    using boost::regex;
    using boost::match_results;
    using boost::regex_search;
}
#else
#include <regex>
#endif
#include <algorithm>
#include <cassert>
#include <string>
#include <unordered_map>
#include <vector>
#include <cybozu/string.hpp>
#include <cybozu/string_operation.hpp>
#include <cybozu/mmap.hpp>
#include <random>

namespace cybozu {
namespace ldacvb0 {

/*

*/
std::regex rexword("(\\S+)/\\S+");

/*

*/
typedef std::vector<double> Vec;
typedef std::vector<Vec> Mat;

/*
id and counter of vocabulary
*/
class IdCount {
public:
	size_t id;
	int count;
	IdCount() : id(0), count(0) {}
	IdCount(size_t id_, int count_) : id(id_), count(count_) {}
};



/*
identify and count freaquency of vocabulary
*/
template <class WORD>
class Vocabularies {
private:
	std::vector<WORD> vocalist;
	std::unordered_map<WORD, IdCount> voca;

public:
	size_t add(const WORD &word) {
		WORD key(word);
		normalize(key);
		auto x = voca.find(key);
		if (x != voca.end()) {
			x->second.count += 1;
			return x->second.id;
		} else {
			size_t new_id = vocalist.size();
			voca[key] = IdCount(new_id, 1);
			vocalist.push_back(key);
			return new_id;
		}
	}

	const IdCount* idcount(const WORD & word) const {
		WORD key(word);
		normalize(key);
		if (voca.find(key) == voca.end()) return NULL;
		return &voca[key];
	}

	int size() const {
		return vocalist.size();
	}

	size_t id(const WORD &word) const {
		WORD key(word);
		normalize(key);
		auto x = voca.find(key);
		if (voca.find(key) == voca.end()) return SIZE_MAX;
		return x->second.id;
	}

	int count(const WORD &word) const {
		WORD key(word);
		normalize(key);
		auto x = voca.find(key);
		if (x == voca.end()) return 0;
		return x->second.count;
	}



	static void normalize(WORD &s) {
		cybozu::ToLower(s);
	}
};

class Term {
public:
	size_t id;
	int freq;
	Term() : id(0), freq(0) {}
	Term(size_t id_, int freq_) : id(id_), freq(freq_) {}
};
typedef std::vector<Term> Document;

/*
doocument loader
*/
template <class STRING, class CHAR>
class Documents : public std::vector<Document> {
public:
	Vocabularies<STRING> vocabularies;

private:
	template <class T> void addeachword(std::regex_iterator<T> i) {
		std::unordered_map<size_t, int> count;
		std::regex_iterator<T> iend;
		for (; i != iend; ++i) {
			const std::string& w = (*i)[1].str();
			char c = w[0];
			if (c < 'A' || (c > 'Z' && c < 'a') || c > 'z') continue;
			size_t id = vocabularies.add(w);
			auto x = count.find(id);
			if (x != count.end()) {
				x->second += 1;
			} else {
				count[id] = 1;
			}
		}

		push_back(Document());
		Document& doc = back();
		auto j = count.begin(), jend = count.end();
		for (;j!=jend;++j) {
			doc.push_back(Term(j->first, j->second));
		}
	}

public:
	Documents() {
		// TODO : stop words
	}

	bool add(const CHAR* p, const CHAR* end) {
		std::regex_iterator<const CHAR*> i( p, end, rexword );
		addeachword(i);
		return true;
	}

	bool add(const STRING& st) {
		std::regex_iterator<STRING::const_iterator> i( st.begin(), st.end(), rexword );
		addeachword(i);
		return true;
	}


};


/*
call the procedure for each word (mmap)
*/
template <class T, class CHAR>
void eachwords(const CHAR* p, const CHAR* end, T& func) {
};

/*
call the procedure for each word (std::string, cybozu::String)
*/
template <class T, class STRING>
void eachwords(STRING st, T& func) {
	std::regex_token_iterator<STRING::iterator> i( st.begin(), st.end(), rexword ), iend;
	while (i != iend) {
		func(*i++);
	}
};

template <class T> void loadeachwords(std::string filename, T& func) {

	try {
		cybozu::Mmap map(filename);
		const char *p = map.get();
		const char *end = p + map.size();
		eachwords(p, end, func);

	} catch (std::exception& e) {
		printf("%s\n", e.what());
	}
}






/*

*/
void update_for_word(
	Vec& gamma_k,
	Vec::iterator i_wk_buf, Vec::iterator i_jk_buf, Vec::iterator i_k_buf,
	Vec::const_iterator& i_wk, Vec::const_iterator i_jk, Vec::const_iterator i_k,
	const size_t w, const int freq, const int K
	) {
		i_wk += w * K;
		i_wk_buf += w * K;

		auto i_gamma = gamma_k.begin();
		double sum_gamma = 0;
		for (int k=0;k<K;++k) {
			double gamma = *i_gamma;
			double new_gamma = (*i_wk++ - gamma) * (*i_jk++ - gamma) / (*i_k++ - gamma);
			*i_gamma++ = new_gamma;
			sum_gamma += new_gamma;
		}
		i_gamma = gamma_k.begin();
		for (int k=0;k<K;++k) {
			double gamma = *i_gamma / sum_gamma;
			*i_gamma++ = gamma;
			gamma *= freq;
			*i_wk_buf++ += gamma;
			*i_jk_buf++ += gamma;
			*i_k_buf++ += gamma;
		}
}


void parameter_init(Vec& n_wk, Vec& n_jk, Vec& n_k, const Documents<std::string, char>& docs, const int K) {
	const size_t M = docs.size();
	const size_t V = docs.vocabularies.size();
	n_wk.resize(V*K);
	n_jk.resize(M*K);
	n_k.resize(K);

}

class dirichlet_distribution {
private:
	std::mt19937 generator;
public:
	dirichlet_distribution() {

	}

	dirichlet_distribution(unsigned long seed) {
		if (seed>0) {
			generator.seed(seed);
		} else {
			dirichlet_distribution();
		}
	}
	void draw(Vec& vec, const int K, const double alpha) {
		std::gamma_distribution<double> distribution(alpha, 1.0);
		if (vec.size() != K) vec.resize(K);
		double sum = 0;
		auto i = vec.begin(), iend = vec.end();
		for (;i!=iend;++i) {
			double x = distribution(generator);
			sum += x;
			*i = x;
		}
		for (i=vec.begin();i!=iend;++i) {
			*i /= sum;
		}
	}
	void draw(Vec& vec, const Vec& alpha) {
		if (vec.size() != alpha.size()) vec.resize(alpha.size());
		double sum = 0;
		auto i = vec.begin(), iend = vec.end();
		auto a = alpha.begin();
		for (;i!=iend;++i,++a) {
			std::gamma_distribution<double> distribution(*a, 1.0);
			double x = distribution(generator);
			sum += x;
			*i = x;
		}
		for (i=vec.begin();i!=iend;++i) {
			*i /= sum;
		}
	}
};

/*

*/
class LDA_CVB0 {
public:
	int K_, V_;
	double alpha_;
	double beta_;
	Vec n_wk1, n_wk2, n_jk1, n_jk2, n_k1, n_k2;
	Vec &n_wk, &n_wk_buf, &n_jk, &n_jk_buf, &n_k, &n_k_buf;
	Mat gamma_jik;
	const Documents<std::string, char>& docs_;
	LDA_CVB0(int K, int V, double alpha, double beta, const Documents<std::string, char>& docs) :
	K_(K), V_(V), alpha_(alpha), beta_(beta), docs_(docs),
		n_wk(n_wk1), n_wk_buf(n_wk2), n_jk(n_jk1), n_jk_buf(n_jk2), n_k(n_k1), n_k_buf(n_k2) {
			parameter_init(n_wk1, n_jk1, n_k1, docs, K);
			parameter_init(n_wk2, n_jk2, n_k2, docs, K);

			std::fill(n_wk.begin(), n_wk.end(), beta_);
			std::fill(n_jk.begin(), n_jk.end(), alpha_);
			std::fill(n_k.begin(), n_k.end(), beta_ * V_);

			cybozu::ldacvb0::dirichlet_distribution dd(1U);

			auto j = docs_.begin(), jend = docs_.end();
			auto j_jk = n_jk.begin();
			for (;j!=jend;++j) {
				auto i = j->begin(), iend = j->end();
				for (;i!=iend;++i) {
					size_t w = i->id;
					int freq = i->freq;

					Vec aph(K);
					auto aend = aph.end();
					double sum = 0;
					{
						auto i_wk = n_wk.begin() + w * K;
						auto i_jk = j_jk;
						auto i_k = n_k.begin();
						for (auto ai = aph.begin();ai!=aend;++ai,++i_wk,++i_jk,++i_k) {
							sum += *ai = *i_wk * *i_jk / *i_k;
						}
					}
					sum = alpha / sum;
					for (auto ai = aph.begin(); ai != aend; ++ai) *ai *= sum;

					gamma_jik.push_back(Vec());
					Vec& gamma = gamma_jik.back();
					dd.draw(gamma, aph);

					auto gi = gamma.begin(), gend = gamma.end();
					auto i_wk = n_wk.begin() + w * K;
					auto i_jk = j_jk;
					auto i_k = n_k.begin();
					for (;gi!=gend;++gi,++i_wk,++i_jk,++i_k) {
						double g = *gi * freq;
						*i_wk += g;
						*i_jk += g;
						*i_k += g;
					}
				}
				j_jk+=K;
			}
	}

	void learn() {
		std::fill(n_wk_buf.begin(), n_wk_buf.end(), beta_);
		std::fill(n_jk_buf.begin(), n_jk_buf.end(), alpha_);
		std::fill(n_k_buf.begin(), n_k_buf.end(), beta_ * V_);

		auto gamma_k = gamma_jik.begin();
		auto j = docs_.begin(), jend = docs_.end();
		auto j_jk = n_jk.begin();
		auto j_jk_buf = n_jk_buf.begin();
		for (;j!=jend;++j) {
			auto i = j->begin(), iend = j->end();
			for (;i!=iend;++i) {

				size_t w = i->id;
				int freq = i->freq;

				update_for_word(
					*gamma_k,
					n_wk.begin(), j_jk, n_k.begin(),
					n_wk_buf.begin(), j_jk_buf, n_k_buf.begin(),
					w, freq, K_
					);

				++gamma_k;
			}
			j_jk += K_;
			j_jk_buf += K_;

		}



		n_wk, n_wk_buf = n_wk_buf, n_wk;
		n_jk, n_jk_buf = n_jk_buf, n_jk;
		n_k, n_k_buf = n_k_buf, n_k;
	}
};

} }

#endif
