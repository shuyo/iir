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
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cybozu/mmap.hpp>
#include <cybozu/string.hpp>
#include <cybozu/string_operation.hpp>

namespace cybozu {
namespace ldacvb0 {




/*
	basic data types
*/
typedef std::vector<double> Vec;

typedef std::vector<Vec> Mat;

class Term {
public:
	size_t id;
	int freq;
	Term() : id(0), freq(0) {}
	Term(size_t id_, int freq_) : id(id_), freq(freq_) {}
};

typedef std::vector<Term> Document;


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
	Sampler from Dirichlet Distribution
*/
const double MIN = 0.001;
class dirichlet_distribution {
private:
	std::mt19937 generator;
public:
	dirichlet_distribution() {
		// TODO :
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
			sum += *i = distribution(generator);
		}
		for (i=vec.begin();i!=iend;++i) {
			*i /= sum;
		}
	}
	void draw(Vec& vec, const Vec& alpha) {
		if (vec.size() != alpha.size()) vec.resize(alpha.size());
		double sum = 0;
		auto i = vec.begin(), iend = vec.end();
		for (auto a = alpha.begin();i!=iend;++i,++a) {
			if (*a>MIN) {
				std::gamma_distribution<double> distribution(*a, 1.0);
				sum += *i = distribution(generator);
			} else {
				std::exponential_distribution<double> distribution(1.0 / *a);
				sum += *i = distribution(generator);
			}
		}
		for (i=vec.begin();i!=iend;++i) {
			*i /= sum;
		}
	}
};


/*

*/
const std::regex rexword("(\\S+)/\\S+");

const std::string STOPWORDS_[] = {"the", "of", "and", "at", "so", "a", "i", "one" ,"may", "but", "also", "that", "in", "for", "to", "it", "on", "as", "with", "from", "an", "by", "this", "or", "have", "be", "all", "not",
"no", "any", "which", "had", "was", "been", "only", "such", "are", "these", "them", "other", "two", "is", "take", "has", "they", "some", "will", "its", "into", "when", "there", "his", "more", "than", "he", "who", "would", "up", "out", "were", "first", "time", "what", "made", "then", "can", "about", "over", "their", "if", "even",
"you", "my", "me", "we", "do", "your", "like", "our", "could", "him", "man", "know", "see", "now", "very", "just", "years", "way", "us", "never",
"men", "down", "back", "said", "off", "though", "came", "away", "again", "against", "before", "get", "still", "new", "many",
"good", "how", "much", "each", "most", "make", "long", "little", "should", "where", "under", "both", "here",
"because", "own", "being", "did", "well", "through", "too", "those", "after", "she", "her", "come", "might", "few", "another", "himself", "between", "why", "am", "always", "nothing", "got", "going", "must",
"i'm", "i'll", "don't", "didn't"};

/*

[topic 140]
year	220	701	0.0138434
university	86	224	0.0111295
great	291	666	0.00802565
school	140	496	0.00730239
during	265	585	0.00668247
while	329	680	0.0060898
often	206	369	0.00572025
every	274	491	0.00511144
almost	256	432	0.00509078
set	235	415	0.00508447
same	336	686	0.00502589
special	155	250	0.00466388
fact	233	447	0.00459556
become	209	359	0.00455742
best	227	352	0.00445423
performance	65	122	0.00441623
college	78	269	0.00433846
others	212	323	0.00432374
times	194	300	0.00428213
early	209	366	0.00419036

[topic 171]
right	265	613	0.0119862
go	275	626	0.01026
mr	148	844	0.00906082
something	222	450	0.00867626
wasn't	83	154	0.00779252
think	219	433	0.00754351
went	222	507	0.00749453
want	172	328	0.00666444
she'd	20	68	0.00629034
thought	237	517	0.00615435
wanted	129	226	0.00615119
couldn't	92	175	0.00613035
looked	170	367	0.00602287
sure	168	263	0.00601362
say	242	504	0.00601181
i'd	57	104	0.00597094
yes	89	144	0.00591194
let	222	453	0.00587024
i've	76	125	0.00585789
maybe	74	133	0.00575668
*/

const std::unordered_set<std::string> STOPWORDS(&STOPWORDS_[0], &STOPWORDS_[sizeof(STOPWORDS_)/sizeof(STOPWORDS_[0])]);

/*
identify and count freaquency of vocabulary
*/
template <class WORD>
class Vocabularies {
public:
	std::vector<WORD> vocalist;
	std::unordered_map<WORD, IdCount> voca;

	size_t add(const WORD &word) {
		WORD key(word);
		normalize(key);
		if (STOPWORDS.find(key)!=STOPWORDS.end()) return SIZE_MAX;
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

	size_t size() const {
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
		size_t len = s.size();
		if (len>1 && s.compare(len-1, len, ".")==0) {
			s.resize(len-1);
		} else if (len>2 && s.compare(len-2, len, "'s")==0) {
			s.resize(len-2);
		}

	}
};

/*
doocument loader
*/
template <class STRING, class CHAR>
class Documents : public std::vector<Document> {
public:
	Vocabularies<STRING> vocabularies;
	int N;
	std::unordered_map<size_t, int> docfreq;

private:
	template <class T> void addeachword(std::regex_iterator<T> i) {
		std::unordered_map<size_t, int> count;
		std::regex_iterator<T> iend;
		for (; i != iend; ++i) {
			const std::string& w = (*i)[1].str();
			char c = w[0];
			if (c < 'A' || (c > 'Z' && c < 'a') || c > 'z') continue;
			size_t id = vocabularies.add(w);
			if (id == SIZE_MAX) continue;
			auto x = count.find(id);
			if (x != count.end()) {
				x->second += 1;
			} else {
				count[id] = 1;
			}
		}

		if (count.size()==0) return;

		push_back(Document());
		Document& doc = back();
		auto j = count.begin(), jend = count.end();
		for (;j!=jend;++j) {
			doc.push_back(Term(j->first, j->second));
			N += j->second;
			auto df = docfreq.find(j->first);
			if (df!=docfreq.end()) {
				df->second += 1;
			} else {
				docfreq[j->first] = 1;
			}
		}
	}

public:
	Documents() : N(0) {
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

*/
void parameter_init(Vec& n_wk, Vec& n_jk, Vec& n_k, const Documents<std::string, char>& docs, const int K) {
	const size_t M = docs.size();
	const size_t V = docs.vocabularies.size();
	n_wk.resize(V*K);
	n_jk.resize(M*K);
	n_k.resize(K);
}


inline void update_for_word(
	Vec& gamma_k,
	Vec::iterator i_wk_buf, Vec::iterator i_jk_buf, Vec::iterator i_k_buf,
	Vec::const_iterator i_wk, Vec::const_iterator i_jk, Vec::const_iterator i_k,
	const size_t w, const int freq, const int K
	) {
		i_wk += w * K;
		i_wk_buf += w * K;

		auto iend = gamma_k.end();
		double sum_gamma = 0;
		for (auto i_gamma = gamma_k.begin();i_gamma != iend; ++i_gamma) {
			double gamma = *i_gamma;
			double new_gamma = (*i_wk++ - gamma) * (*i_jk++ - gamma) / (*i_k++ - gamma);
			sum_gamma += *i_gamma = new_gamma;
		}

		for (auto i_gamma = gamma_k.begin();i_gamma != iend; ++i_gamma) {
			double gamma = *i_gamma / sum_gamma;
			*i_gamma = gamma;
			gamma *= freq;
			*i_wk_buf++ += gamma;
			*i_jk_buf++ += gamma;
			*i_k_buf++ += gamma;
		}
}


/*
	LDA model and CVB0 inference
*/
class LDA_CVB0 {
public:
	int K_, V_;
	double alpha_;
	double beta_;
	Vec n_wk1, n_wk2, n_jk1, n_jk2, n_k1, n_k2;
	Vec *n_wk, *n_wk_buf, *n_jk, *n_jk_buf, *n_k, *n_k_buf;
	Mat gamma_jik;
	const Documents<std::string, char>& docs_;
	LDA_CVB0(int K, int V, double alpha, double beta, const Documents<std::string, char>& docs) :
	K_(K), V_(V), alpha_(alpha), beta_(beta), docs_(docs),
		n_wk(&n_wk1), n_wk_buf(&n_wk2), n_jk(&n_jk1), n_jk_buf(&n_jk2), n_k(&n_k1), n_k_buf(&n_k2) {
			parameter_init(n_wk1, n_jk1, n_k1, docs, K);
			parameter_init(n_wk2, n_jk2, n_k2, docs, K);

			std::fill(n_wk->begin(), n_wk->end(), beta_);
			std::fill(n_jk->begin(), n_jk->end(), alpha_);
			std::fill(n_k->begin(), n_k->end(), beta_ * V_);

			cybozu::ldacvb0::dirichlet_distribution dd(1U);

			auto j = docs_.begin(), jend = docs_.end();
			auto j_jk = n_jk->begin();
			Vec aph(K);
			auto aend = aph.end();
			for (;j!=jend;++j) {
				auto i = j->begin(), iend = j->end();
				for (;i!=iend;++i) {
					size_t w = i->id;
					int freq = i->freq;

					double sum = 0;
					{
						auto i_wk = n_wk->begin() + w * K;
						auto i_jk = j_jk;
						auto i_k = n_k->begin();
						for (auto ai = aph.begin();ai!=aend;++ai,++i_wk,++i_jk,++i_k) {
							sum += *ai = *i_wk * *i_jk / *i_k;
						}
					}
					sum = alpha / sum;
					for (auto ai = aph.begin(); ai != aend; ++ai) {
						*ai *= sum;
						//if (*ai<0.00001) *ai = 0.00001;
					}

					gamma_jik.push_back(Vec());
					Vec& gamma = gamma_jik.back();
					dd.draw(gamma, aph);

					auto gi = gamma.begin(), gend = gamma.end();
					auto i_wk = n_wk->begin() + w * K;
					auto i_jk = j_jk;
					auto i_k = n_k->begin();
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
		std::fill(n_wk_buf->begin(), n_wk_buf->end(), beta_);
		std::fill(n_jk_buf->begin(), n_jk_buf->end(), alpha_);
		std::fill(n_k_buf->begin(), n_k_buf->end(), beta_ * V_);

		auto gamma_k = gamma_jik.begin();
		auto j = docs_.begin(), jend = docs_.end();
		auto j_jk = n_jk->begin();
		auto j_jk_buf = n_jk_buf->begin();
		for (;j!=jend;++j) {
			auto i = j->begin(), iend = j->end();
			for (;i!=iend;++i) {

				size_t w = i->id;
				int freq = i->freq;

				update_for_word(
					*gamma_k,
					n_wk_buf->begin(), j_jk_buf, n_k_buf->begin(),
					n_wk->begin(), j_jk, n_k->begin(),
					w, freq, K_
					);

				++gamma_k;
			}
			j_jk += K_;
			j_jk_buf += K_;

		}

		{auto x = n_wk; n_wk = n_wk_buf; n_wk_buf = x;}
		{auto x = n_jk; n_jk = n_jk_buf; n_jk_buf = x;}
		{auto x = n_k; n_k = n_k_buf; n_k_buf = x;}
	}

	void worddist(Vec& dist) const {
		dist.clear();
		auto i = n_wk->begin(), iend = n_wk->end(), jend = n_k->end();
		while(i != iend) {
			for (auto j = n_k->begin();j!=jend;++j,++i) {
				dist.push_back(*i / *j);
			}
		}
	}

	void docdist(Vec& dist) const {
		dist.clear();
		auto i = n_jk->begin(), iend = n_jk->end();
		while(i!=iend) {
			double sum = 0;
			for (int k=0;k<K_;++k) {
				sum += *(i+k);
			}
			for (int k=0;k<K_;++k) {
				dist.push_back(*i++/sum);
			}
		}
	}

	double perplexity () const {
		return perplexity(docs_);
	}

	double perplexity (const Documents<std::string, char>& docs) const {

		Vec phi;
		worddist(phi);

		double loglikelihood = 0;
		auto j = docs.begin(), jend = docs.end();
		auto i_jk = n_jk->begin();
		Vec vec(K_);
		auto vend = vec.end();
		for(;j!=jend;++j) {
			double sum = 0;
			for(int k=0;k<K_;++k) sum += *(i_jk+k);
			for(auto v = vec.begin(); v!=vend; ++v) {
				*v = *i_jk++ / sum;
			}

			auto i = j->begin(), iend = j->end();
			for (;i!=iend;++i) {
				size_t w = i->id;
				int freq = i->freq;

				auto k = phi.begin() + w * K_;
				double prob = 0;
				for(auto v = vec.begin(); v!=vend; ++v, ++k) {
					prob += *v * *k;
				}

				loglikelihood -= log(prob) * freq;
			}

		}

		return exp(loglikelihood / docs.N);
	}
};


} }

#endif
