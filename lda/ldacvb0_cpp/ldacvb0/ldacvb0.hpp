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
#include <cybozu/string.hpp>
#include <cybozu/string_operation.hpp>

namespace cybozu {
namespace ldacvb0 {


/*
	unordered_map increment for key (utility function)
*/
template <class S, class T>
inline void inc(std::unordered_map<S, T> &map, const S &key) {
	auto x = map.find(key);
	if (x != map.end()) {
		x->second += 1;
	} else {
		map[key] = 1;
	}
}


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


const double ALPHA_THRESHOLD = 0.001;

/*
	Sampler from Dirichlet Distribution
*/
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

	/*
		draw from symmetric dirichlet distribution
	*/
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

	/*
		draw from asymmetric dirichlet distribution
	*/
	void draw(Vec& vec, const Vec& alpha) {
		if (vec.size() != alpha.size()) vec.resize(alpha.size());
		double sum = 0;
		auto i = vec.begin(), iend = vec.end();
		for (auto a = alpha.begin();i!=iend;++i,++a) {
			if (*a > ALPHA_THRESHOLD) {
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
const std::string STOPWORDS_[] = {"the", "of", "and", "at", "so", "a", "i", "one" ,"may", "but", "also", "that", "in", "for", "to", "it", "on", "as", "with", "from", "an", "by", "this", "or", "have", "be", "all", "not",
"no", "any", "which", "had", "was", "been", "only", "such", "are", "these", "them", "other", "two", "is", "take", "has", "they", "some", "will", "its", "into", "when", "there", "his", "more", "than", "he", "who", "would", "up", "out", "were", "first", "time", "what", "made", "then", "can", "about", "over", "their", "if", "even",
"you", "my", "me", "we", "do", "your", "like", "our", "could", "him", "man", "know", "see", "now", "very", "just", "years", "way", "us", "never",
"men", "down", "back", "said", "off", "though", "came", "away", "again", "against", "before", "get", "still", "new", "many",
"good", "how", "much", "each", "most", "make", "long", "little", "should", "where", "under", "both", "here", "however", "since", "without", "enough", "because", "does",
"because", "own", "being", "did", "well", "through", "too", "those", "after", "she", "her", "come", "might", "few", "another", "himself", "between", "why", "am", "always", "nothing", "got", "going", "must",
"during", "while", "often", "every", "almost", "same", "become", "best", "others", "something", "set", "go", "think", "went", "say",
"three", "around",
"i'm", "i'll", "don't", "didn't", "wasn't", "she'd", "couldn't", "i'd", "i've"};

const std::unordered_set<std::string> STOPWORDS(&STOPWORDS_[0], &STOPWORDS_[sizeof(STOPWORDS_)/sizeof(STOPWORDS_[0])]);

/*
identify and count freaquency of vocabulary
*/
template <class WORD>
class Vocabularies {
	bool uses_stopwords;
public:
	std::vector<WORD> vocalist;
	std::unordered_map<WORD, IdCount> voca;

	Vocabularies() : uses_stopwords(true) {}
	Vocabularies(bool excludes_stopwords) : uses_stopwords(!excludes_stopwords) {}

	size_t add(const WORD &word) {
		WORD key(word);
		normalize(key);
		if (uses_stopwords && STOPWORDS.find(key)!=STOPWORDS.end()) return SIZE_MAX;
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
		} else if (len>1 && s.compare(len-1, len, ",")==0) {
			s.resize(len-1);
		} else if (len>2 && s.compare(len-2, len, "'s")==0) {
			s.resize(len-2);
		}

	}
};

const std::regex REXWORD("(\\S+)");
const std::regex REXWORD_WITH_POS("(\\S+)/\\S+");

/*
doocument loader
*/
template <class STRING, class CHAR>
class Documents : public std::vector<Document> {
public:
	Vocabularies<STRING> vocabularies;
	int N;
	std::unordered_map<size_t, size_t> docfreq;

private:
	const std::regex &rexword;

	template <class T> void addeachword(std::regex_iterator<T> i) {
		std::unordered_map<size_t, int> count;
		std::regex_iterator<T> iend;
		for (; i != iend; ++i) {
			const std::string& w = (*i)[1].str();
			if (w.length() <= 2) continue;
			char c = w[0];
			if (c < 'A' || (c > 'Z' && c < 'a') || c > 'z') continue;
			size_t id = vocabularies.add(w);
			if (id == SIZE_MAX) continue;
			inc(count, id);
		}

		if (count.size()==0) return;

		push_back(Document());
		Document& doc = back();
		auto j = count.begin(), jend = count.end();
		for (;j!=jend;++j) {
			doc.push_back(Term(j->first, j->second));
			N += j->second;
			inc(docfreq, j->first);
		}
	}

public:
	Documents() : N(0), rexword(REXWORD) {
	}
	Documents(const std::regex &r) : N(0), rexword(r) {
	}
	Documents(const std::regex &r, bool excludes_stopwords) : N(0), rexword(r), vocabularies(excludes_stopwords) {
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



template <class STRING, class CHAR>
void truncDocFreq(Documents<STRING, CHAR> &docs, const Documents<STRING, CHAR> &orgdocs, size_t ldf, size_t udf) {
	std::unordered_map<size_t, size_t> conv;
	for (auto i=orgdocs.docfreq.begin(), iend=orgdocs.docfreq.end();i!=iend;++i) {
		size_t df = i->second;
		if (df <= ldf || df >= udf) continue;

		size_t oldid = i->first;
		size_t newid = docs.vocabularies.vocalist.size();
		const std::string &w = orgdocs.vocabularies.vocalist[oldid];
		int c = orgdocs.vocabularies.voca.at(w).count;

		conv[oldid] = newid;
		docs.vocabularies.vocalist.push_back(w);
		docs.vocabularies.voca[w] = IdCount(newid, c);
		docs.N += c;
		docs.docfreq[newid] = df;
	}

	for (auto j=orgdocs.begin(), jend=orgdocs.end();j!=jend;++j) {
		docs.push_back(Document());
		Document &doc = docs.back();
		for (auto i=j->begin(), iend=j->end();i!=iend;++i) {
			auto x = conv.find(i->id);
			if (x!=conv.end()) {
				doc.push_back(Term(conv.at(i->id), i->freq));
			}
		}
	}
}



/*

*/
void parameter_init(Vec& n_wk, Vec& n_jk, Vec& n_k, const Documents<std::string, char>& docs, const size_t K) {
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
	const size_t w, const int freq, const size_t K
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
private:
	mutable Vec phi;	// for worddist at perplecity calcuration
public:
	size_t K_, V_;
	double alpha_;
	double beta_;
	Vec n_wk, n_wk_buf, n_jk, n_jk_buf, n_k, n_k_buf;
	Mat gamma_jik;
	const Documents<std::string, char>& docs_;
	LDA_CVB0(size_t K, size_t V, double alpha, double beta, const Documents<std::string, char>& docs) :
	K_(K), V_(V), alpha_(alpha), beta_(beta), docs_(docs) {
			parameter_init(n_wk, n_jk, n_k, docs, K);
			parameter_init(n_wk_buf, n_jk_buf, n_k_buf, docs, K);

			std::fill(n_wk.begin(), n_wk.end(), beta_);
			std::fill(n_jk.begin(), n_jk.end(), alpha_);
			std::fill(n_k.begin(), n_k.end(), beta_ * V_);

			cybozu::ldacvb0::dirichlet_distribution dd(1U);

			Vec aph(K);
			auto aend = aph.end();
			auto j_jk = n_jk.begin();
			for (auto j = docs_.begin(), jend = docs_.end(); j != jend; ++j) {
				for (auto i = j->begin(), iend = j->end();i!=iend;++i) {
					size_t w = i->id;
					int freq = i->freq;

					double sum = 0;
					for (Vec::iterator ai = aph.begin(), i_wk = n_wk.begin() + w * K, i_jk = j_jk, i_k = n_k.begin();
							ai != aend; ++ai, ++i_wk, ++i_jk, ++i_k) {
						sum += *ai = *i_wk * *i_jk / *i_k;
					}
					sum = alpha / sum;
					for (auto ai = aph.begin(); ai != aend; ++ai) {
						*ai *= sum;
					}

					gamma_jik.push_back(Vec());
					Vec& gamma = gamma_jik.back();
					dd.draw(gamma, aph);

					for (Vec::iterator gi = gamma.begin(), gend = gamma.end(),
							i_wk = n_wk.begin() + w * K, i_jk = j_jk, i_k = n_k.begin();
							gi != gend; ++gi, ++i_wk, ++i_jk, ++i_k) {
						double g = *gi * freq;
						*i_wk += g;
						*i_jk += g;
						*i_k += g;
					}
				}
				j_jk += K;
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
					n_wk_buf.begin(), j_jk_buf, n_k_buf.begin(),
					n_wk.begin(), j_jk, n_k.begin(),
					w, freq, K_
					);

				++gamma_k;
			}
			j_jk += K_;
			j_jk_buf += K_;

		}

		n_wk.swap(n_wk_buf);
		n_jk.swap(n_jk_buf);
		n_k.swap(n_k_buf);
	}

	void worddist(Vec& dist) const {
		if (dist.size() != V_ * K_) {
			dist.resize(V_ * K_);
		}
		auto i = n_wk.begin(), iend = n_wk.end(), jend = n_k.end();
		auto d = dist.begin();
		while(i != iend) {
			for (auto j = n_k.begin();j!=jend;++j,++i,++d) {
				*d = *i / *j;
			}
		}
	}

	void docdist(Vec& dist) const {
		dist.clear();
		auto i = n_jk.begin(), iend = n_jk.end();
		while(i!=iend) {
			double sum = 0;
			for (size_t k=0;k<K_;++k) {
				sum += *(i+k);
			}
			for (size_t k=0;k<K_;++k) {
				dist.push_back(*i++/sum);
			}
		}
	}

	double perplexity () const {
		return perplexity(docs_);
	}

	double perplexity (const Documents<std::string, char>& docs) const {
		worddist(phi);

		double loglikelihood = 0;
		auto j = docs.begin(), jend = docs.end();
		auto i_jk = n_jk.begin();
		Vec vec(K_);
		auto vend = vec.end();
		for(;j!=jend;++j) {
			double sum = 0;
			for(size_t k=0;k<K_;++k) sum += *(i_jk+k);
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
