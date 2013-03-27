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
#include <string>
#include <unordered_map>
#include <cybozu/string.hpp>
#include <cybozu/string_operation.hpp>
#include <cybozu/mmap.hpp>

namespace cybozu {
namespace ldacvb0 {

/*

*/
std::regex rexword("(\\S+)/\\S+");

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
		if (voca.find(key) != voca.end()) {
			IdCount& x = voca[key];
			x.count += 1;
			return x.id;
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

typedef std::vector<size_t> Document;

/*
	doocument loader
*/
template <class STRING, class CHAR>
class Documents : std::vector<Document> {
public:
	Vocabularies<STRING> vocabularies;

private:
	template <class T> void addeachword(std::regex_iterator<T> i) {
		push_back(Document());
		Document& doc = back();

		std::regex_iterator<T> iend;
		for (; i != iend; ++i) {
			const std::string& w = (*i)[1].str();
			char c = w[0];
			if (c < 'A' || (c > 'Z' && c < 'a') || c > 'z') continue;
			size_t id = vocabularies.add(w);
			doc.push_back(id);
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

class LDA_CVB0 {
private:
	int K_;
	double alpha_;
	double beta_;
public:
	LDA_CVB0(int K, double alpha, double beta) :
	  K_(K), alpha_(alpha), beta_(beta) {

	  }
};

} }

#endif
