#include <vector>
#include <sstream>
#include "ldacvb0.hpp"
#include "cybozu/test.hpp"

/*
void split(const std::string &str, const std::string &delim, vocabularies& func) {
	size_t current = 0, found, delimlen = delim.size();
	while((found = str.find(delim, current)) != std::string::npos){
		std::string word = std::string(str, current, found - current);
		func(word);
		std::cout << func.size() << std::endl;
		current = found + delimlen;
	}
	std::string word = std::string(str, current, str.size() - current);
	func(word);
		std::cout << func.size() << std::endl;
}

CYBOZU_TEST_AUTO(test)
{
	CYBOZU_TEST_ASSERT(true);

	auto s = "a b   c d\te\nf";

	std::istringstream is(s);
	while(is) {
		std::string word;
		is >> word;
		if (word.length() > 0) {
		std::cout << "'" << word << "'" << std::endl;
		}

	}

}
*/

	/*
	vocabularies h;
	split(s, " ", h);
	std::cout << h.x.size() << " " << h.size() << std::endl;
	CYBOZU_TEST_EQUAL(h.size(), 4);
	*/


CYBOZU_TEST_AUTO(test2)
{
	std::string st =
"ALICE'S ADVENTURES IN WONDERLAND\n""\n"
"\n"
"Lewis Carroll\n""\n"
"\n"
"\n"
"CHAPTER I. Down the Rabbit-Hole\n"
"\n"
"Alice was beginning to get very tired of sitting by her sister on the\n"
"bank, and of having nothing to do: once or twice she had peeped into the\n"
"book her sister was reading, but it had no pictures or conversations in\n"
"it, 'and what is the use of a book,' thought Alice 'without pictures or\n"
"conversation?'\n";
	vocabularies<std::string> h;
	eachwords(st, h);
	CYBOZU_TEST_EQUAL(h.size(), 53);

}

CYBOZU_TEST_AUTO(test3)
{
	std::string filename = "c:/works/iir/lda/ali.txt";
	vocabularies<std::string> h;
	loadeachwords(filename, h);
	CYBOZU_TEST_EQUAL(h.size(), 53);

}
