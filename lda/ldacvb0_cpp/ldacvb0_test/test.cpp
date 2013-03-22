#include <vector>
#include <sstream>
#include "ldacvb0.hpp"
#include "cybozu/test.hpp"

CYBOZU_TEST_AUTO(test_vocabularies_for_std_string)
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
	CYBOZU_TEST_EQUAL(h.size(), 51);
	CYBOZU_TEST_EQUAL(h.count("alice"), 3);
}

/*
CYBOZU_TEST_AUTO(test3)
{
	std::string filename = "c:/works/iir/lda/ali.txt";
	vocabularies<std::string> h;
	//vocabularies<cybozu::String> h;
	loadeachwords(filename, h);
	CYBOZU_TEST_EQUAL(h.size(), 53);
}
*/
