#include <vector>
#include <sstream>
#include "ldacvb0.hpp"
#include "cybozu/test.hpp"

CYBOZU_TEST_AUTO(test_vocabularies_for_std_string)
{
	std::string st =
"	The/at Fulton/np-tl County/nn-tl Grand/jj-tl Jury/nn-tl said/vbd Friday/nr an/at investigation/nn of/in Atlanta's/np$ recent/jj "
"primary/nn election/nn produced/vbd ``/`` no/at evidence/nn ''/'' that/cs any/dti irregularities/nns took/vbd place/nn ./.\n"
"\n"
"	The/at jury/nn further/rbr said/vbd in/in term-end/nn presentments/nns that/cs the/at City/nn-tl Executive/jj-tl Committee/nn-tl ,/, "
"which/wdt had/hvd over-all/jj charge/nn of/in the/at election/nn ,/, ``/`` deserves/vbz the/at praise/nn and/cc thanks/nns of/in the/at "
"City/nn-tl of/in-tl Atlanta/np-tl ''/'' for/in the/at manner/nn in/in which/wdt the/at election/nn was/bedz conducted/vbn ./.";
	cybozu::ldacvb0::Documents<std::string, char> d;
	d.add(st);
	cybozu::ldacvb0::Vocabularies<std::string>& v = d.vocabularies;
	CYBOZU_TEST_EQUAL(v.size(), 42);
	CYBOZU_TEST_EQUAL(v.count("the"), 8);
	CYBOZU_TEST_EQUAL(v.count("tHE"), 8);
	CYBOZU_TEST_EQUAL(v.count("in"), 2);
	CYBOZU_TEST_EQUAL(v.count("grand"), 1);

	CYBOZU_TEST_EQUAL(v.count("the/at"), 0);
	CYBOZU_TEST_EQUAL(v.count("``"), 0);
	CYBOZU_TEST_EQUAL(v.count(","), 0);
	CYBOZU_TEST_EQUAL(v.count("."), 0);
	CYBOZU_TEST_EQUAL(v.count("np"), 0);
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
