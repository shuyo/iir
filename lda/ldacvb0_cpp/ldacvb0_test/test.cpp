#include <vector>
#include <sstream>
#include "ldacvb0.hpp"
#include "cybozu/test.hpp"

using cybozu::ldacvb0::Vec;
using cybozu::ldacvb0::Mat;
using cybozu::ldacvb0::Term;
using cybozu::ldacvb0::Document;
using cybozu::ldacvb0::Documents;

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

CYBOZU_TEST_AUTO(test_learn_update_parameter_for_word)
{

	const int K = 3, V = 3;

	double GAMMA0[] = {0.2, 0.3, 0.5},
		GAMMA2[] = {0.9, 0.05, 0.05};
	double NWK[] = {1.1, 2.1, 3.1,  1.1, 1.1, 1.1,  2.1, 0.1, 0.1},
		NJK[] = {4.2, 3.2, 4.2},
		NK[] = {4.3, 3.3, 4.3};

	Vec gamma0(&GAMMA0[0], &GAMMA0[K]);
	Vec gamma2(&GAMMA2[0], &GAMMA2[K]);
	Vec n_wk(&NWK[0], &NWK[V * K]), n_jk(&NJK[0], &NJK[K]), n_k(&NK[0], &NK[K]);
	Vec n_wk_buf(V * K, 0.1), n_jk_buf(K, 0.2), n_k_buf(K, 0.3);


	cybozu::ldacvb0::update_for_word(
		gamma0,
		n_wk_buf.begin(), n_jk_buf.begin(), n_k_buf.begin(),
		n_wk.begin(), n_jk.begin(), n_k.begin(),
		0, 1, K
		);

	CYBOZU_TEST_NEAR(gamma0[0], 0.17050723, 1e-6);
	CYBOZU_TEST_NEAR(gamma0[1], 0.3378885 , 1e-6);
	CYBOZU_TEST_NEAR(gamma0[2], 0.49160426, 1e-6);

	CYBOZU_TEST_NEAR(n_wk_buf[0], 0.17050723 * 1 + 0.1, 1e-6);
	CYBOZU_TEST_NEAR(n_wk_buf[1], 0.3378885  * 1 + 0.1, 1e-6);
	CYBOZU_TEST_NEAR(n_wk_buf[2], 0.49160426 * 1 + 0.1, 1e-6);

	CYBOZU_TEST_NEAR(n_jk_buf[0], 0.17050723 * 1 + 0.2, 1e-6);
	CYBOZU_TEST_NEAR(n_jk_buf[1], 0.3378885  * 1 + 0.2, 1e-6);
	CYBOZU_TEST_NEAR(n_jk_buf[2], 0.49160426 * 1 + 0.2, 1e-6);

	CYBOZU_TEST_NEAR(n_k_buf[0], 0.17050723 * 1 + 0.3, 1e-6);
	CYBOZU_TEST_NEAR(n_k_buf[1], 0.3378885  * 1 + 0.3, 1e-6);
	CYBOZU_TEST_NEAR(n_k_buf[2], 0.49160426 * 1 + 0.3, 1e-6);

		cybozu::ldacvb0::update_for_word(
		gamma2,
		n_wk_buf.begin(), n_jk_buf.begin(), n_k_buf.begin(),
		n_wk.begin(), n_jk.begin(), n_k.begin(),
		2, 3, K
		);

	CYBOZU_TEST_NEAR(gamma2[0], 0.922911 , 1e-6);
	CYBOZU_TEST_NEAR(gamma2[1], 0.0384009, 1e-6);
	CYBOZU_TEST_NEAR(gamma2[2], 0.0386877, 1e-6);

	CYBOZU_TEST_NEAR(n_wk_buf[6], 0.922911  * 3 + 0.1, 1e-5);
	CYBOZU_TEST_NEAR(n_wk_buf[7], 0.0384009 * 3 + 0.1, 1e-5);
	CYBOZU_TEST_NEAR(n_wk_buf[8], 0.0386877 * 3 + 0.1, 1e-5);

	CYBOZU_TEST_NEAR(n_jk_buf[0], 0.922911  * 3 + 0.17050723 * 1 + 0.2, 1e-5);
	CYBOZU_TEST_NEAR(n_jk_buf[1], 0.0384009 * 3 + 0.3378885  * 1 + 0.2, 1e-5);
	CYBOZU_TEST_NEAR(n_jk_buf[2], 0.0386877 * 3 + 0.49160426 * 1 + 0.2, 1e-5);

	CYBOZU_TEST_NEAR(n_k_buf[0], 0.922911  * 3 + 0.17050723 * 1 + 0.3, 1e-5);
	CYBOZU_TEST_NEAR(n_k_buf[1], 0.0384009 * 3 + 0.3378885  * 1 + 0.3, 1e-5);
	CYBOZU_TEST_NEAR(n_k_buf[2], 0.0386877 * 3 + 0.49160426 * 1 + 0.3, 1e-5);
}

CYBOZU_TEST_AUTO(test_dirichlet_distribution)
{
	cybozu::ldacvb0::dirichlet_distribution dd(5489U);
	Vec vec;
	dd.draw(vec, 10, 0.1);
	CYBOZU_TEST_EQUAL(vec.size(), 10);
	CYBOZU_TEST_NEAR(vec[0]+vec[1]+vec[2]+vec[3]+vec[4]+vec[5]+vec[6]+vec[7]+vec[8]+vec[9], 1.0, 1e-5);
	CYBOZU_TEST_NEAR(vec[0], 1.33922e-9,  1e-9);
	CYBOZU_TEST_NEAR(vec[1], 7.01106e-10, 1e-15);
	CYBOZU_TEST_NEAR(vec[2], 1.78974e-7,  1e-12);
	CYBOZU_TEST_NEAR(vec[3], 5.01237e-11, 1e-16);
	CYBOZU_TEST_NEAR(vec[4], 3.61902e-8,  1e-13);
	CYBOZU_TEST_NEAR(vec[5], 0.664095,    1e-5);
	CYBOZU_TEST_NEAR(vec[6], 1.64688e-10, 1e-15);
	CYBOZU_TEST_NEAR(vec[7], 0.0692819,   1e-6);
	CYBOZU_TEST_NEAR(vec[8], 4.03312e-24, 1e-29);
	CYBOZU_TEST_NEAR(vec[9], 0.266623,    1e-5);

}

CYBOZU_TEST_AUTO(test_dirichlet_distribution2)
{
	cybozu::ldacvb0::dirichlet_distribution dd(1U);
	double ALPHA[] = {1.0, 2.0, 3.0, 4.0};
	Vec vec, alpha(&ALPHA[0], &ALPHA[4]);
	dd.draw(vec, alpha);
	CYBOZU_TEST_EQUAL(vec.size(), 4);
	CYBOZU_TEST_NEAR(vec[0]+vec[1]+vec[2]+vec[3], 1.0, 1e-5);
	CYBOZU_TEST_NEAR(vec[0], 0.0324463, 1e-6);
	CYBOZU_TEST_NEAR(vec[1], 0.0198952, 1e-6);
	CYBOZU_TEST_NEAR(vec[2], 0.673485,  1e-6);
	CYBOZU_TEST_NEAR(vec[3], 0.274174,  1e-6);
}

CYBOZU_TEST_AUTO(test_lda_cvb0_initialization)
{
	Documents<std::string, char> docs;
	docs.vocabularies.add("a");	// 0
	docs.vocabularies.add("b");	// 1
	docs.vocabularies.add("c");	// 2

	int V = docs.vocabularies.size();
	CYBOZU_TEST_EQUAL(V, 3);

	{
		docs.push_back(Document());
		Document& doc = docs.back();
		doc.push_back(Term(0, 2));
		doc.push_back(Term(1, 1));
	}
	{
		docs.push_back(Document());
		Document& doc = docs.back();
		doc.push_back(Term(0, 1));
		doc.push_back(Term(2, 2));
	}
	{
		docs.push_back(Document());
		Document& doc = docs.back();
		doc.push_back(Term(0, 1));
		doc.push_back(Term(1, 1));
		doc.push_back(Term(2, 1));
	}
	int K = 3, M = 3;
	double alpha = 0.1, beta = 0.01;
	cybozu::ldacvb0::LDA_CVB0 model(K, V, alpha, beta, docs);
	CYBOZU_TEST_EQUAL(model.n_wk.size(), V * K);

	auto &g = model.gamma_jik;
	CYBOZU_TEST_EQUAL(g.size(), 7);
	CYBOZU_TEST_EQUAL(g[0].size(), K);
	CYBOZU_TEST_NEAR(g[0][0], 0.0, 1e-5);
	CYBOZU_TEST_NEAR(g[0][1], 1.0, 1e-5);
	CYBOZU_TEST_NEAR(g[0][2], 0.0, 1e-5);
	CYBOZU_TEST_NEAR(g[1][0], 0.0, 1e-5);
	CYBOZU_TEST_NEAR(g[1][1], 0.0, 1e-5);
	CYBOZU_TEST_NEAR(g[1][2], 1.0, 1e-5);
	CYBOZU_TEST_NEAR(g[2][0], 0.975824,  1e-5);
	CYBOZU_TEST_NEAR(g[2][1], 0.0241761, 1e-5);
	CYBOZU_TEST_NEAR(g[2][2], 0.0, 1e-5);
	CYBOZU_TEST_NEAR(g[3][0], 1.0, 1e-5);
	CYBOZU_TEST_NEAR(g[3][1], 0.0, 1e-5);
	CYBOZU_TEST_NEAR(g[3][2], 0.0, 1e-5);
	CYBOZU_TEST_NEAR(g[4][0], 0.999487, 1e-5);
	CYBOZU_TEST_NEAR(g[4][1], 0.000513486, 1e-7);
	CYBOZU_TEST_NEAR(g[4][2], 0.0, 1e-5);
	CYBOZU_TEST_NEAR(g[5][0], 0.0, 1e-5);
	CYBOZU_TEST_NEAR(g[5][1], 0.0, 1e-5);
	CYBOZU_TEST_NEAR(g[5][2], 1.0, 1e-5);
	CYBOZU_TEST_NEAR(g[6][0], 1.0, 1e-5);
	CYBOZU_TEST_NEAR(g[6][1], 0.0, 1e-5);
	CYBOZU_TEST_NEAR(g[6][2], 0.0, 1e-5);

	CYBOZU_TEST_EQUAL(model.n_k.size(), K);
	double vb = V * beta;
	CYBOZU_TEST_NEAR(model.n_k[0], vb+g[0][0]*2+g[1][0]+g[2][0]+g[3][0]*2+g[4][0]+g[5][0]+g[6][0], 1e-6);
	CYBOZU_TEST_NEAR(model.n_k[1], vb+g[0][1]*2+g[1][1]+g[2][1]+g[3][1]*2+g[4][1]+g[5][1]+g[6][1], 1e-6);
	CYBOZU_TEST_NEAR(model.n_k[2], vb+g[0][2]*2+g[1][2]+g[2][2]+g[3][2]*2+g[4][2]+g[5][2]+g[6][2], 1e-6);

	CYBOZU_TEST_EQUAL(model.n_jk.size(), M * K);
	CYBOZU_TEST_NEAR(model.n_jk[0], alpha+g[0][0]*2+g[1][0], 1e-6);
	CYBOZU_TEST_NEAR(model.n_jk[1], alpha+g[0][1]*2+g[1][1], 1e-6);
	CYBOZU_TEST_NEAR(model.n_jk[2], alpha+g[0][2]*2+g[1][2], 1e-6);
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
