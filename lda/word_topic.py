#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Word/Text Clustering for Cyrillic & Kanji documents with LDA
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import lda_cvb0
import codecs, re, numpy

def output_doc_topic_prob(lda, corpus, g):
    topics = [[] for i in xrange(lda.K)]
    for j, text in enumerate(corpus):
        theta = lda.n_jk[j]
        theta = theta / theta.sum()
        g.write("[%s] %s\n" % (",".join(["%.3f" % x for x in theta]), text))
        topics[theta.argmax()].append(text)
    for k, textlist in enumerate(topics):
        g.write("\n# topic = %d\n" % k)
        for text in textlist:
            g.write(text)
            g.write("\n")
    g.write("\n")

def output_word_topic_dist(lda, voca, g):
    phi = lda.worddist()
    for k in range(lda.K):
        g.write("\n-- topic: %d\n" % k)
        for w in numpy.argsort(-phi[k])[:50]:
            g.write("%s: %f\n" % (voca[w], phi[k,w]))

def main():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=3)
    parser.add_option("-a", "--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
    parser.add_option("-b", "--beta", dest="beta", type="float", help="parameter beta", default=0.5)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=10)
    parser.add_option("--kanji", dest="kanji", help="Kanji as word", action="store_true")
    (options, args) = parser.parse_args()

    if options.kanji:
        word_re = re.compile(u'[\u3040-\u30ff\u4e00-\u9fff]')
    else:
        word_re = re.compile(u'[\u0400-\u04ff]+')

    kanjis = []
    kanji_ids = dict()
    docs = []
    corpus = []
    with codecs.open(args[0], "r", "UTF-8") as f:
        for s in f:
            s = s.rstrip()
            corpus.append(s)
            doc = []
            for m in word_re.finditer(s):
                x = m.group()
                if x not in kanji_ids:
                    kanji_ids[x] = len(kanjis)
                    kanjis.append(x)
                doc.append(kanji_ids[x])
            docs.append(doc)
    V = len(kanjis)
    print "corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(docs), V, options.K, options.alpha, options.beta)

    lda = lda_cvb0.LDA_CVB0(options.K, options.alpha, options.beta, docs, V, True)
    pre_perp = lda.perplexity()
    print "initial perplexity=%f" % pre_perp
    for i in xrange(options.iteration):
        lda.inference()
        perp = lda.perplexity()
        print "-%d p=%f" % (i + 1, perp)

    with codecs.open(args[1], "w", "UTF-8") as g:
        output_doc_topic_prob(lda, corpus, g)
        output_word_topic_dist(lda, kanjis, g)




if __name__ == "__main__":
    main()
