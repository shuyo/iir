#!/usr/bin/python

# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy

class FileOutput:
    def __init__(self, file):
        import datetime
        self.file = file + datetime.datetime.now().strftime('_%m%d_%H%M%S.txt')
    def out(self, st):
        with open(self.file, 'a') as f:
            print >>f,  st

def lda_learning(f, LDA, smartinit, options, docs, voca, plimit=1):
    import time
    t0 = time.time()

    if options.seed != None: numpy.random.seed(options.seed)
    lda = LDA(options.K, options.alpha, options.beta, docs, voca.size(), smartinit)

    pre_perp = lda.perplexity()
    f.out("alg=%s smart_init=%s initial perplexity=%f" % (LDA.__name__, smartinit, pre_perp))

    pc = 0
    for i in range(options.iteration):
        if i % 10==0: output_word_topic_dist(f, lda, voca)
        lda.inference()
        perp = lda.perplexity()
        f.out("-%d p=%f" % (i + 1, perp))
        if pre_perp is not None:
            if pre_perp < perp:
                pc += 1
                if pc >= plimit:
                    output_word_topic_dist(f, lda, voca)
                    pre_perp = None
            else:
                pc = 0
                pre_perp = perp
    output_word_topic_dist(f, lda, voca)

    t1 = time.time()
    f.out("time = %f\n" % (t1 - t0))

def output_word_topic_dist(f, lda, voca):
    phi = lda.worddist()
    for k in range(lda.K):
        f.out("\n-- topic: %d" % k)
        for w in numpy.argsort(-phi[k])[:20]:
            f.out("%s: %f" % (voca[w], phi[k,w]))

def main():
    import optparse
    import vocabulary
    import lda
    import lda_cvb0
    parser = optparse.OptionParser()
    parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)", default="1:100")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.5)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--stopwords", dest="stopwords", help="exclude stop words", action="store_true", default=False)
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=1)
    (options, args) = parser.parse_args()

    corpus = vocabulary.load_corpus(options.corpus)
    voca = vocabulary.Vocabulary(options.stopwords)
    docs = [voca.doc_to_ids(doc) for doc in corpus]
    if options.df > 0: docs = voca.cut_low_freq(docs, options.df)

    f = FileOutput("lda_test")
    f.out("corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(docs), len(voca.vocas), options.K, options.alpha, options.beta))

    lda_learning(f, lda_cvb0.LDA_CVB0, False, options, docs, voca)
    lda_learning(f, lda_cvb0.LDA_CVB0, True, options, docs, voca)
    lda_learning(f, lda.LDA, False, options, docs, voca, 2)
    lda_learning(f, lda.LDA, True, options, docs, voca, 2)

if __name__ == "__main__":
    main()

