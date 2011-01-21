#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# (c)2010 Nakatani Shuyo / Cybozu Labs Inc.

from optparse import OptionParser
import sys, re, numpy

def load_corpus(filename):
    corpus = []
    f = open(filename, 'r')
    for line in f:
        doc = re.findall(r'\w+(?:\'\w+)?',line)
        if len(doc)>0:
            corpus.append(doc)
    f.close()
    return corpus

import nltk
class Vocabulary:
    def __init__(self, excluding_stopwords=True):
        self.vocas = []        # id to word
        self.vocas_id = dict() # word to id
        if excluding_stopwords:
            #self.stopwords = nltk.corpus.stopwords.words('english')
            self.stopwords = "a,s,able,about,above,according,accordingly,across,actually,after,afterwards,again,against,ain,t,all,allow,allows,almost,alone,along,already,also,although,always,am,among,amongst,an,and,another,any,anybody,anyhow,anyone,anything,anyway,anyways,anywhere,apart,appear,appreciate,appropriate,are,aren,t,around,as,aside,ask,asking,associated,at,available,away,awfully,be,became,because,become,becomes,becoming,been,before,beforehand,behind,being,believe,below,beside,besides,best,better,between,beyond,both,brief,but,by,c,mon,c,s,came,can,can,t,cannot,cant,cause,causes,certain,certainly,changes,clearly,co,com,come,comes,concerning,consequently,consider,considering,contain,containing,contains,corresponding,could,couldn,t,course,currently,definitely,described,despite,did,didn,t,different,do,does,doesn,t,doing,don,t,done,down,downwards,during,each,edu,eg,eight,either,else,elsewhere,enough,entirely,especially,et,etc,even,ever,every,everybody,everyone,everything,everywhere,ex,exactly,example,except,far,few,fifth,first,five,followed,following,follows,for,former,formerly,forth,four,from,further,furthermore,get,gets,getting,given,gives,go,goes,going,gone,got,gotten,greetings,had,hadn,t,happens,hardly,has,hasn,t,have,haven,t,having,he,he,s,hello,help,hence,her,here,here,s,hereafter,hereby,herein,hereupon,hers,herself,hi,him,himself,his,hither,hopefully,how,howbeit,however,i,d,i,ll,i,m,i,ve,ie,if,ignored,immediate,in,inasmuch,inc,indeed,indicate,indicated,indicates,inner,insofar,instead,into,inward,is,isn,t,it,it,d,it,ll,it,s,its,itself,just,keep,keeps,kept,know,knows,known,last,lately,later,latter,latterly,least,less,lest,let,let,s,like,liked,likely,little,look,looking,looks,ltd,mainly,many,may,maybe,me,mean,meanwhile,merely,might,more,moreover,most,mostly,much,must,my,myself,name,namely,nd,near,nearly,necessary,need,needs,neither,never,nevertheless,new,next,nine,no,nobody,non,none,noone,nor,normally,not,nothing,novel,now,nowhere,obviously,of,off,often,oh,ok,okay,old,on,once,one,ones,only,onto,or,other,others,otherwise,ought,our,ours,ourselves,out,outside,over,overall,own,particular,particularly,per,perhaps,placed,please,plus,possible,presumably,probably,provides,que,quite,qv,rather,rd,re,really,reasonably,regarding,regardless,regards,relatively,respectively,right,said,same,saw,say,saying,says,second,secondly,see,seeing,seem,seemed,seeming,seems,seen,self,selves,sensible,sent,serious,seriously,seven,several,shall,she,should,shouldn,t,since,six,so,some,somebody,somehow,someone,something,sometime,sometimes,somewhat,somewhere,soon,sorry,specified,specify,specifying,still,sub,such,sup,sure,t,s,take,taken,tell,tends,th,than,thank,thanks,thanx,that,that,s,thats,the,their,theirs,them,themselves,then,thence,there,there,s,thereafter,thereby,therefore,therein,theres,thereupon,these,they,they,d,they,ll,they,re,they,ve,think,third,this,thorough,thoroughly,those,though,three,through,throughout,thru,thus,to,together,too,took,toward,towards,tried,tries,truly,try,trying,twice,two,un,under,unfortunately,unless,unlikely,until,unto,up,upon,us,use,used,useful,uses,using,usually,value,various,very,via,viz,vs,want,wants,was,wasn,t,way,we,we,d,we,ll,we,re,we,ve,welcome,well,went,were,weren,t,what,what,s,whatever,when,whence,whenever,where,where,s,whereafter,whereas,whereby,wherein,whereupon,wherever,whether,which,while,whither,who,who,s,whoever,whole,whom,whose,why,will,willing,wish,with,within,without,won,t,wonder,would,would,wouldn,t,yes,yet,you,you,d,you,ll,you,re,you,ve,your,yours,yourself,yourselves,zero".split(',')
        else:
            self.stopwords = []
        self.wl = nltk.WordNetLemmatizer()
    def term_to_id(self, term):
        term = self.wl.lemmatize(term.lower())
        if not re.match(r'[a-z]+$', term): return None
        if term in self.stopwords: return None
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            self.vocas.append(term)
        else:
            voca_id = self.vocas_id[term]
        return voca_id
    def doc_to_ids(self, doc):
        list = []
        for term in doc:
            id = self.term_to_id(term)
            if id: list.append(id)
        return list
    def __getitem__(self, v):
        return self.vocas[v]
    def size(self):
        return len(self.vocas)

    @staticmethod
    def load_reuters(start, end):
        from nltk.corpus import reuters
        #return [reuters.words(fileid) for fileid in reuters.fileids(category)]
        return [reuters.words(fileid) for fileid in reuters.fileids()[start:end]]


class LDA:
    def __init__(self, K, alpha, beta):
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior

    def set_corpus(self, corpus):
        """set courpus and initialize"""
        voca = Vocabulary()
        self.docs = [voca.doc_to_ids(doc) for doc in corpus]

        M = len(self.docs)
        self.V = voca.size()

        self.z_m_n = [] # topics of words of documents
        self.n_m_z = numpy.zeros((M, self.K), dtype=int) # word count of each document and topic
        self.n_z_t = numpy.zeros((self.K, self.V), dtype=int) # word count of each topic and vocabulary
        self.n_z = numpy.zeros(self.K, dtype=int)        # word count of each topic

        for m, doc in zip(range(M), self.docs):
            z_n = numpy.random.randint(0, self.K, len(doc))
            self.z_m_n.append(z_n)
            for t, z in zip(doc, z_n):
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
        return voca

    def inference(self):
        for m, doc in zip(range(len(self.docs)), self.docs):
            for n, t, z in zip(range(len(doc)), doc, self.z_m_n[m]):
                # discount for n-th word t with topic z
                self.n_m_z[m, z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                # sampling topic new_z for t
                denom_b = self.n_z_t.sum(axis=1) + self.V * self.beta
                p_z = (self.n_z_t[:, t] + self.beta) * (self.n_m_z[m] + self.alpha) / denom_b
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                self.z_m_n[m][n] = new_z
                self.n_m_z[m, new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def phi(self):
        return (self.n_z_t + self.beta) / (self.n_z[:, numpy.newaxis] + self.V * self.beta)

    def predictive(self, doc):
        pass

def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-r", dest="reuters", help="corpus range of Reuters' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.001)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.001)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    (options, args) = parser.parse_args()
    if not (options.filename or options.reuters): parser.error("need corpus filename(-f) or Reuters range(-r)")

    if options.filename:
        corpus = load_corpus(options.filename)
    else:
        m = re.match(r'(\d+):(\d+)$', options.reuters)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            corpus = Vocabulary.load_reuters(start, end)
        else:
            parser.error("Reuters range(-r) forms 'start:end'")
    lda = LDA(options.K, options.alpha, options.beta)
    voca = lda.set_corpus(corpus)
    print "corpus : %d, words : %d" % (len(corpus), len(voca.vocas))

    for i in range(options.iteration):
        sys.stderr.write("-- %d " % (i + 1))
        lda.inference()
    #print lda.z_m_n

    phi = lda.phi()
    #for v, term in enumerate(voca):
    #    print ','.join([term]+[str(x) for x in phi[:,v]])
    for k in range(options.K):
        print "topic: %d" % k
        for w in numpy.argsort(-phi[k,:])[:20]:
            print "%s: %f" % (voca[w], phi[k,w])
        print

if __name__ == "__main__":
    main()
