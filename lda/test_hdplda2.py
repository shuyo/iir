#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy
import hdplda2

class TestHDPLDA(unittest.TestCase):
    def test1(self):
        self.sequence1(0.1, 0.1, 0.1)

    def test2(self):
        self.sequence1(0.2, 0.01, 0.5)

    def test4(self):
        self.sequence3(0.2, 0.01, 0.5)
        pass

    def test5(self):
        self.sequence4(0.2, 0.01, 0.5)
        pass

    def test7(self):
        self.sequence2(0.01, 0.001, 10)

    def test8(self):
        self.sequence2(0.01, 0.001, 0.05)


    def test_random_sequences(self):
        self.sequence_random(0.2, 0.01, 0.5, 0)
        self.sequence_random(0.2, 0.01, 0.01, 6)
        self.sequence_random(0.2, 0.01, 0.5, 2)
        self.sequence_random(0.01, 0.001, 0.05, 13)
        pass


    def sequence_random(self, alpha, beta, gamma, seed):
        print (alpha, beta, gamma)
        numpy.random.seed(seed)
        docs = [[0,1,2,3], [0,1,4,5], [0,1,5,6]]
        V = 7
        model = hdplda2.HDPLDA(alpha, beta, gamma, docs, V)
        print model.perplexity()
        for i in xrange(10):
            model.inference()
            print model.perplexity()

    def sequence4(self, alpha, beta, gamma):
        docs = [[0,1,2,3], [0,1,4,5], [0,1,5,6]]
        V = 7
        model = hdplda2.HDPLDA(alpha, beta, gamma, docs, V)
        Vbeta = V * beta

        k1 = model.add_new_dish()
        k2 = model.add_new_dish()

        j = 0
        t1 = model.add_new_table(j, k1)
        t2 = model.add_new_table(j, k2)
        model.seat_at_table(j, 0, t1)
        model.seat_at_table(j, 1, t2)
        model.seat_at_table(j, 2, t2)
        model.seat_at_table(j, 3, t2)

        j = 1
        t1 = model.add_new_table(j, k1)
        t2 = model.add_new_table(j, k2)
        model.seat_at_table(j, 0, t2)
        model.seat_at_table(j, 1, t2)
        model.seat_at_table(j, 2, t1)
        model.seat_at_table(j, 3, t2)

        j = 2
        t1 = model.add_new_table(j, k1)
        t2 = model.add_new_table(j, k2)
        model.seat_at_table(j, 0, t1)
        model.seat_at_table(j, 1, t2)
        model.seat_at_table(j, 2, t2)
        model.seat_at_table(j, 3, t2)


        model.leave_from_dish(2, 1)
        model.seat_at_dish(2, 1, 2)


        model.leave_from_table(2, 0)
        model.seat_at_table(2, 0, 2)


        model.leave_from_dish(0, 1)
        model.seat_at_dish(0, 1, 2)
        self.assertEqual(model.m, 5)
        self.assertEqual(model.m_k[1], 1)
        self.assertEqual(model.m_k[2], 4)

        model.leave_from_dish(1, 1)
        model.seat_at_dish(1, 1, 2)

        model.leave_from_table(2, 3)
        k_new = model.add_new_dish()
        self.assertEqual(k_new, 1)
        t_new = model.add_new_table(j, k_new)
        self.assertEqual(t_new, 1)
        model.seat_at_table(2, 3, 1)

        #model.dump()
        #using_t: [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        #t_ji: [[1, 2, 2, 2], [2, 2, 1, 2], [2, 2, 2, 1]]
        #using_k: [0, 1, 2]
        #k_jt: [[0, 2, 2], [0, 2, 2], [0, 1, 2]]

        j = 0
        t = 1
        model.leave_from_dish(j, t)

        #print "n_jt=", model.n_jt[j][t]

        p_k = model.calc_dish_posterior_t(j, t)
        #print "p_k=", p_k
        p0 = gamma / V
        p1 = 1 * beta / (V * beta + 1)
        p2 = 4 * (beta + 2) / (Vbeta + 10)
        #print "[p0, p1, p2]=", [p0, p1, p2]
        self.assertAlmostEqual(p_k[0], p0 / (p0 + p1 + p2))
        self.assertAlmostEqual(p_k[1], p1 / (p0 + p1 + p2))
        self.assertAlmostEqual(p_k[2], p2 / (p0 + p1 + p2))

        #k_new = self.add_new_dish()
        model.seat_at_dish(j, t, 1)

        t = 2
        model.leave_from_dish(j, t)

        #print "n_jt=", model.n_jt[j][t]

        p_k = model.calc_dish_posterior_t(j, t)
        #print "p_k=", p_k

        p0 = gamma * beta * beta * beta / (Vbeta * (Vbeta + 1) * (Vbeta + 2))
        p1 = 2 * (beta + 0) * beta * beta / ((Vbeta + 2) * (Vbeta + 3) * (Vbeta + 4))
        p2 = 3 * (beta + 2) * beta * beta / ((Vbeta + 7) * (Vbeta + 8) * (Vbeta + 9))
        #print "[p0, p1, p2]=", [p0, p1, p2]
        self.assertAlmostEqual(p_k[0], p0 / (p0 + p1 + p2))
        self.assertAlmostEqual(p_k[1], p1 / (p0 + p1 + p2))
        self.assertAlmostEqual(p_k[2], p2 / (p0 + p1 + p2))

        #k_new = self.add_new_dish()
        model.seat_at_dish(j, t, 1)



    def sequence3(self, alpha, beta, gamma):
        docs = [[0,1,2,3], [0,1,4,5], [0,1,5,6]]
        V = 7
        model = hdplda2.HDPLDA(alpha, beta, gamma, docs, V)

        k1 = model.add_new_dish()
        k2 = model.add_new_dish()

        j = 0
        t1 = model.add_new_table(j, k1)
        t2 = model.add_new_table(j, k2)
        model.seat_at_table(j, 0, t1)
        model.seat_at_table(j, 1, t2)
        model.seat_at_table(j, 2, t1)
        model.seat_at_table(j, 3, t1)

        j = 1
        t1 = model.add_new_table(j, k1)
        t2 = model.add_new_table(j, k2)
        model.seat_at_table(j, 0, t1)
        model.seat_at_table(j, 1, t2)
        model.seat_at_table(j, 2, t2)
        model.seat_at_table(j, 3, t2)

        j = 2
        t1 = model.add_new_table(j, k1)
        t2 = model.add_new_table(j, k2)
        model.seat_at_table(j, 0, t1)
        model.seat_at_table(j, 1, t2)
        model.seat_at_table(j, 2, t2)
        model.seat_at_table(j, 3, t2)

        #model.dump()

        # test for topic-word distribution
        phi = model.worddist()
        self.assertEqual(len(phi), 2)
        self.assertAlmostEqual(phi[0][0], (beta+3)/(V*beta+5))
        self.assertAlmostEqual(phi[0][2], (beta+1)/(V*beta+5))
        self.assertAlmostEqual(phi[0][3], (beta+1)/(V*beta+5))
        for v in [1,4,5,6]:
            self.assertAlmostEqual(phi[0][v], (beta+0)/(V*beta+5))
        self.assertAlmostEqual(phi[1][1], (beta+3)/(V*beta+7))
        self.assertAlmostEqual(phi[1][4], (beta+1)/(V*beta+7))
        self.assertAlmostEqual(phi[1][5], (beta+2)/(V*beta+7))
        self.assertAlmostEqual(phi[1][6], (beta+1)/(V*beta+7))
        for v in [0,2,3]:
            self.assertAlmostEqual(phi[1][v], (beta+0)/(V*beta+7))


        # test for document-topic distribution
        theta = model.docdist()
        self.assertEqual(theta.shape, (3, 3))
        self.assertAlmostEqual(theta[0][0], (  alpha*gamma/(6+gamma))/(4+alpha))
        self.assertAlmostEqual(theta[0][1], (3+alpha*  3  /(6+gamma))/(4+alpha))
        self.assertAlmostEqual(theta[0][2], (1+alpha*  3  /(6+gamma))/(4+alpha))
        self.assertAlmostEqual(theta[1][0], (  alpha*gamma/(6+gamma))/(4+alpha))
        self.assertAlmostEqual(theta[1][1], (1+alpha*  3  /(6+gamma))/(4+alpha))
        self.assertAlmostEqual(theta[1][2], (3+alpha*  3  /(6+gamma))/(4+alpha))
        self.assertAlmostEqual(theta[2][0], (  alpha*gamma/(6+gamma))/(4+alpha))
        self.assertAlmostEqual(theta[2][1], (1+alpha*  3  /(6+gamma))/(4+alpha))
        self.assertAlmostEqual(theta[2][2], (3+alpha*  3  /(6+gamma))/(4+alpha))

        j = 0
        i = 0
        v = docs[j][i]

        model.leave_from_table(j, i)

        f_k = model.calc_f_k(v)
        self.assertEqual(len(f_k), 3)
        self.assertAlmostEqual(f_k[1], (beta+2)/(V*beta+4))
        self.assertAlmostEqual(f_k[2], (beta+0)/(V*beta+7))

        p_t = model.calc_table_posterior(j, f_k)
        self.assertEqual(len(p_t), 3)
        p1 = 2 * f_k[1]
        p2 = 1 * f_k[2]
        p0 = alpha / (6+gamma) * (3*f_k[1] + 3*f_k[2] + gamma/V)
        self.assertAlmostEqual(p_t[0], p0 / (p0+p1+p2))
        self.assertAlmostEqual(p_t[1], p1 / (p0+p1+p2))
        self.assertAlmostEqual(p_t[2], p2 / (p0+p1+p2))

        model.seat_at_table(j, i, 1)

        j = 0
        i = 1
        v = docs[j][i]

        model.leave_from_table(j, i)
        self.assertEqual(len(model.using_t[j]), 2)
        self.assertEqual(model.using_t[j][0], 0)
        self.assertEqual(model.using_t[j][1], 1)

        f_k = model.calc_f_k(v)
        self.assertEqual(len(f_k), 3)
        self.assertAlmostEqual(f_k[1], (beta+0)/(V*beta+5))
        self.assertAlmostEqual(f_k[2], (beta+2)/(V*beta+6))

        p_t = model.calc_table_posterior(j, f_k)
        self.assertEqual(len(p_t), 2)
        p1 = 3 * f_k[1]
        p0 = alpha / (5+gamma) * (3*f_k[1] + 2*f_k[2] + gamma/V)
        self.assertAlmostEqual(p_t[0], p0 / (p0+p1))
        self.assertAlmostEqual(p_t[1], p1 / (p0+p1))

        model.seat_at_table(j, i, 1)




    def sequence2(self, alpha, beta, gamma):
        docs = [[0,1,2,3], [0,1,4,5], [0,1,5,6]]
        V = 7
        model = hdplda2.HDPLDA(alpha, beta, gamma, docs, V)

        # assign all words to table 1 and all tables to dish 1
        k_new = model.add_new_dish()
        self.assertEqual(k_new, 1)
        for j in xrange(3):
            t_new = model.add_new_table(j, k_new)
            self.assertEqual(t_new, 1)
            for i in xrange(4):
                model.seat_at_table(j, i, t_new)

        self.assertAlmostEqual(model.n_k[0], beta * V)
        self.assertAlmostEqual(model.n_k[1], beta * V + 12)
        self.assertAlmostEqual(model.n_kv[1][0], beta + 3)
        self.assertAlmostEqual(model.n_kv[1][1], beta + 3)
        self.assertAlmostEqual(model.n_kv[1][2], beta + 1)
        self.assertAlmostEqual(model.n_kv[1][3], beta + 1)
        self.assertAlmostEqual(model.n_kv[1][4], beta + 1)
        self.assertAlmostEqual(model.n_kv[1][5], beta + 2)
        self.assertAlmostEqual(model.n_kv[1][6], beta + 1)
        self.assertEqual(model.m_k[0], 1) # dummy
        self.assertEqual(model.m_k[1], 3)

        #model.sampling_k(0, 1)
        model.leave_from_dish(0, 1) # decrease m and m_k only
        self.assertEqual(model.m, 2)
        self.assertEqual(model.m_k[1], 2)

        model.seat_at_dish(0, 1, 1)
        self.assertEqual(model.m, 3)
        self.assertEqual(model.m_k[1], 3)

        for i in xrange(1):
            for j in xrange(3):
                model.sampling_k(j, 1)
                #model.dump()


    def sequence1(self, alpha, beta, gamma):
        docs = [[0,1,2,3], [0,1,4,5], [0,1,5,6]]
        V = 7
        model = hdplda2.HDPLDA(alpha, beta, gamma, docs, V)

        j = 0
        i = 0
        v = docs[j][i]
        self.assertEqual(v, 0)

        f_k = model.calc_f_k(v)
        #self.assertSequenceEqual(f_k, [0.])
        p_t = model.calc_table_posterior(j, f_k)
        self.assertSequenceEqual(p_t, [1.])

        p_k = model.calc_dish_posterior_w(f_k)
        self.assertEqual(len(p_k), 1)
        self.assertAlmostEqual(p_k[0], 1)

        k_new = model.add_new_dish()
        self.assertEqual(k_new, 1)
        t_new = model.add_new_table(j, k_new)
        self.assertEqual(t_new, 1)
        self.assertEqual(model.k_jt[j][t_new], 1)

        self.assertListEqual(model.using_t[j], [0, 1])
        self.assertListEqual(model.using_k, [0, 1])
        self.assertEqual(model.n_jt[j][t_new], 0) # まだ 0

        model.seat_at_table(j, i, t_new)
        self.assertEqual(model.t_ji[j][i], 1)
        self.assertEqual(model.n_jt[j][t_new], 1) # ふえた
        self.assertEqual(model.n_kv[k_new][v], beta + 1)


        i = 1 # the existed table
        v = docs[j][i]
        self.assertEqual(v, 1)

        f_k = model.calc_f_k(v)
        self.assertEqual(len(f_k), 2)
        #self.assertAlmostEqual(f_k[0], 0)
        self.assertAlmostEqual(f_k[1], (beta+0)/(V*beta+1))
        p_t = model.calc_table_posterior(j, f_k)
        self.assertEqual(len(p_t), 2)
        p0 = alpha / (1 + gamma) * (beta / (V * beta + 1) + gamma / V)
        p1 = 1 * beta / (V * beta + 1)
        self.assertAlmostEqual(p_t[0], p0 / (p0 + p1))  # 0.10151692
        self.assertAlmostEqual(p_t[1], p1 / (p0 + p1))  # 0.89848308

        t_new = 1
        model.seat_at_table(j, i, t_new)
        self.assertEqual(model.t_ji[j][i], t_new)
        self.assertEqual(model.n_jt[j][t_new], 2) # ふえた
        self.assertEqual(model.n_kv[k_new][v], beta + 1)


        i = 2
        v = docs[j][i]
        self.assertEqual(v, 2)

        f_k = model.calc_f_k(v)
        self.assertEqual(len(f_k), 2)
        self.assertAlmostEqual(f_k[0], 0)
        self.assertAlmostEqual(f_k[1], (beta+0)/(V*beta+2))
        p_t = model.calc_table_posterior(j, f_k)
        self.assertEqual(len(p_t), 2)
        p0 = alpha / (1 + gamma) * (beta / (V * beta + 2) + gamma / V)
        p1 = 2 * beta / (V * beta + 2)
        self.assertAlmostEqual(p_t[0], p0 / (p0 + p1))  # 0.05925473
        self.assertAlmostEqual(p_t[1], p1 / (p0 + p1))  # 0.94074527

        p_k = model.calc_dish_posterior_w(f_k)
        self.assertEqual(len(p_k), 2)
        p0 = gamma / V
        p1 = 1 * f_k[1]
        self.assertAlmostEqual(p_k[0], p0 / (p0 + p1))  # 0.27835052
        self.assertAlmostEqual(p_k[1], p1 / (p0 + p1))  # 0.72164948

        k_new = 1 # TODO : calculate posterior of k

        t_new = model.add_new_table(j, k_new)
        self.assertEqual(t_new, 2)
        self.assertEqual(k_new, model.k_jt[j][t_new])

        self.assertListEqual(model.using_t[j], [0, 1, 2])
        self.assertListEqual(model.using_k, [0, 1])

        model.seat_at_table(j, i, t_new)
        self.assertEqual(model.t_ji[j][i], t_new)
        self.assertEqual(model.n_jt[j][t_new], 1)
        self.assertEqual(model.n_kv[k_new][v], beta + 1)


        i = 3
        v = docs[j][i]
        self.assertEqual(v, 3)

        f_k = model.calc_f_k(v)
        self.assertEqual(len(f_k), 2)
        self.assertAlmostEqual(f_k[0], 0)
        self.assertAlmostEqual(f_k[1], (beta+0)/(V*beta+3))
        p_t = model.calc_table_posterior(j, f_k)
        self.assertEqual(len(p_t), 3)
        p0 = alpha / (2 + gamma) * (2 * beta / (V * beta + 3) + gamma / V)
        p1 = 2 * beta / (V * beta + 3)
        p2 = 1 * beta / (V * beta + 3)
        self.assertAlmostEqual(p_t[0], p0 / (p0 + p1 + p2))  # 0.03858731
        self.assertAlmostEqual(p_t[1], p1 / (p0 + p1 + p2))  # 0.64094179
        self.assertAlmostEqual(p_t[2], p2 / (p0 + p1 + p2))  # 0.3204709

        t_new = 1
        model.seat_at_table(j, i, t_new)
        self.assertEqual(model.t_ji[j][i], t_new)
        self.assertEqual(model.n_jt[j][t_new], 3)
        self.assertEqual(model.n_kv[k_new][v], beta + 1)


        j = 1
        i = 0
        v = docs[j][i]
        self.assertEqual(v, 0)

        f_k = model.calc_f_k(v)
        self.assertEqual(len(f_k), 2)
        self.assertAlmostEqual(f_k[0], 0)
        self.assertAlmostEqual(f_k[1], (beta+1)/(V*beta+4)) # 0.23404255

        p_t = model.calc_table_posterior(j, f_k)
        self.assertEqual(len(p_t), 1)
        self.assertAlmostEqual(p_t[0], 1)

        # add x_10 into a new table with dish 1
        k_new = 1
        t_new = model.add_new_table(j, k_new)
        self.assertEqual(t_new, 1)

        self.assertListEqual(model.using_t[j], [0, 1])
        self.assertListEqual(model.using_k, [0, 1])
        self.assertEqual(model.n_jt[j][t_new], 0) # まだ 0

        model.seat_at_table(j, i, t_new)
        self.assertEqual(model.t_ji[j][i], 1)
        self.assertEqual(model.n_jt[j][t_new], 1) # ふえた
        self.assertAlmostEqual(model.n_kv[k_new][v], beta + 2)


unittest.main()

