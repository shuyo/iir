#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import hdplda2

class TestHDPLDA(unittest.TestCase):
    def test1(self):
        alpha = beta = gamma = 0.1
        docs = [[0,1,2,3], [0,1,4,5], [0,1,5,6]]
        V = 7
        model = hdplda2.HDPLDA(alpha, beta, gamma, docs, V)

        j = 0
        i = 0
        v = docs[j][i]
        self.assertEqual(v, 0)

        f_k = model.calc_f_k(v)
        self.assertSequenceEqual(f_k, [0.])
        p_t = model.calc_table_posterior(j, f_k)
        self.assertSequenceEqual(p_t, [1.])

        p_k = model.calc_dish_posterior_w(f_k)
        self.assertEqual(len(p_k), 1)
        self.assertAlmostEqual(p_k[0], 1)

        k_new = model.add_new_topic()
        self.assertEqual(k_new, 1)
        t_new = model.add_new_table(j, f_k, k_new)
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
        self.assertAlmostEqual(f_k[0], 0)
        self.assertAlmostEqual(f_k[1], (beta+0)/(V*beta+1))
        p_t = model.calc_table_posterior(j, f_k)
        self.assertEqual(len(p_t), 2)
        self.assertAlmostEqual(p_t[0], 0.10151692)
        self.assertAlmostEqual(p_t[1], 0.89848308)

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

        t_new = model.add_new_table(j, f_k, k_new)
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

        k_new = 1
        t_new = model.add_new_table(j, f_k, k_new)
        self.assertEqual(t_new, 1)

        self.assertListEqual(model.using_t[j], [0, 1])
        self.assertListEqual(model.using_k, [0, 1])
        self.assertEqual(model.n_jt[j][t_new], 0) # まだ 0

        model.seat_at_table(j, i, t_new)
        self.assertEqual(model.t_ji[j][i], 1)
        self.assertEqual(model.n_jt[j][t_new], 1) # ふえた
        self.assertAlmostEqual(model.n_kv[k_new][v], beta + 2)


unittest.main()

