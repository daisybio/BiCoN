#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing as mp
from multiprocessing import Process, Queue
import time
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.cluster import KMeans
import gc
import operator
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

sns.set(color_codes=True)
flatten = lambda l: [item for sublist in l for item in sublist]


class BiCoN(object):
    def __init__(self, GE, G, L_g_min, L_g_max):
        self.GE = GE
        self.G = G
        self.L_g_min = L_g_min
        self.L_g_max = L_g_max

    def run_search(self, n_proc=1, a=1, b=1, K=20, evaporation=0.5, th=1, eps=0.02,
                   times=6, clusters=2, cost_limit=5, max_iter=100, ls=True, opt=None, show_pher=False,
                   show_plot=False, save=None, show_nets=False, verbose=True, logging = False):

        """
        Parallel implementation of network constrained bi-clustering

        Attributes:
        -----------
        non-default:

        GE - pandas data frame with gene expression data. Genes are rows, patients - columns
        G - networkX graph with a network
        L_g_min - minimal number of genes in one subnetwork
        L_g_max - maximal number of genes in one subnetwork

        default:
        K - number of ants (less ants - less space exploration. Usually set between 20 and 50, default - 20)
        n_proc = number of processes that should be used (default 1)
        a - pheromone significance (default 1 - does not need to be changed)
        b - heuristic information significance (default 1 - does not need to be changed)
        evaporation - the rate at which pheromone evaporates (default 0.5)
        th - similarity threshold (default 1 - does not need to be changed)
        eps - conservative convergence criteria: score_max - score_min < eps (default- 0.02)
        times - allows faster convergence criteria: stop if the maximum so far was reached more than x times (default 6)
        clusters - # of clusters, right now does not work for more than 2
        cost_limit - defines the radius of the search for ants (default 5)
        max_iter - maximum number of interactions allowed (default 200)
        opt - given if the best score is known apriori (in case of simulated data for instance)
        show_pher - set true if plotting of pheromone heatmap at every iteration is desirable (strictly NOT recommended for # of genes > 2000)
        show_plot - set true if convergence plots should be shown
        save - set an output file name  if the convergence plot should be saved in the end
        show_nets - set true if the selected network should be shown at each iteration
        verbose - set true to print information about the algorithm's progress
        logging - set true to save information about algorithm's progress to bicon_logging.txt file


        """
        assert n_proc > 0, "Set a correct number for n_proc, right now the value is {0}".format(n_proc)
        assert n_proc <= mp.cpu_count() - 1, 'n_proc should not exceed {0}. The value of n_proc was: {1}'.format(
            mp.cpu_count(), n_proc)
        assert n_proc <= K, 'Number of ants (K) can not be lower as number of processes, please set higher K or lower n_proc'
        # adjacency matrix
        A = nx.adj_matrix(self.G).todense()
        # heurisic information
        H = self.HI_big(self.GE, A)
        H = H.astype(np.short)
        if logging:
            f = open("bicon_logs.txt", "w")
        n, m = self.GE.shape
        # TODO: rewrite functions such that they all could use numpy matrices
        ge = self.GE.values
        # determination of search radius for each patient
        N = self.neigborhood(H, n, th)
        # inner patients IDs
        patients = np.arange(n, n + m)
        # cost of transitions for ants
        cost = H / 10
        cost = 0.1 + np.max(cost) - cost
        # stores all scores
        scores = []
        avs = []
        count_big = 0
        max_total_score = -100
        n1, n2 = (0,0)
        max_round_score = -100
        av_score = 0
        # initial pheromone level set to a maximal possible level (5 standard deviations)
        t0 = np.ones((n + m, n + m)) * 5
        t0 = t0.astype(np.short)
        t_min = 0
        # initial probabilities
        st = time.time()
        probs,war = self.prob_upd(H, t0, a, b, n, th, N)
        while war == True and th >-1:
            th = th-0.5
            N = self.neigborhood(H, n, th)
            probs, war = self.prob_upd(H, t0, a, b, n, th, N)


        end = time.time()
        # flag tracks when the score stops improving and terminates the optimization as convergence is reached
        score_change = []
        p_time = str(round(end - st, 3))
        if verbose:
            print("Runtime statistics:")
            print("###############################################################")
            print("the joint graph has " + str(n + m) + " nodes")

            print("probability update takes " + p_time)
        if logging:
            f.write("Runtime statistics:")
            f.write("\n")
            f.write("###############################################################")
            f.write("\n")
            f.write("the joint graph has " + str(n + m) + " nodes")
            f.write("\n")
            f.write("probability update takes " + p_time)
            f.write("\n")





        count_small = 0
        W = 0
        # termination if the improvements are getting too small
        while np.abs(max_round_score - av_score) > eps and count_small < times and count_big < max_iter and (W<m/3):
            # MULTIPROCESSING SCHEMA
            if n_proc > 1:
                av_score = 0
                W = 0
                max_round_score = 0
                result = Queue()
                jobs = []
                ants_per_batch = round(K / n_proc)

                for pr in range(n_proc):
                    # random random seeds to avoid identical random walks
                    ss = np.random.choice(np.arange(pr * ants_per_batch, pr * ants_per_batch + ants_per_batch),
                                          ants_per_batch, replace=False)
                    p = Process(target=self.ant_job_paral, args=(
                        self.GE, N, H, th, clusters, probs, a, b, cost, m, n, patients, count_big, cost_limit,
                        self.L_g_min,
                        self.L_g_max, self.G, ge, ants_per_batch, pr, ss, result,))
                    jobs.append(p)
                    p.start()
                while 1:
                    running = any(p.is_alive() for p in jobs)
                    while not result.empty():
                        res = result.get()
                        s1, s2 = res[0]
                        s = s1 + s2
                        av_score = av_score + res[-2]
                        W = W + res[-1]
                        # if maximum round score is larger than the current value for this round
                        if s > max_round_score:
                            # save the results
                            max_round_score = s
                            n1 = s1
                            n2 = s2
                            solution= res[1]
                            solution_big = res[2]
                    if not running:
                        break

                av_score = av_score / n_proc

                # after all ants have finished:
                scores.append(max_round_score)
                avs.append(av_score)
                gc.collect()
            # SINGLE PROCESS SCHEMA (RECOMMENDED FOR PC)
            else:
                av_score = 0
                max_round_score = 0
                scores_per_round = []
                st = time.time()
                for i in range(K):
                    # for each ant
                    tot_score, gene_groups, patients_groups, new_scores, wars, no_int = self.ant_job(self.GE, N, H, th,
                                                                                                     clusters, probs, a,
                                                                                                     b, cost, m, n,
                                                                                                     patients,
                                                                                                     count_big,
                                                                                                     cost_limit,
                                                                                                     self.L_g_min,
                                                                                                     self.L_g_max,
                                                                                                     self.G, ge)
                    scores_per_round.append(tot_score)
                    av_score = av_score + tot_score
                    W = W + wars
                    if tot_score > max_round_score:
                        max_round_score = tot_score
                        solution = (gene_groups, patients_groups)
                        solution_big = (no_int, patients_groups)
                        n1, n2 = (new_scores[0][0] * new_scores[0][1], new_scores[1][0] * new_scores[1][1])

                av_score = av_score / K
                avs.append(av_score)
                # after all ants have finished:
                scores.append(max_round_score)

                end = time.time()
                if verbose:
                    print("total run-time is {0}".format(end - st))
                if logging:
                    f.write("total run-time is {0}".format(end - st))
                    f.write("\n")

            # saving rhe best overall solution
            if np.round(max_round_score, 3) == np.round(max_total_score, 3):
                count_small = count_small + 1

            if max_round_score > max_total_score:
                max_total_score = max_round_score
                best_solution = solution
                solution_big_best = solution_big
                count_small = 0

            score_change.append(round(max_round_score, 3))
            ant_time = round(time.time() - st, 2)
            if verbose:
                print("Iteration # " + str(count_big + 1))
                if count_big == 0:
                    print("One ant work takes {0} with {1} processes".format(ant_time, n_proc))
                print("best round score: " + str(round(max_round_score, 3)))
                print("average score: " + str(round(av_score, 3)))
                print("Count small = {}".format(count_small))
            if logging:
                f.write("Iteration # " + str(count_big + 1))
                f.write("\n")
                if count_big == 0:
                    f.write("One ant work takes {0} with {1} processes".format(ant_time, n_proc))
                    f.write("\n")
                f.write("best round score: " + str(round(max_round_score, 3)))
                f.write("\n")
                f.write("average score: " + str(round(av_score, 3)))
                f.write("\n")
                f.write("Count small = {}".format(count_small))
                f.write("\n")

            # Pheromone update
            t0 = self.pher_upd(t0, t_min, evaporation, [n1, n2], solution_big_best)
            # Probability update
            probs,_ = self.prob_upd(H, t0, a, b, n, th, N)
            # if probs[0][0, :].sum() == 0:
            #     print("threshold lowered")
            #     th = 0
            #     probs = self.prob_upd(H, t0, a, b, n, th, N)
            # # lower the threshold even more even there are still not enough conected genes
            # if probs[0][0, :].sum() == 0:
            #     th = -1
            #     print("minimal possible threshold employed)")
            #     probs = self.prob_upd(H, t0, a, b, n, th, N)
            # assert probs[0][0,
            #        :].sum() != 0, 'Bad probability update. This often happens when there are not enough connected genes or ' \
            #                       'the signal in the data is too weak. Try selecting more genes (e.g. 3000)'
            if count_big == 0 and verbose:
                print("One full iteration takes {0} with {1} processes".format(round(time.time() - st, 2), n_proc))
            if count_big == 0 and logging:
                f.write("One full iteration takes {0} with {1} processes".format(round(time.time() - st, 2), n_proc))
                f.write("\n")


            count_big = count_big + 1


            # visualization options:

            if show_pher:
                fig = plt.figure(figsize=(18, 12))
                ax = fig.add_subplot(111)
                t_max = np.max(t0)
                cax = ax.matshow(t0, interpolation='nearest', cmap=plt.cm.RdPu, vmin=t_min, vmax=t_max)
                plt.colorbar(cax)
                plt.title("Pheromones")
                plt.show(block=False)
                plt.close(fig)

            if show_nets:
                self.features(solution, self.GE, self.G)
            if show_plot:
                fig = plt.figure(figsize=(10, 8))
                plt.plot(np.arange(count_big), scores, 'g-')
                plt.plot(np.arange(count_big), avs, '--')
                if opt != None:
                    plt.axhline(y=opt, label="optimal solution score", c="r")
                plt.show(block=False)
                plt.close(fig)

        if save != None:
            fig = plt.figure(figsize=(10, 8))
            plt.plot(np.arange(count_big), scores, 'g-')
            plt.plot(np.arange(count_big), avs, '--')
            plt.savefig(save + ".png")
            plt.close(fig)
        if (np.abs(max_round_score - av_score) <= eps or count_small >= times) and verbose:
            print("Full convergence achieved")
        elif W>=m/3 and verbose:
            print("The algorithm did not converge, please run again")
        elif verbose:
            print("Maximal allowed number of iterations is reached")

        if (np.abs(max_round_score - av_score) <= eps or count_small >= times) and logging:
            f.write("Full convergence achieved")
            f.write("\n")
        elif W>=m/3 and logging:
            f.write("The algorithm did not converge, please run again")
            f.write("\n")
        elif logging:
            f.write("Maximal allowed number of iterations is reached")
            f.write("\n")

        # after the solutution is found we make sure to cluster patients the last time with that exact solution:
        data_new = ge[solution[0][0] + solution[0][1], :]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data_new.T)
        labels = kmeans.labels_
        patients_groups = []
        for clust in range(clusters):
            wh = np.where(labels == clust)[0]
            group_p = [patients[i] for i in wh]
            patients_groups.append(group_p)
        if np.mean(ge[best_solution[0][0], :][:, (np.asarray(patients_groups[0]) - n)]) < np.mean(
                ge[best_solution[0][1], :][:, (np.asarray(patients_groups[0]) - n)]):
            patients_groups = patients_groups[::-1]
        best_solution = [best_solution[0], patients_groups]
        if verbose:
            print("best  score: " + str(max_total_score))
        if logging:
            f.write("best  score: " + str(max_total_score))
            f.write("\n")

        if ls:
            components, sizes = self.opt_net(best_solution[0], patients_groups, self.L_g_min, self.L_g_max, self.G, self.GE, clusters)
            max_ls_score = self.score(self.G, patients_groups, components, n, m, ge, sizes, self.L_g_min, self.L_g_max)
            max_ls_score = max_ls_score[0][0] * max_ls_score[0][1] + max_ls_score[1][0] * max_ls_score[1][1]
            if max_ls_score>max_total_score:
                scores.append(max_ls_score)
                max_total_score = max_ls_score
                best_solution = [components, patients_groups]
                best_solution = [best_solution[0], patients_groups]
            if verbose:
                print("best  score after LS: " + str(max_total_score))
            if logging:
                f.write("best  score after LS: " + str(max_total_score))
                f.close()


        # # print_clusters(GE,best_solution)
        # # features(best_solution, GE,G)
        return (best_solution, [count_big, scores, avs])

    def ant_job_paral(self, GE, N, H, th, clusters, probs, a, b, cost, m, n, patients, count_big, cost_limit, L_g_min,
                      L_g_max, G, ge, ants_per_batch, pr, ss, result):
        # organising parallel distribution of work between ants batches
        max_round_score = -100
        W = 0
        av_score = 0
        for i in range(ants_per_batch):
            seed = ss[i]
            tot_score, gene_groups, patients_groups, new_scores, wars, no_int = self.ant_job(GE, N, H, th, clusters,
                                                                                             probs, a, b, cost, m, n,
                                                                                             patients, count_big,
                                                                                             cost_limit, L_g_min,
                                                                                             L_g_max, G, ge, seed)
            W = W + wars
            av_score = av_score + tot_score
            if tot_score > max_round_score:
                max_round_score = tot_score
                solution = (gene_groups, patients_groups)
                solution_big = (no_int, patients_groups)
                new_scores_best = new_scores
                s1 = new_scores_best[0][0] * new_scores_best[0][1]
                s2 = new_scores_best[1][0] * new_scores_best[1][1]
        result.put([(s1, s2), solution, solution_big, av_score / ants_per_batch, W])

    def neigborhood(self, H, n, th):
        # defines search area for each ant

        N_per_patient = []
        dim = len(H)
        for i in range(n, dim):
            if th < 0:
                N = np.where(H[i, :] > 0.001)[0]
            else:
                rad = np.mean(H[i, :]) + th * np.std(H[i, :])

                N = np.where(H[i, :] > rad)[0]
            # N = np.where(H[i,:]>0)[0]
            N_per_patient.append(N)
        return N_per_patient

    def prob_upd(self, H, t, a, b, n, th, N_per_patient):
        # updates probability
        P_per_patient = []
        dim = len(H)
        t = t.astype('float64')
        H = H.astype('float64')
        temp_t = np.power(t, a)
        temp_H = np.power(H, b)
        temp = temp_t * temp_H
        war = False
        for i in range(n, dim):
            N_temp = N_per_patient[i - n]
            P = temp[:, N_temp]
            s = np.sum(P, axis=1)
            for j in range(len(s)):
                if s[j] < 1.e-4:
                    P[j,:] = P[j,:] + 0.01
                    s[j] = np.sum(P[j,:])
                    
            sum_p = 1 / s
            sum_p = sum_p[:, None]
            P_new = P * sum_p[:np.newaxis]
            P_per_patient.append(P_new)
            zeros = np.where(~P_new.any(axis=1))[0]
            if len(zeros) != 0:
                war = True
        return (P_per_patient, war)

    def walk(self, start, Nn, P_small, cost, k, n, seed=None):
        # Initialize a random walk
        path = []
        path.append(start)
        go = True
        while go == True:
            P_new = P_small[start, :]
            # if there is any node inside the radious - keep mooving

            # transition:
            if seed != None:
                np.random.seed(seed)
            try:
                tr = np.random.choice(Nn, 1, False, p=P_new)[0]
            except ValueError:
                tr = np.random.choice(Nn, 1, False, p=np.ones(len(P_new))*(1/len(P_new)))[0]
            c = cost[start, tr]
            # if there is any cost left we keep going
            if k - c > 0:
                path.append(tr)
                start = tr
                k = k - c
            # if not we are done and we save only genes from the path
            else:
                go = False

        path = np.asarray(path)
        path = path[path < n]
        # we are saving only genes
        return (path)

    def ant_job(self, GE, N, H, th, clusters, probs, a, b, cost, m, n, patients, count_big, cost_limit, L_g_min,
                L_g_max, G, ge, seed=None):

        paths = []
        wars = 0
        # set an ant on every patient
        for w in range(m):
            k = cost_limit
            start = patients[w]
            Nn = N[w]  # neigbohood
            P_small = probs[w]
            path = self.walk(start, Nn, P_small, cost, k, n)
            paths.append(path)
        #    print("Random walks: {0}\n".format(end-st))
        data_new = ge[list(set(flatten(paths))), :]
        kmeans = KMeans(n_clusters=2).fit(data_new.T)
        labels = kmeans.labels_
        #    print("Patients clustering: {0}\n".format(end-st))

        gene_groups_set = []
        patients_groups = []
        for clust in range(clusters):
            wh = np.where(labels == clust)[0]
            group_g = [paths[i] for i in wh]
            group_g = flatten(group_g)
            gene_groups_set.append(set(group_g))
            # save only most common genes for a group
            group_p = [patients[i] for i in wh]
            patients_groups.append(group_p)

        # delete intersecting genes between groups

        I = set.intersection(*gene_groups_set)
        no_int = [list(gene_groups_set[i].difference(I)) for i in range(clusters)]
        gene_groups = no_int
        #    print("Genes clustering: {0}\n".format(end-st))

        # make sure that gene clusters correspond to patients clusters:
        if np.mean(ge[gene_groups[0], :][:, (np.asarray(patients_groups[0]) - n)]) < np.mean(
                ge[gene_groups[1], :][:, (np.asarray(patients_groups[0]) - n)]):
            patients_groups = patients_groups[::-1]
        #    print("Switch: {0}\n".format(end-st))

        gene_groups, sizes = self.clean_net(gene_groups, patients_groups, clusters, L_g_min, G, GE)
        #    print("Clean net: {0}\n".format(end-st))

        new_scores = self.score(G, patients_groups, gene_groups, n, m, ge, sizes, L_g_min, L_g_max)
        #    print("Score: {0}\n".format(end-st))

        tot_score = new_scores[0][0] * new_scores[0][1] + new_scores[1][0] * new_scores[1][1]
        return tot_score, gene_groups, patients_groups, new_scores, wars, no_int

    def pher_upd(self, t, t_min, p, scores, solution):
        t = t * (1 - p)
        t_new = np.copy(t)
        assert t_new.sum() > 0, "bad pheromone input"
        for i in range(len(solution[0])):
            group_g = solution[0][i]
            group_p = solution[1][i]
            sc = scores[i]
            # ge_score = new_scores[i][0]*10
            # ppi_score = new_scores[i][1]*10
            for g1 in group_g:
                for p1 in group_p:
                    t_new[g1, p1] = t[g1, p1] + sc
                    t_new[p1, g1] = t[p1, g1] + sc
                for g2 in group_g:
                    t_new[g1, g2] = t[g1, g2] + sc

        if t_new.sum() < 0:
            t_new = np.copy(t)
            
        t_new[t_new < t_min] = t_min
        assert t_new.sum() != 0, "Bad pheromone update"

        return (t_new)

    def score(self, G, patients_groups, gene_groups, n, m, ge, sizes, L_g_min, L_g_max):
        clusters = len(patients_groups)
        conf_matrix = np.zeros((clusters, clusters))
        conect_ppi = []
        for i in range(clusters):  # over genes
            group_g = np.asarray(gene_groups[i])
            s = sizes[i]
            if len(group_g) > 0:
                for j in range(clusters):  # over patients
                    group_p = np.asarray(patients_groups[j])
                    if len(group_p) > 0:
                        # gene epression inside the group
                        conf_matrix[i, j] = np.mean(ge[group_g, :][:, (group_p - n)])
                # ppi score
                con_ppi = 1
                if s < L_g_min:
                    con_ppi = s / L_g_min
                elif s > L_g_max:
                    con_ppi = L_g_max / s
                conect_ppi.append(con_ppi)
            else:
                conect_ppi.append(0)
        ans = []
        for i in range(clusters):
            all_ge = np.sum(conf_matrix[i, :])
            in_group = conf_matrix[i, i]
            out_group = all_ge - in_group
            ge_con = in_group - out_group
            # scaled = scaleBetween(num,0,0.5,0,1)
            ans.append((ge_con, conect_ppi[i]))

        return (ans)

    def HI_big(self, data_aco, A_new):
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  #
        H_g_to_g = (data_aco.T.corr())
        H_p_to_p = data_aco.corr()
        H_g_to_g = scaler.fit_transform(H_g_to_g)
        H_p_to_p = scaler.fit_transform(H_p_to_p)
        H_g_to_p = scaler.fit_transform(data_aco)
        H_full_up = np.concatenate([H_g_to_g, H_g_to_p], axis=1)
        H_full_down = np.concatenate([H_g_to_p.T, H_p_to_p], axis=1)
        H_full = np.concatenate([H_full_up, H_full_down], axis=0) * 10
        #    H_full[H_full < 1] = 1
        #    np.fill_diagonal(H_full, 1)
        np.fill_diagonal(H_full, 0)
        n, _ = A_new.shape
        H_small = H_full[:n, :n]
        H_small = np.multiply(H_small, A_new)
        H_full[:n, :n] = H_small
        th = np.median(H_full.mean(axis = 1))
        H_full[H_full < th] = 0
        return (H_full)

    def print_clusters(self, GE, solution):
        grouping_p = []
        p_num = list(GE.columns)
        for p in p_num:
            if p in solution[1][0]:
                grouping_p.append(1)
            else:
                grouping_p.append(2)
        grouping_p = pd.DataFrame(grouping_p, index=p_num)
        grouping_g = []
        g_num = list(GE.index)
        for g in g_num:
            if g in solution[0][0]:
                grouping_g.append(1)
            elif g in solution[0][1]:
                grouping_g.append(2)
            else:
                grouping_g.append(3)

        grouping_g = pd.DataFrame(grouping_g, index=g_num)
        species = grouping_p[0]
        lut = {1: '#A52A2A', 2: '#7FFFD4'}
        row_colors = species.map(lut)
        species = grouping_g[0]
        lut = {1: '#A52A2A', 2: '#7FFFD4', 3: '#FAEBD7'}
        col_colors = species.map(lut)
        sns.clustermap(GE.T, row_colors=row_colors, col_colors=col_colors, figsize=(15, 10))

    def features(self, solution, GE, G, pos=None):
        genes1, genes2 = solution[0]
        patients1, patients2 = solution[1]

        means1 = list(np.mean(GE[patients1].reindex(index = genes1), axis=1) - np.mean(GE[patients2].reindex(index = genes1), axis=1).values)
        means2 = list(np.mean(GE[patients1].reindex(index = genes2), axis=1) - np.mean(GE[patients2].reindex(index = genes2), axis=1).values)

        G_small = nx.subgraph(G, genes1 + genes2)

        fig = plt.figure(figsize=(15, 10))
        vmin = -2
        vmax = 2
        if pos == None:
            pos = nx.spring_layout(G_small)
        ec = nx.draw_networkx_edges(G_small, pos)
        nc1 = nx.draw_networkx_nodes(G_small, nodelist=genes1, pos=pos, node_color=means1, node_size=200, alpha=1.0,
                                     vmin=vmin, vmax=vmax, node_shape="^", cmap=plt.cm.PRGn)
        nc2 = nx.draw_networkx_nodes(G_small, nodelist=genes2, pos=pos, node_color=means2, node_size=200,
                                     alpha=1.0,
                                     vmin=vmin, vmax=vmax, node_shape="o", cmap=plt.cm.PRGn)
        nx.draw_networkx_labels(G_small, pos)
        plt.colorbar(nc1)
        plt.axis('off')

        plt.show(block=False)
        plt.close(fig)

    def clean_net(self, gene_groups, patients_groups, clusters, L_g, G, GE, d_cut=2):
        genes_components = []
        sizes = []
        for clust in range(clusters):
            group_g = gene_groups[clust]
            if clust == 0:
                not_clust = 1
            else:
                not_clust = 0
            if len(group_g) >= L_g:
                g = nx.subgraph(G, group_g)
                # we are taking only the biggest connected component
                comp_big = g.subgraph(max(nx.connected_components(g), key=len))

                # measure the degree of each node in it
                dg = dict(nx.degree(comp_big))
                # separate those with d == 1 as nodes that we can kick out potentially
                ones = [x for x in dg if dg[x] == 1]
                nodes = list(comp_big.nodes)
                size_comp = len(nodes)
                # maximum # of nodes we can kick out
                max_out = len(nodes) - L_g
                while max_out > 0:
                    # measure the difference in the expression between two groups for d == 1 nodes

                    dif = np.mean(GE[patients_groups[clust]].reindex(index = ones), axis=1) - np.mean(GE[patients_groups[not_clust]].reindex(index =ones), axis=1)
                    dif = dif.sort_values()
                    # therefore we select the nodes with d == 1 and low difference
                    ones = list(dif[dif < 1.5].index)
                    if len(ones) > 0:
                        if len(ones) <= max_out:
                            outsiders = ones
                        else:
                            outsiders = list(ones)[:max_out]

                        nodes = list(set(nodes) - set(outsiders))
                        g = nx.subgraph(G, nodes)
                        comp_big = g.subgraph(max(nx.connected_components(g), key=len))

                        dg = dict(nx.degree(comp_big))
                        if d_cut == 1:
                            ones = [x for x in dg if (dg[x] == 1)]
                        else:
                            ones = [x for x in dg if ((dg[x] == 1) or (dg[x] == d_cut))]
                        nodes = list(comp_big.nodes)
                        size_comp = len(nodes)
                        max_out = len(nodes) - L_g
                    else:
                        max_out = 0

                group_g = nodes
            elif len(group_g) > 0:
                g = nx.subgraph(G, group_g)
                try:
                    comp_big = comp_big = g.subgraph(max(nx.connected_components(g), key=len))
                    nodes = list(comp_big.nodes)
                    size_comp = len(nodes)
                except ValueError:
                    size_comp = 0



            else:
                size_comp = 0

            genes_components.append(group_g)
            sizes.append(size_comp)
        return genes_components, sizes

    def opt_net(self, gene_groups, patients_groups, L_min, L_max, G, GE, clusters):
        genes_components = []
        sizes = []
        for clust in range(clusters):
            not_clust = int(clust == 0)
            g = nx.subgraph(G, gene_groups[clust])
            # we are taking only the biggest connected component
            try:
                nodes0 =  g.subgraph(max(nx.connected_components(g), key=len))
                nodes0 =  list(nodes0.nodes)
            except ValueError:
                nodes0 = []
                nodes = gene_groups[clust]
                size = 0


            if len(nodes0) != 0:
                score0 = self.new_score(GE, patients_groups[clust], patients_groups[not_clust], nodes0)
                if score0 >0 :
                    move = True
                    while move:
                        results = {**self.insertion(L_max, nodes0, G, GE, patients_groups, clust),
                                   **self.deletion(L_min, nodes0, G, GE, patients_groups, clust),
                                   **self.subst(L_max, nodes0, G, GE, patients_groups, clust)}

                        if len(results) != 0:
                            action = max(results.items(), key=operator.itemgetter(1))[0]
                            score1 = results[action]

                            delta = score0 - score1
                            if delta < 0:  # move on
                                # print(action)
                                # print(score1)
                                nodes = self.do_action_nodes(action, nodes0)
                                nodes0 = nodes
                                score0 = score1
                                size = len(nodes)

                            else:  # terminate if no improvement
                                move = False
                                nodes = nodes0
                                size = len(nodes)
                        else:
                            nodes = nodes0
                            move = False
                            size = len(nodes)
                else:
                    nodes = gene_groups[clust]
                    size = len(nodes0)
            group_g = nodes
            size_comp = size
            genes_components.append(group_g)
            sizes.append(size_comp)
        return genes_components, sizes

    def is_removable(self, nodes, G, node=None):
        g = nx.subgraph(G, nodes)
        is_rem = dict()
        if node != None:
            nodes = [node]
        for node in nodes:
            g_small = g.copy()
            g_small.remove_node(node)
            n = nx.number_connected_components(g_small)
            if n != 1:
                is_rem[node] = False
            else:
                is_rem[node] = True
        return is_rem

    def get_candidates(self, nodes, G):
        subst_candidates = flatten([[n for n in G.neighbors(x)] for x in nodes])
        subst_candidates = set(subst_candidates).difference(set(nodes))
        return subst_candidates

    def insertion(self, L_max, nodes, G, GE, patients_groups, clust):
        results = dict()
        no_clust = int(clust == 0)
        size = len(nodes)
        if size < L_max:
            candidates = self.get_candidates(nodes, G)
            for c in candidates:
                nodes_new = nodes + [c]
                sc = self.new_score(GE, patients_groups[clust], patients_groups[no_clust], nodes_new)
                results["i_" + str(c)] = sc
        return results

    def deletion(self, L_min, nodes, G, GE, patients_groups, clust):
        results = dict()
        size = len(nodes)
        no_clust = int(clust == 0)

        if size > L_min:
            rem = self.is_removable(nodes, G)
            for node in nodes:
                if rem[node]:
                    nodes_new = list(set(nodes).difference(set([node])))
                    sc = self.new_score(GE, patients_groups[clust], patients_groups[no_clust], nodes_new)
                    results["d_" + str(node)] = sc
        return results

    def subst(self, L_max, nodes, G, GE, patients_groups, clust):
        results = dict()
        size = len(nodes)
        no_clust = int(clust == 0)

        if size < L_max:
            candidates = self.get_candidates(nodes, G)
            for node in nodes:
                for c in candidates:
                    nodes_new = nodes + [c]
                    rem = self.is_removable(nodes_new, G, node)
                    if rem[node]:
                        nodes_new = list(set(nodes_new).difference(set([node])))
                        sc = self.new_score(GE, patients_groups[clust], patients_groups[no_clust], nodes_new)
                        results["s_" + str(node) + "_" + str(c)] = sc
                    else:
                        pass
        return results

    def new_score(self, GE, pg, npg, gg):
        dif = np.mean(np.mean(GE[pg].reindex(index = gg), axis=1)) - np.mean(np.mean(
            GE[npg].reindex(index = gg), axis=1))
        return dif

    def do_action_nodes(self, action, nodes):
        if len(action.split("_")) == 2:  ##inserion or delition
            act, node = action.split("_")
            node = int(node)
            if act == "i":
                nodes = nodes + [node]
            else:
                nodes = list(set(nodes).difference(set([node])))
        else:  # substitution
            act, node, cand = action.split("_")
            node = int(node)
            cand = int(cand)
            nodes = nodes + [cand]
            nodes = list(set(nodes).difference(set([node])))
        return nodes
