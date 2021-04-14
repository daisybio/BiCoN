import multiprocessing as mp
from multiprocessing import Process, Queue
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.cluster import KMeans
import gc
import operator
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category= RuntimeWarning)
warnings.filterwarnings("ignore", category= UserWarning)

sns.set(color_codes=True)
flatten = lambda l: [item for sublist in l for item in sublist]


class BiCoN:
    def __init__(self, GE, G, L_g_min, L_g_max):
        """
        :param GE: pandas data frame with gene expression data. Genes are rows, patients - columns
        :param G: networkX graph with a network
        :param L_g_min: minimal number of genes in one subnetwork
        :param L_g_max: maximal number of genes in one subnetwork
        """
        self.GE = GE
        self.ge = self.GE.values

        self.G = G
        self.L_g_min = L_g_min
        self.L_g_max = L_g_max
        self.A = nx.adj_matrix(self.G).todense()
        self.H = self.HI_big().astype(np.short)
        self.n, self.m = GE.shape
        self.patients = np.arange(self.n, self.n + self.m)
        self.cost = self.H / 10
        self.cost = 0.1 + np.max(self.cost) - self.cost
        self.clusters = 2

    def run_search(self, n_proc=1, a=1, b=1, K=20, evaporation=0.5, th=1, eps=0.02,
                   times=6, cost_limit=5, max_iter=100, ls=True, opt=None,
                   show_plot=False, save=None, verbose=True, logging=False):

        """
        Parallel implementation of network constrained bi-clustering

        :param n_proc: number of processes that should be used (default 1)
        :param a: pheromone significance (default 1 - does not
        need to be changed)
        :param b: heuristic information significance (default 1 - does not need to be changed)
        :param K: number of ants (less ants - less space exploration. Usually set between 20 and 50, default - 20)
        :param evaporation: the rate at which pheromone evaporates (default 0.5)
        :param th: similarity threshold (default 1 - does not need to be changed)
        :param eps: conservative convergence criteria: score_max - score_min < eps (default-
        0.02)
        :param times: allows faster convergence: stop if the maximum so far was reached more than x times
        :param cost_limit: defines the radius of the search for ants (default 5)
        :param max_iter: maximum number of interactions allowed (default 200)
        :param ls:
        :param show_plot: True if convergence plots should be shown
        :param save:  set an output file name
        :param verbose: set true to print information about the algorithm's progress
        :param logging: set true to save information about algorithm's progress to bicon_logging.txt file
        :return:
        best_solution: best solution achieved
        [count_big, scores, avs]: [total iteration count, scores vector, average score]
        """
        assert n_proc > 0, "Set a correct number for n_proc, right now the value is {0}".format(n_proc)
        assert n_proc <= mp.cpu_count() - 1, 'n_proc should not exceed {0}. The value of n_proc was: {1}'.format(
            mp.cpu_count(), n_proc)
        assert n_proc <= K, 'Number of ants (K) can not be lower as number ' \
                            'of processes, please set higher K or lower ' \
                            'n_proc '
        # TODO: rewrite functions such that they all could use numpy matrices
        # determination of search radius for each patient
        N = self.neigborhood(th)
        # inner patients IDs
        # cost of transitions for ants
        f = None
        solution = None
        solution_big = None
        solution_big_best = None
        best_solution = None

        # stores all scores
        scores = []
        avs = []
        count_big = 0
        max_total_score = -100
        n1, n2 = (0, 0)
        max_round_score = -100
        av_score = 0
        # initial pheromone level set to a maximal possible level (5 standard deviations)
        t0 = np.ones((self.n + self.m, self.n + self.m)) * 5
        t0 = t0.astype(np.short)
        t_min = 0
        # initial probabilities
        st = time.time()
        probs, war = self.prob_upd(t0, a, b, N)
        while war and th > -1:
            th = th - 0.5
            N = self.neigborhood(th)
            probs, war = self.prob_upd(t0, a, b, N)

        end = time.time()
        # flag tracks when the score stops improving and terminates the optimization as convergence is reached
        score_change = []
        p_time = str(round(end - st, 3))
        if verbose:
            print("Runtime statistics:")
            print("###############################################################")
            print("the joint graph has " + str(self.n + self.m) + " nodes")

            print("probability update takes " + p_time)
        if logging:
            f = open("bicon_logs.txt", "w")
            f.write("Runtime statistics:")
            f.write("\n")
            f.write("###############################################################")
            f.write("\n")
            f.write("the joint graph has " + str(self.n + self.m) + " nodes")
            f.write("\n")
            f.write("probability update takes " + p_time)
            f.write("\n")

        count_small = 0
        W = 0
        # termination if the improvements are getting too small
        while np.abs(max_round_score - av_score) > eps and count_small < times and count_big < max_iter and (
                W < self.m / 3):
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
                        N, probs, cost_limit, ants_per_batch, ss, result,))
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
                            solution = res[1]
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
                for i in tqdm(range(K)):
                    # for each ant
                    tot_score, gene_groups, patients_groups, new_scores, wars, no_int = self.ant_job(N, probs, cost_limit)
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
            probs, _ = self.prob_upd(t0, a, b, N)
            if count_big == 0 and verbose:
                print("One full iteration takes {0} with {1} processes".format(round(time.time() - st, 2), n_proc))
            if count_big == 0 and logging:
                f.write("One full iteration takes {0} with {1} processes".format(round(time.time() - st, 2), n_proc))
                f.write("\n")

            count_big = count_big + 1

            # visualization options:

            if show_plot:
                fig = plt.figure(figsize=(10, 8))
                plt.plot(np.arange(count_big), scores, 'g-')
                plt.plot(np.arange(count_big), avs, '--')
                if opt is not None:
                    plt.axhline(y=opt, label="optimal solution score", c="r")
                plt.show(block=False)
                plt.close(fig)

        if save is not None:
            fig = plt.figure(figsize=(10, 8))
            plt.plot(np.arange(count_big), scores, 'g-')
            plt.plot(np.arange(count_big), avs, '--')
            plt.savefig(save + ".png")
            plt.close(fig)
        if (np.abs(max_round_score - av_score) <= eps or count_small >= times) and verbose:
            print("Full convergence achieved")
        elif W >= self.m / 3 and verbose:
            print("The algorithm did not converge, please run again")
        elif verbose:
            print("Maximal allowed number of iterations is reached")

        if (np.abs(max_round_score - av_score) <= eps or count_small >= times) and logging:
            f.write("Full convergence achieved")
            f.write("\n")
        elif W >= self.m / 3 and logging:
            f.write("The algorithm did not converge, please run again")
            f.write("\n")
        elif logging:
            f.write("Maximal allowed number of iterations is reached")
            f.write("\n")

        # after the solutution is found we make sure to cluster patients the last time with that exact solution:
        data_new = self.ge[solution[0][0] + solution[0][1], :]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data_new.T)
        labels = kmeans.labels_
        patients_groups = []
        for clust in range(self.clusters):
            wh = np.where(labels == clust)[0]
            group_p = [self.patients[i] for i in wh]
            patients_groups.append(group_p)
        if np.mean(self.ge[best_solution[0][0], :][:, (np.asarray(patients_groups[0]) - self.n)]) < np.mean(
                self.ge[best_solution[0][1], :][:, (np.asarray(patients_groups[0]) - self.n)]):
            patients_groups = patients_groups[::-1]
        best_solution = [best_solution[0], patients_groups]
        if verbose:
            print("best  score: " + str(max_total_score))
        if logging:
            f.write("best  score: " + str(max_total_score))
            f.write("\n")

        if ls:
            components, sizes = self.opt_net(best_solution[0], patients_groups)
            max_ls_score = self.score(patients_groups, components, sizes)
            max_ls_score = max_ls_score[0][0] * max_ls_score[0][1] + max_ls_score[1][0] * max_ls_score[1][1]
            if max_ls_score > max_total_score:
                scores.append(max_ls_score)
                max_total_score = max_ls_score
                best_solution = [components, patients_groups]
                best_solution = [best_solution[0], patients_groups]
            if verbose:
                print("best  score after LS: " + str(max_total_score))
            if logging:
                f.write("best  score after LS: " + str(max_total_score))
                f.close()
        return best_solution, [count_big, scores, avs]

    def ant_job_paral(self, N, probs, cost_limit, ants_per_batch, ss, result):
        """
        :param N: allowed neigbourhood for each patient node
        :param probs: list of probability matrices
        :param cost_limit: cost limt parameter
        :param ants_per_batch: # of ants ber batch
        :param ss: seeds
        :param result: queue
        :return:
        result queue
        """
        # organising parallel distribution of work between ants batches
        max_round_score = -100
        W = 0
        av_score = 0
        s1 = 0
        s2 = 0
        solution_big = None
        solution = None
        for i in range(ants_per_batch):
            seed = ss[i]
            tot_score, gene_groups, patients_groups, new_scores, wars, no_int = self.ant_job(N, probs, cost_limit)
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

    def neigborhood(self, th):
        """
        Defines search area for each ant
        :param th: similarity threshold
        :return: N_per_patient - allowed search area for each patient
        """
        #

        N_per_patient = []
        dim = len(self.H)
        for i in range(self.n, dim):
            if th < 0:
                N = np.where(self.H[i, :] > 0.001)[0]
            else:
                rad = np.mean(self.H[i, :]) + th * np.std(self.H[i, :])

                N = np.where(self.H[i, :] > rad)[0]
            N_per_patient.append(N)
        return N_per_patient

    def prob_upd(self, t, a, b, N):
        # updates probability
        P_per_patient = []
        dim = len(self.H)
        t = t.astype('float64')
        H = self.H.astype('float64')
        temp_t = np.power(t, a)
        temp_H = np.power(H, b)
        temp = temp_t * temp_H
        war = False
        for i in range(self.n, dim):
            N_temp = N[i - self.n]
            P = temp[:, N_temp]
            s = np.sum(P, axis=1)
            for j in range(len(s)):
                if s[j] < 1.e-4:
                    P[j, :] = P[j, :] + 0.01
                    s[j] = np.sum(P[j, :])

            sum_p = 1 / s
            sum_p = sum_p[:, None]
            P_new = P * sum_p[:np.newaxis]
            P_per_patient.append(P_new)
            zeros = np.where(~P_new.any(axis=1))[0]
            if len(zeros) != 0:
                war = True
        return P_per_patient, war

    def walk(self, start, Nn, P_small, cost_limit, seed=None):
        """
        Initialize a random walk
        :param start: start patient node
        :param Nn: allowed seqrch space for the patient
        :param P_small: probability matrix for the patient
        :param cost_limit: cost limit
        :param seed: seed
        :return: random walk  path
        """
        path = [start]
        go = True
        k = cost_limit
        while go:
            P_new = P_small[start, :]
            # if there is any node inside the radious - keep mooving

            # transition:
            if seed is not None:
                np.random.seed(seed)
            try:
                tr = np.random.choice(Nn, 1, False, p=P_new)[0]
            except ValueError:
                tr = np.random.choice(Nn, 1, False, p=np.ones(len(P_new)) * (1 / len(P_new)))[0]
            c = self.cost[start, tr]
            # if there is any cost left we keep going
            if k - c > 0:
                path.append(tr)
                start = tr
                k = k - c
            # if not we are done and we save only genes from the path
            else:
                go = False

        path = np.asarray(path)
        path = path[path < self.n]
        # we are saving only genes
        return path

    def ant_job(self, N, probs, cost_limit):
        """
        Aggregates random walks to get to buclestering results
        :param N: list of neigbourhoods for every patient
        :param probs: list of individual probability matrices for every patient
        :param cost_limit: search limiting parameter
        :return:
        tot_score: score of the solution
        gene_groups: gene biclusters
        patients_groups: patients biclusters
        new_scores: score of the solution (detailed)
        wars: number of warnings
        no_int: gene biclusters to upgrade

        """
        paths = []
        wars = 0
        # set an ant for every patient
        for w in range(self.m):
            start = self.patients[w]
            Nn = N[w]  # neighborhood
            P_small = probs[w]
            path = self.walk(start, Nn, P_small, cost_limit)
            paths.append(path)
        data_new = self.ge[list(set(flatten(paths))), :]
        kmeans = KMeans(n_clusters=2).fit(data_new.T)
        labels = kmeans.labels_
        #    print("Patients clustering: {0}\n".format(end-st))

        gene_groups_set = []
        patients_groups = []
        for clust in range(self.clusters):
            wh = np.where(labels == clust)[0]
            group_g = [paths[i] for i in wh]
            group_g = flatten(group_g)
            gene_groups_set.append(set(group_g))
            # save only most common genes for a group
            group_p = [self.patients[i] for i in wh]
            patients_groups.append(group_p)

        # delete intersecting genes between groups

        I = set.intersection(*gene_groups_set)
        no_int = [list(gene_groups_set[i].difference(I)) for i in range(self.clusters)]
        gene_groups = no_int
        #    print("Genes clustering: {0}\n".format(end-st))

        # make sure that gene clusters correspond to patients clusters:
        if np.mean(self.ge[gene_groups[0], :][:, (np.asarray(patients_groups[0]) - self.n)]) < np.mean(
                self.ge[gene_groups[1], :][:, (np.asarray(patients_groups[0]) - self.n)]):
            patients_groups = patients_groups[::-1]
        gene_groups, sizes = self.clean_net(gene_groups, patients_groups, self.clusters)

        new_scores = self.score(patients_groups, gene_groups, sizes)

        tot_score = new_scores[0][0] * new_scores[0][1] + new_scores[1][0] * new_scores[1][1]
        return tot_score, gene_groups, patients_groups, new_scores, wars, no_int

    @staticmethod
    def pher_upd(t, t_min, p, scores, solution):
        """
        Pheromone update
        :param t: old pheromone matrix
        :param t_min: minimal allowed value
        :param p: evaporation
        :param scores: scores for the update
        :param solution: solution to reinforce
        :return: new pheromone matrix
        """
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

        return t_new

    def score(self, patients_groups, gene_groups, sizes):
        """
        Objective function
        :param patients_groups: patient clusters
        :param gene_groups: gene clusters
        :param sizes: sizes of connected component in gene biclusters
        :return: score
        """
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
                        conf_matrix[i, j] = np.mean(self.ge[group_g, :][:, (group_p - self.n)])
                # ppi score
                con_ppi = 1
                if s < self.L_g_min:
                    con_ppi = s / self.L_g_min
                elif s > self.L_g_max:
                    con_ppi = self.L_g_max / s
                conect_ppi.append(con_ppi)
            else:
                conect_ppi.append(0)
        ans = []
        for i in range(clusters):
            all_ge = np.sum(conf_matrix[i, :])
            in_group = conf_matrix[i, i]
            out_group = all_ge - in_group
            ge_con = in_group - out_group
            ans.append((ge_con, conect_ppi[i]))

        return ans

    def HI_big(self):
        """
        Computes heuristic information matrix
        :return:
        """
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  #
        H_g_to_g = (self.GE.T.corr())
        H_p_to_p = self.GE.corr()
        H_g_to_g = scaler.fit_transform(H_g_to_g)
        H_p_to_p = scaler.fit_transform(H_p_to_p)
        H_g_to_p = scaler.fit_transform(self.GE)
        H_full_up = np.concatenate([H_g_to_g, H_g_to_p], axis=1)
        H_full_down = np.concatenate([H_g_to_p.T, H_p_to_p], axis=1)
        H_full = np.concatenate([H_full_up, H_full_down], axis=0) * 10
        np.fill_diagonal(H_full, 0)
        n, _ = self.A.shape
        H_small = H_full[:n, :n]
        H_small = np.multiply(H_small, self.A)
        H_full[:n, :n] = H_small
        th = np.median(H_full.mean(axis=1))
        H_full[H_full < th] = 0
        return H_full

    def clean_net(self, gene_groups, patients_groups, d_cut=2):
        """
        Refines gene biclusters after every iteration
        :param gene_groups: gene biclusters
        :param patients_groups: patient biclusters
        :param d_cut:
        :return:
        genes_components: refined gene biclusters
        sizes: sizes of the largest connected components for each bicluster
        """
        genes_components = []
        sizes = []
        for clust in range(self.clusters):
            group_g = gene_groups[clust]
            if clust == 0:
                not_clust = 1
            else:
                not_clust = 0
            if len(group_g) >= self.L_g_min:
                g = nx.subgraph(self.G, group_g)
                # we are taking only the biggest connected component
                comp_big = g.subgraph(max(nx.connected_components(g), key=len))

                # measure the degree of each node in it
                dg = dict(nx.degree(comp_big))
                # separate those with d == 1 as nodes that we can kick out potentially
                ones = [x for x in dg if dg[x] == 1]
                nodes = list(comp_big.nodes)
                size_comp = len(nodes)
                # maximum # of nodes we can kick out
                max_out = len(nodes) - self.L_g_min
                while max_out > 0:
                    # measure the difference in the expression between two groups for d == 1 nodes

                    dif = np.mean(self.GE[patients_groups[clust]].reindex(index=ones), axis=1) - \
                          np.mean(self.GE[patients_groups[not_clust]].reindex(index=ones), axis=1)
                    dif = dif.sort_values()
                    # therefore we select the nodes with d == 1 and low difference
                    ones = list(dif[dif < 1.5].index)
                    if len(ones) > 0:
                        if len(ones) <= max_out:
                            outsiders = ones
                        else:
                            outsiders = list(ones)[:max_out]

                        nodes = list(set(nodes) - set(outsiders))
                        g = nx.subgraph(self.G, nodes)
                        comp_big = g.subgraph(max(nx.connected_components(g), key=len))

                        dg = dict(nx.degree(comp_big))
                        if d_cut == 1:
                            ones = [x for x in dg if (dg[x] == 1)]
                        else:
                            ones = [x for x in dg if ((dg[x] == 1) or (dg[x] == d_cut))]
                        nodes = list(comp_big.nodes)
                        size_comp = len(nodes)
                        max_out = len(nodes) - self.L_g_min
                    else:
                        max_out = 0

                group_g = nodes
            elif len(group_g) > 0:
                g = nx.subgraph(self.G, group_g)
                try:
                    comp_big = g.subgraph(max(nx.connected_components(g), key=len))
                    nodes = list(comp_big.nodes)
                    size_comp = len(nodes)
                except ValueError:
                    size_comp = 0
            else:
                size_comp = 0

            genes_components.append(group_g)
            sizes.append(size_comp)
        return genes_components, sizes

    def opt_net(self, gene_groups, patients_groups):
        """
        Local search for the final solution refinement
        :param gene_groups: gene biclusters
        :param patients_groups: patient biclusters
        :return:
        genes_components: refined gene biclusters
        sizes: sizes of the largest connected components for each bicluster
        """
        genes_components = []
        sizes = []
        for clust in range(self.clusters):
            not_clust = int(clust == 0)
            g = nx.subgraph(self.G, gene_groups[clust])
            # we are taking only the biggest connected component
            try:
                nodes0 = g.subgraph(max(nx.connected_components(g), key=len))
                nodes0 = list(nodes0.nodes)
            except ValueError:
                nodes0 = []
                nodes = gene_groups[clust]
                size = 0

            if len(nodes0) != 0:
                score0 = self.new_score(patients_groups[clust], patients_groups[not_clust], nodes0)
                if score0 > 0:
                    move = True
                    while move:
                        results = {**self.insertion(nodes0, patients_groups, clust),
                                   **self.deletion(nodes0, patients_groups, clust),
                                   **self.subst(nodes0, patients_groups, clust)}

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

    def is_removable(self, nodes, node=None):
        """
        Checks if a node can be removed withut breaking an induced subgraph
        :param nodes: node set for the induced subgraph building
        :param node: a particular node to check (default - all nodes will be checked)
        :return:
        dictionary that indicates if a node can be removed without breaking the subgraph
        """
        g = nx.subgraph(self.G, nodes)
        is_rem = dict()
        if node is not None:
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

    def get_candidates(self, nodes):
        """
        Possible substitution candidates for nodes in the induced subgraph
        :param nodes: node set for the induced subgraph building
        :return: list of possible substitutions
        """
        subst_candidates = flatten([[n for n in self.G.neighbors(x)] for x in nodes])
        subst_candidates = set(subst_candidates).difference(set(nodes))
        return subst_candidates

    def insertion(self, nodes, patients_groups, clust):
        """
        Insertion candidates
        :param nodes: node set for the induced subgraph building
        :param patients_groups: patient biclusters
        :param clust: cluster number
        :return: scored solutions
        """
        results = dict()
        no_clust = int(clust == 0)
        size = len(nodes)
        if size < self.L_g_max:
            candidates = self.get_candidates(nodes)
            for c in candidates:
                nodes_new = nodes + [c]
                sc = self.new_score(patients_groups[clust], patients_groups[no_clust], nodes_new)
                results["i_" + str(c)] = sc
        return results

    def deletion(self, nodes, patients_groups, clust):
        """
        Deletion candidates
        :param nodes: node set for the induced subgraph building
        :param patients_groups: patient biclusters
        :param clust: cluster number
        :return: scored solutions
        """
        results = dict()
        size = len(nodes)
        no_clust = int(clust == 0)

        if size > self.L_g_min:
            rem = self.is_removable(nodes)
            for node in nodes:
                if rem[node]:
                    nodes_new = list(set(nodes).difference({node}))
                    sc = self.new_score(patients_groups[clust], patients_groups[no_clust], nodes_new)
                    results["d_" + str(node)] = sc
        return results

    def subst(self, nodes, patients_groups, clust):
        """
        Substitution candidates
        :param nodes: node set for the induced subgraph building
        :param patients_groups: patient biclusters
        :param clust: cluster number
        :return: scored solutions
        """
        results = dict()
        size = len(nodes)
        no_clust = int(clust == 0)

        if size < self.L_g_max:
            candidates = self.get_candidates(nodes)
            for node in nodes:
                for c in candidates:
                    nodes_new = nodes + [c]
                    rem = self.is_removable(nodes_new, node)
                    if rem[node]:
                        nodes_new = list(set(nodes_new).difference({node}))
                        sc = self.new_score(patients_groups[clust], patients_groups[no_clust], nodes_new)
                        results["s_" + str(node) + "_" + str(c)] = sc
                    else:
                        pass
        return results

    def new_score(self, pg, npg, gg):
        """
        Final score computation
        :param pg: patient bicluster 1
        :param npg: patient bicluster 2
        :param gg: gene set
        :return: score
        """
        dif = np.mean(np.mean(self.GE[pg].reindex(index=gg), axis=1)) - np.mean(np.mean(
            self.GE[npg].reindex(index=gg), axis=1))
        return dif

    @staticmethod
    def do_action_nodes(action, nodes):
        """
        Modify the current node set
        :param action: the action to be taken
        :param nodes: nodes set
        :return: updated nodes
        """
        if len(action.split("_")) == 2:  ##inserion or delition
            act, node = action.split("_")
            node = int(node)
            if act == "i":
                nodes = nodes + [node]
            else:
                nodes = list(set(nodes).difference({node}))
        else:  # substitution
            act, node, cand = action.split("_")
            node = int(node)
            cand = int(cand)
            nodes = nodes + [cand]
            nodes = list(set(nodes).difference({node}))
        return nodes
