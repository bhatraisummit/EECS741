from utils import *

if __name__ == '__main__':
    # PART I
    sim_mat = create_sim_matrix('I')
    gen, imp = gen_imp_score(sim_mat)
    print('-' * 23, 'Snippet for Part I', '-' * 23, '\n')
    print(sim_mat[:9, :9])
    print()
    print('-' * 23, "d'index for Part I", '-' * 23)
    print(round (d_index(gen, imp), 2))

    # PART II
    sim_mat = create_sim_matrix('II')
    gen, imp = gen_imp_score(sim_mat)
    print('-' * 23, 'Snippet for Part II', '-' * 23)
    print(sim_mat[:9, :9])
    print()
    print('-' * 23, "d'index for Part II", '-' * 23)
    print(round(d_index(gen, imp), 2))
