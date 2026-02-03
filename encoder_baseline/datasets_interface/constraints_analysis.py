
from datasets_interface.maven_ere import maven_ere

def ikt(d, k, t):
    if not k in d:
        d[k]=t
    return

def ikn(d, k, t=dict, n='n', i=0):
    if not k in d:
        d[k]=t()
        d[k][n]=i
    return

def list_to_dict(rel, rel_typ=list(range(100))):
    res = dict()
    for r in rel:
        ikt(res, r[0][0], dict())
        ikt(res[r[0][0]], r[0][1], list())
        res[r[0][0]][r[0][1]].append(rel_typ[r[1]])
    return res

def manifold(evts, dr, tr, n='n'):
    """
    I am a real Ph.D. student
    """
    for evt1, gr1 in evts.items():
        for evt2, rels12 in gr1.items():
            #print(evt1, gr1, evt2, rels12)
            for rel12 in rels12:
                ikn(dr, rel12)
                ikn(tr, rel12)
                dr[rel12][n]+=1
                tr[rel12][n]+=1
                for rel12b in rels12:
                    if rel12b!=rel12:
                        ikt(dr[rel12], rel12b, [0, 0])
                        dr[rel12][rel12b][0]+=1
                if evt2 in evts:
                    for evt3, rels23 in evts[evt2].items():
                        for rel23 in rels23:
                            ikn(tr[rel12], rel23)
                            tr[rel12][rel23][n] += 1
                            if evt3==evt1:
                                ikt(dr[rel12], rel23, [0, 0])
                                dr[rel12][rel23][1]+=1
                            elif evt3 in evts[evt1]:
                                for rel13 in evts[evt1][evt3]:
                                    ikt(tr[rel12][rel23], rel13, [0, 0])
                                    tr[rel12][rel23][rel13][0]+=1
                            elif evt3 in evts and evt1 in evts[evt3]:
                                for rel31 in evts[evt3][evt1]:
                                    ikt(tr[rel12][rel23], rel31, [0, 0])
                                    tr[rel12][rel23][rel31][1]+=1
    return dr, tr

def const_search(dataset):
    """
    To use to extract constrains on relation sequences for 2 (dr) and 3 (tr)
    """
    dr = dict()
    tr = dict()
    for section in ['train', 'valid', 'eval']:
        dataset.set_dataset(section)
        for idx, data in enumerate(dataset.rel_list()):
            print(f"Attending data {idx} in {section}", end="\r")
            manifold(list_to_dict(data, dataset.ere_types), dr, tr)
        print("")
    return dr, tr

def const_analysis(tr, lam=0.99, sup=100):
    """
    forward triple, backward triple
    """
    f = []
    b = []
    for i in tr:
        if i=='n':
            continue
        for j in tr[i]:
            if j=='n':
                continue
            for k in tr[i][j]:
                if k=='n':
                    continue
                if sum(tr[i][j][k]) > sup:
                    """
                    if tr[i][j][k][0]/tr[i][j]['n']>lam:
                        f.append((i, j, k))
                    """
                    if tr[i][j][k][1]/tr[i][j]['n']>lam:
                        b.append((i, j, k))
    for i, j, k in f:
        print(i, j, "->", k)
    for i, j, k in b:
        print(i, j, "<-", k)
    return f, b

def backward_cons(dr):
    total = 0
    rev = 0
    seek = dict()
    for i in dr:
        if i=='n':
            continue
        for j in dr[i]:
            if j=='n':
                continue
            rev += dr[i][j][1]
            total += rev + dr[i][j][0]
            if dr[i][j][1]>0:
                seek[(i, j)] = dr[i][j][1]
    return total, rev, seek
