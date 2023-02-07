import numpy as np
from scipy.stats import norm
import scipy.sparse as sparse



def calc_Ks(S, s, h, vec=True):
    # calculate kernel weight
    # S is either a scalar or vector
    # s is the state
    # h is bandwidth
    # if vec is False, return a matrix whose diag is Ks
    if vec:
        return 1/h * norm.pdf((S - s)/h)
    else:
        return np.diag(1/h * norm.pdf((S - s)/h))


def calc_Ts(S, s, h, default=True):
    # calculate the sum of kernel weights
    if default:
        return np.sum(calc_Ks(S, s, h))
    else:
        return S.size


def estimate(X, S, s, h, k, transpose=False, return_e=False):
    # estimate the state-varying factor model
    N, T = X.shape

    Ks = calc_Ks(S, s, h, vec=False)
    Ts = calc_Ts(S, s, h)
    if transpose is False:
        V, Fhat = sparse.linalg.eigsh(np.sqrt(Ks).dot(X.T).dot(X).dot(np.sqrt(Ks)) / (N * Ts), k)
        V = V[range(k - 1, -1, -1)]
        Fhat = Fhat[:, range(k - 1, -1, -1)]
        Fhat = np.sqrt(Ts) * Fhat
        Lambdahat = X.dot(np.sqrt(Ks)).dot(Fhat) / Ts
        if return_e:
            ehat = X.dot(np.sqrt(Ks)) - Lambdahat.dot(Fhat.T)
            return Fhat, Lambdahat, V, ehat
        else:
            return Fhat, Lambdahat, V
    else:
        V, Lambdahat = sparse.linalg.eigsh(X.dot(Ks).dot(X.T) / (N * Ts), k)
        V = V[range(k - 1, -1, -1)]
        Lambdahat = Lambdahat[:, range(k - 1, -1, -1)]
        Lambdahat = np.sqrt(N) * Lambdahat
        Fhat = np.sqrt(Ks).dot(X.T).dot(Lambdahat) / N
        if return_e:
            ehat = X.dot(np.sqrt(Ks)) - Lambdahat.dot(Fhat.T)
            return Fhat, Lambdahat, V, ehat
        else:
            return Fhat, Lambdahat, V



def calc_rho(G1, G2, G3, G4):
    rho = np.matrix.trace(np.linalg.inv(G1).dot(G2).dot(np.linalg.inv(G4)).dot(G3))
    return rho


def estimate_generalized_correlation(X, S, s1, s2, h, k=1):
    N, T = X.shape
    Fhat1, Lambdahat1, Vhat1, ehat1 = estimate(X, S, s1, h, k, transpose=True, return_e=True)
    Fhat2, Lambdahat2, Vhat2, ehat2 = estimate(X, S, s2, h, k, transpose=True, return_e=True)

    SigmaLamlist = [(Lambdahat1.T.dot(Lambdahat1) / N).reshape((k, k)),
                    (Lambdahat1.T.dot(Lambdahat2) / N).reshape((k, k)),
                    (Lambdahat2.T.dot(Lambdahat2) / N).reshape((k, k))]

    rhohat = calc_rho(SigmaLamlist[0], SigmaLamlist[1], SigmaLamlist[1],
                      SigmaLamlist[2])

    return rhohat


def inference_generalized_correlation(X, S, s1, s2, h, k=1):
    N, T = X.shape
    Fhat1, Lambdahat1, Vhat1, ehat1 = estimate(X, S, s1, h, k, transpose=True, return_e=True)
    Fhat2, Lambdahat2, Vhat2, ehat2 = estimate(X, S, s2, h, k, transpose=True, return_e=True)
    if np.corrcoef(Lambdahat1.T, Lambdahat2.T)[0, 1] < 0:
        Lambdahat2 = -Lambdahat2
        Fhat2 = -Fhat2

    # point estimator
    SigmaLamlist = [(Lambdahat1.T.dot(Lambdahat1) / N).reshape((k, k)),
                    (Lambdahat1.T.dot(Lambdahat2) / N).reshape((k, k)),
                    (Lambdahat2.T.dot(Lambdahat2) / N).reshape((k, k))]
    rhohat = calc_rho(SigmaLamlist[0], SigmaLamlist[1], SigmaLamlist[1],
                      SigmaLamlist[2])

    # variance estimator
    Fhatlist = [Fhat1, Fhat2]
    Lamhatlist = [Lambdahat1, Lambdahat2]
    Vlist = [np.diag(Vhat1), np.diag(Vhat2)]
    ehatlist = [ehat1, ehat2]
    slist = [s1, s2]
    sigmaBBhat = calc_sigmaBBhat(Fhatlist, Lamhatlist, ehatlist, S, slist, h)
    Dhat = calc_Dhat(Vlist, SigmaLamlist)

    G1, G2, G3, G4 = calc_G(Lamhatlist)
    xihat = calc_xi(G1, G2, G3, G4)
    rhohat_var = xihat.T.dot(Dhat).dot(sigmaBBhat).dot(Dhat.T).dot(xihat)
    rhohat_sd = np.sqrt(rhohat_var)

    # bias-correction estimator
    G1, G2, G3, G4 = calc_G(Lamhatlist)
    xihat = calc_xi(G1, G2, G3, G4)
    bhat = calc_b(Fhatlist, Lamhatlist, ehatlist, Vlist, S, slist, h)

    # calculate test statistics
    rho_test_stats = (rhohat - k - xihat.T.dot(bhat)) / rhohat_sd * np.sqrt(N * T * h)
    rho_test_stats = rho_test_stats[0, 0]

    return rho_test_stats



def calc_xi(G1, G2, G3, G4):
    # calculate the term xi in the bias correction and asymptotic variance of rho
    partial1 = -(np.linalg.inv(G1).dot(G2).dot(np.linalg.inv(G4)).dot(G3).dot(np.linalg.inv(G1))).T
    partial2 = np.linalg.inv(G1).dot(G2).dot(np.linalg.inv(G4))
    partial3 = np.linalg.inv(G4).dot(G3).dot(np.linalg.inv(G1))
    partial4 = -(np.linalg.inv(G4).dot(G3).dot(np.linalg.inv(G1)).dot(G2).dot(np.linalg.inv(G4))).T
    xi = np.concatenate((partial1.T.reshape((-1,1)), partial2.T.reshape((-1,1)), partial3.T.reshape((-1,1)), partial4.T.reshape((-1,1))), axis=0)
    return xi


def calc_G(Lamlist):
    # calculate G1, G2, G3, G4 that are used to calculate the term xi
    N = Lamlist[0].shape[0]
    G1 = Lamlist[0].T.dot(Lamlist[0])/N
    G2 = Lamlist[0].T.dot(Lamlist[1])/N
    G3 = Lamlist[1].T.dot(Lamlist[0])/N
    G4 = Lamlist[1].T.dot(Lamlist[1])/N
    return G1, G2, G3, G4


########################################################
### helper functions to calculate the asymptotic variance of rho
########################################################

def calc_sigmaBBhat(Fhatlist, Lamhatlist, ehatlist, S, slist, h):
    # calculate the term sigmaBB in the variance of rho
    T, k = Fhatlist[0].shape

    sigmaBBhat = np.zeros((4*k*k, 4*k*k))
    Tslist = [calc_Ts(S, slist[0], h), calc_Ts(S, slist[0], h), calc_Ts(S, slist[1], h), calc_Ts(S, slist[1], h)]
    Flist = [Fhatlist[0], Fhatlist[0], Fhatlist[1], Fhatlist[1]]
    Lamlist = [Lamhatlist[0], Lamhatlist[1], Lamhatlist[0], Lamhatlist[1]]
    elist = [ehatlist[0], ehatlist[0], ehatlist[1], ehatlist[1]]
    for i in range(4):
        for j in range(4):
            F = Flist[i] * Flist[j]/Tslist[i]/Tslist[j]
            Lam = Lamlist[i] * Lamlist[j]
            e = elist[i] * elist[j]
            if k == 1:
                partial_sigmaBB = calc_upsilon(F, Lam, e)
            else:
                partial_sigmaBB = np.diag(np.squeeze(calc_upsilon(F, Lam, e).T.reshape((-1,1))))
            sigmaBBhat[(i*k*k):((i+1)*k*k), (j*k*k):((j+1)*k*k)] = partial_sigmaBB
    sigmaBBhat = sigmaBBhat * T * h / 2
    return sigmaBBhat


def calc_upsilon(F, Lam, e):
    # helper function to calculate sigmaBBhat
    # F is T by r, Lam is N by r, e is T by N
    T, _ = F.shape
    N, _ = Lam.shape
    if e.shape[0] == T:
        return F.T.dot(e).dot(Lam)/N
    else:
        return F.T.dot(e.T).dot(Lam)/N


def calc_Dhat(Vlist, Lamlist):
    # calculate the term D in the variance of rho
    C11 = calc_Chat(1, 1, [Vlist[0], Vlist[0]], [Lamlist[0], Lamlist[0], Lamlist[0]])
    C12 = calc_Chat(1, 2, [Vlist[0], Vlist[1]], [Lamlist[0], Lamlist[1], Lamlist[2]])
    C21 = calc_Chat(2, 1, [Vlist[1], Vlist[0]], [Lamlist[2], Lamlist[1], Lamlist[0]])
    C22 = calc_Chat(2, 2, [Vlist[1], Vlist[1]], [Lamlist[2], Lamlist[2], Lamlist[2]])
    D = np.concatenate((C11, C12, C21, C22), axis=0)
    return D



def calc_Chat(l, lprime, Vlist, Lamlist):
    # helper function to calculate Dhat in the variance of rho
    k = Lamlist[0].shape[0]
    ksq = k**2
    M1 = np.linalg.inv(Vlist[0]).dot(Lamlist[0])
    M2 = np.eye(k)
    M3 = np.linalg.inv(Vlist[0])
    M4 = Lamlist[1]
    M5 = np.eye(k)
    M6 = Lamlist[2].dot(np.linalg.inv(Vlist[1]))
    M7 = Lamlist[1]
    M8 = np.linalg.inv(Vlist[1])
    C1 = np.tensordot(M2.T, M1, axes=0).transpose(0,2,1,3).reshape((ksq,ksq))
    C2 = np.tensordot(M3, M4.T, axes=0).transpose(0,2,1,3).reshape((ksq,ksq))
    C3 = np.tensordot(M5, M6.T, axes=0).transpose(0,2,1,3).reshape((ksq,ksq))
    C4 = np.tensordot(M8.T, M7, axes=0).transpose(0,2,1,3).reshape((ksq,ksq))
    if l == 1 and lprime == 1:
        C = np.concatenate((C1 + C2 + C3 + C4, np.zeros((ksq, ksq)), np.zeros((ksq, ksq)), np.zeros((ksq, ksq))), axis=1)
    elif l == 1 and lprime == 2:
        C = np.concatenate((C2, C1, C3, C4), axis=1)
    elif l == 2 and lprime == 1:
        C = np.concatenate((C4, C3, C1, C2), axis=1)
    else:
        C = np.concatenate((np.zeros((ksq, ksq)), np.zeros((ksq, ksq)), np.zeros((ksq, ksq)), C1 + C2 + C3 + C4), axis=1)
        
    return C


########################################################
### helper functions to calculate the bias correction term of rho
########################################################


def calc_b(Flist, Lamlist, elist, Vlist, S, slist, h):
    # in order to generalized correlation hist use x11, x12, x21, x22, use x1
    # count number that fail the one side test, use x11, x22 (x12 and x21 set as 0), use x1, x2, x3 and divide by N
    k = Flist[0].shape[1]
    x11 = calc_x([Flist[0], Flist[0]], [Lamlist[0], Lamlist[0]], [elist[0], elist[0]], [Vlist[0], Vlist[0]], S, [slist[0], slist[0]], h)
    x12 = calc_x([Flist[0], Flist[1]], [Lamlist[0], Lamlist[1]], [elist[0], elist[1]], [Vlist[0], Vlist[1]], S, [slist[0], slist[1]], h)
    x21 = calc_x([Flist[1], Flist[0]], [Lamlist[1], Lamlist[0]], [elist[1], elist[0]], [Vlist[1], Vlist[0]], S, [slist[1], slist[0]], h)
    x22 = calc_x([Flist[1], Flist[1]], [Lamlist[1], Lamlist[1]], [elist[1], elist[1]], [Vlist[1], Vlist[1]], S, [slist[1], slist[1]], h)
    b = np.concatenate((x11.T.reshape((-1,1)), x12.T.reshape((-1,1)), x21.T.reshape((-1,1)), x22.T.reshape((-1,1))), axis=0)
    return b



def calc_x(Flist, Lamlist, elist, Vlist, S, slist, h):
    # calculate the term x in b (bias correction term)
    T = Flist[0].shape[0]
    N = Lamlist[0].shape[0]
    Ts1 = calc_Ts(S, slist[0], h)
    Ts2 = calc_Ts(S, slist[1], h)

    k = Flist[0].shape[1]
    middle1 = np.zeros((k,k))
    for i in range(N): 
        middle1 += Flist[0].T.dot(np.diag(elist[0][i,:] * elist[1][i,:])).dot(Flist[1])/Ts1/Ts2
    x1 = np.linalg.inv(Vlist[0]).dot(Lamlist[0].T.dot(Lamlist[0])/N).dot(middle1).dot(Lamlist[1].T.dot(Lamlist[1])/N).dot(np.linalg.inv(Vlist[1]))

    middle2 = np.zeros((k,k))
    for i in range(N): 
        middle2 += Flist[0].T.dot(np.diag(elist[0][i,:] * elist[0][i,:])).dot(Flist[0])/Ts1/Ts1
    x2 = np.linalg.inv(Vlist[0]).dot(Lamlist[0].T.dot(Lamlist[0])/N).dot(middle2).dot(Lamlist[0].T.dot(Lamlist[1])/N)

    middle3 = np.zeros((k,k))
    for i in range(N): 
        middle3 += Flist[1].T.dot(np.diag(elist[1][i,:] * elist[1][i,:])).dot(Flist[1])/Ts2/Ts2
    x3 = (Lamlist[0].T.dot(Lamlist[1])/N).dot(middle3).dot(Lamlist[1].T.dot(Lamlist[1])/N).dot(np.linalg.inv(Vlist[1]))
    x = (x1 + x2 + x3)/N
    return x




def calc_y(Flist, Lamlist, elist, Vlist, S, slist, h):
    # calculate the term y in b (bias correction term)
    T = Flist[0].shape[0]
    N = Lamlist[0].shape[0]
    Ts1 = calc_Ts(S, slist[0], h)
    Ts2 = calc_Ts(S, slist[1], h)
    k = Flist[0].shape[1]
    middle1 = np.zeros((k,k))
    for i in range(T): 
        middle1 += Lamlist[0].T.dot(np.diag(elist[0][:,i] * elist[0][:,i])).dot(Lamlist[1])/Ts1/N
    y1 = np.linalg.inv(Vlist[0]).dot(middle1)


    middle2 = np.zeros((k,k))
    for i in range(T): 
        middle2 += Lamlist[0].T.dot(np.diag(elist[1][:,i] * elist[1][:,i])).dot(Lamlist[1])/Ts2/N
    y2 = middle2.dot(np.linalg.inv(Vlist[1]))


    middle3 = np.zeros((k,k))
    for i in range(T): 
        middle3 += Lamlist[0].T.dot(np.diag(elist[0][:,i] * elist[1][:,i])).dot(Lamlist[1])/np.sqrt(Ts1)/np.sqrt(Ts2)/N
    y3 = np.linalg.inv(Vlist[0]).dot(middle3).dot(np.linalg.inv(Vlist[1]))

    y = (y1 + y2 + y3)/N
    return y










    
