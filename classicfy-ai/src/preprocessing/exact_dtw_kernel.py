"""Compile the published DTWSL 0/1 membership recurrence without changing it.

Parangonar 3.3.3's cdist_dtw_single_loop is a Python quadratic loop. The full
float64 cumulative matrix and its ORIGINAL traceback are retained. No band,
approximation, pruning, new distance, or altered tie-breaking is introduced.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def membership_cumulative(pitches, pitch_sets):
    n,m=len(pitches),len(pitch_sets)
    costs=np.full((n+1,m+1),np.inf,np.float64)
    costs[0,0]=0.
    for i in range(1,n+1):
        for j in range(1,m+1):
            distance=0. if pitch_sets[j-1,pitches[i-1]] else 1.
            costs[i,j]=distance+min(costs[i-1,j],costs[i,j-1],costs[i-1,j-1])
    return costs[1:,1:]


@njit(cache=True)
def pairwise_cumulative(distances):
    n,m=distances.shape
    costs=np.full((n+1,m+1),np.inf,np.float64)
    costs[0,0]=0.
    for i in range(1,n+1):
        for j in range(1,m+1):
            costs[i,j]=distances[i-1,j-1]+min(costs[i-1,j],costs[i-1,j-1],costs[i,j-1])
    return costs[1:,1:]


def install_exact_kernel():
    import parangonar.dp.dtw as module
    from parangonar.dp.metrics import element_of_set_metric
    from scipy.spatial.distance import cdist, euclidean
    if getattr(module.cdist_dtw_single_loop,'_classicfy_exact',False): return
    original=module.cdist_dtw_single_loop
    def compiled(arr1,arr2,metric):
        if metric is not element_of_set_metric:
            return original(arr1,arr2,metric)
        masks=np.zeros((len(arr2),128),np.bool_)
        for j,pitches in enumerate(arr2):
            for pitch in pitches: masks[j,int(pitch)]=True
        return membership_cumulative(np.asarray(arr1,np.int64),masks)
    compiled._classicfy_exact=True
    compiled._classicfy_original=original
    module.cdist_dtw_single_loop=compiled
    original_pairwise=module.dtw_dmatrix_from_pairwise_dmatrix
    pairwise_cumulative._classicfy_original=original_pairwise
    module.dtw_dmatrix_from_pairwise_dmatrix=pairwise_cumulative
    original_init=module.DynamicTimeWarping.__init__
    def fast_cdist(x,y,metric):
        if metric is euclidean and x.shape[1]==1 and y.shape[1]==1:
            # In one dimension SciPy's Euclidean norm is exactly abs(x-y).
            return np.abs(x[:,0,None]-y[None,:,0])
        return cdist(x,y,metric)
    def init(self,metric=euclidean,cdist_fun=cdist):
        original_init(self,metric=metric,cdist_fun=fast_cdist if cdist_fun is cdist else cdist_fun)
    module.DynamicTimeWarping.__init__=init
