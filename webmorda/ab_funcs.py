import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm
import matplotlib.pyplot as plt 
import streamlit as st
import warnings
tqdm.pandas()
plt.style.use('ggplot')
pd.set_option('use_inf_as_na', True)
plt.rcParams['figure.figsize'] = (15,6)





def get_ab_size(mde:float=None,
                    std:float=None,
                    alpha:float=0.05,
                    beta:float=0.1,
                    population:int=None):
    """Расчет кол-ва объектов в группе для АБ теста

    mde - минимальные детектируемые эффект\n
    std - стандартное отклонение на выборке ( сигма)\n
    alpha - False positive rate\n
    beta - False negative rate\n
    population : Кол-во данных в выборке\n
        - Если указано, то высчитывает MDE
        - В ином случае None
    
    """
    z = stats.norm.isf(alpha/2)+stats.norm.isf(beta)

    if population:
        return (z * std**.5) / np.sqrt(population)
    else:
        return (z**2 * std) / mde**2




def generate_data(skew: float = 100,
                  N: int = 5000,
                  success_rate: float = 0.02,
                  uplift: float = 0.1,
                  beta: float = 200.,
                  is_ctr = False):
    """
    Generates experimental data for N users in NN experiments
    skew: float, skewness of views distribution
    N: int, number of users in each experimental group (in control and in treatment)
    success_rate: float, mean success rate in control group
    uplift: float, relative uplift of mean success rate in treatment group
    beta: float, parameter of success rate distribution
    return: pd.DataFrame([Clicks,Views]),pd.DataFrame([Clicks,Views])
    """
    views_0 = stats.expon(1, skew).rvs(N).astype(int) + 1
    views_1 = stats.expon(1, skew).rvs(N).astype(int) + 1

    # # views are always positive, abs is fixing numerical issues with high skewness
    views_0 = np.absolute(views_0)
    views_1 = np.absolute(views_1)

    alpha_0 = success_rate * beta / (1 - success_rate)
    alpha_1 = success_rate * (1 + uplift) * beta / (1 - success_rate * (1 + uplift))


    success_rate_0 = stats.beta(alpha_0, beta).rvs(N)
    success_rate_1 = stats.beta(alpha_1, beta).rvs(N)

    clicks_0 = stats.binom(n=views_0.astype(int), p=success_rate_0).rvs()
    clicks_1 = stats.binom(n=views_1.astype(int), p=success_rate_1).rvs()
    
    if is_ctr:
        return pd.Series(clicks_0 / views_0), pd.Series(clicks_1 / views_1)

    return pd.DataFrame({'views':views_0, 'clicks':clicks_0}),\
        pd.DataFrame({'views':views_1, 'clicks':clicks_1}),



def ci_check (control:np.ndarray,threatment:np.ndarray):

    effect = threatment.mean() - control.mean()
    std_diff = (control.var() / control.shape[0] + threatment.var() / threatment.shape[0])**(1/2)

    left_bound, right_bound = stats.norm(scale=std_diff,loc=effect).ppf([0.025, 0.975])
    ci_lenght = right_bound - left_bound
    
    t = effect / std_diff
    pvalue_tt = 2 * (1 - stats.norm.cdf(np.abs(t)))

    pvalue_mw = stats.mannwhitneyu(control,threatment).pvalue
    
    uplift = effect / control.mean()
    return {'ttest':pvalue_tt,
            'mwtest':pvalue_mw,
            'effect':effect,
            'left_ci':left_bound,
            'right_ci':right_bound,
            'ci_lenght':ci_lenght,
            'relative_uplift':uplift
            }

def raw_ctr_test(control_views:np.ndarray = [],
                    control_clicks:np.ndarray = [],
                    threatment_views:np.ndarray = [],
                    threatment_clicks:np.ndarray = []):
    control_ctr = (control_clicks / control_views)
    threatment_ctr = (threatment_clicks / threatment_views)
    return {'Raw ctr ttest' : ci_check(np.asarray(control_ctr),
                                        np.asarray(threatment_ctr))
                }
    
def rctr_test(control_views:np.ndarray = [],
                    control_clicks:np.ndarray = [],
                    threatment_views:np.ndarray = [],
                    threatment_clicks:np.ndarray = []):
    
    control_ctr = (control_clicks / control_views) * np.sqrt(control_views) / np.sqrt(control_views).sum()
    threatment_ctr = (threatment_clicks / threatment_views) * np.sqrt(threatment_views) / np.sqrt(threatment_views).sum()

    return {'weighted_ctr' : ci_check(np.asarray(control_ctr),
                                    np.asarray(threatment_ctr))
                }
    
    
def rlctr_test(control_views:np.ndarray = [],
                control_clicks:np.ndarray = [],
                threatment_views:np.ndarray = [],
                threatment_clicks:np.ndarray = []):
    

    # k = np.mean((control_clicks / control_views) * np.sqrt(control_views) / np.sqrt(control_views).sum())
    k = control_clicks.sum() / control_views.sum()

    control_ctr = (control_clicks - control_views * k) * np.sqrt(control_views) / np.sqrt(control_views).sum()
    threatment_ctr = (threatment_clicks - threatment_views * k) * np.sqrt(threatment_views) / np.sqrt(threatment_views).sum()


    return {'reweighted_linear_ctr' : ci_check(np.asarray(control_ctr),
                                            np.asarray(threatment_ctr))
                }



def bootstrap_poisson(control_views:np.ndarray = [],
                    control_clicks:np.ndarray = [],
                    threatment_views:np.ndarray = [],
                    threatment_clicks:np.ndarray = [], n_bootstrap=1000, raw = False):


    bts_shape = np.minimum(control_clicks.shape[0],threatment_clicks.shape[0])
    

    control_clicks = control_clicks[:bts_shape]
    threatment_clicks = threatment_clicks[:bts_shape]

    if len(control_views) == 0:
        control_views = np.ones_like(control_clicks)
        threatment_views = np.ones_like(control_clicks)
    
    control_views = control_views[:bts_shape]
    threatment_views = threatment_views[:bts_shape]
    
    poisson_bootstraps = stats.poisson(1).rvs((n_bootstrap, bts_shape)).astype(np.int64)

    ctr_control = np.matmul(control_clicks, poisson_bootstraps.T) / np.matmul(control_views, poisson_bootstraps.T)
    ctr_threatment = np.matmul(threatment_clicks, poisson_bootstraps.T) / np.matmul(threatment_views, poisson_bootstraps.T)

    left_bound,right_bound = np.quantile(ctr_threatment - ctr_control,[0.025, 0.975])
    ci_lenght = right_bound - left_bound

    effect = ctr_threatment.mean() - ctr_control.mean()
    
    bts_diff = np.sum(ctr_threatment < ctr_control )
    pvalue_tt = 2 * np.minimum (bts_diff,n_bootstrap - bts_diff) / n_bootstrap


    uplift = effect / ctr_control.mean()
    if raw:
        return ctr_control,ctr_threatment
        
    return {'bootstrap':{'ttest': pvalue_tt,
            # 'mwtest': None,
            'effect': effect,
            'left_ci': left_bound,
            'right_ci': right_bound,
            'ci_lenght': ci_lenght,
            'relative_uplift':uplift}}


            
def bucketing(control_views:np.ndarray = [],
                    control_clicks:np.ndarray = [],
                    threatment_views:np.ndarray = [],
                    threatment_clicks:np.ndarray = [],bucket_size=100,raw=False):


    if len(control_views) == 0:
        control_views = np.ones_like(control_clicks)
        threatment_views = np.ones_like(control_clicks)


    ctrl_bucket = []
    thr_bucket = []

    min_range = np.minimum(control_views.shape[0],threatment_views.shape[0])
    bucket_range = min_range // bucket_size
    

    for i in range(bucket_range):
        bucket_ctrl = control_clicks[bucket_size*i:bucket_size*(i+1)].sum() / control_views[bucket_size*i:bucket_size*(i+1)].sum()
        bucket_thr = threatment_clicks[bucket_size*i:bucket_size*(i+1)].sum() / threatment_views[bucket_size*i:bucket_size*(i+1)].sum()

        ctrl_bucket.append(bucket_ctrl)
        thr_bucket.append(bucket_thr)
    if raw:
        return ctrl_bucket,thr_bucket  

    return {'bucketing':ci_check(np.asarray(ctrl_bucket),
                                np.asarray(thr_bucket))
            }







def linearization(control_views:np.ndarray = [],
                    control_clicks:np.ndarray = [],
                    threatment_views:np.ndarray = [],
                    threatment_clicks:np.ndarray = [], raw=False):

    ctr_c = control_clicks.sum() / control_views.sum()

    linear_control = control_clicks - ctr_c*control_views
    linear_threatment = threatment_clicks - ctr_c*threatment_views
    if raw:
        return linear_control,linear_threatment
    result = {'linearization':ci_check(linear_control,linear_threatment)
            }
    del result['linearization']['relative_uplift']
    return result


def bayes(control_views:np.ndarray = [],
                    control_clicks:np.ndarray = [],
                    threatment_views:np.ndarray = [],
                    threatment_clicks:np.ndarray = []):


    beta_control = stats.beta(control_clicks.sum(),control_views.sum())
    beta_threatment = stats.beta(threatment_clicks.sum(),threatment_views.sum())
    
    effect = beta_threatment.mean() - beta_control.mean()
    
    diff =  beta_threatment.rvs(10000) - beta_control.rvs(10000)

    uplift = effect / beta_control.mean()
    
    left_bound, right_bound = stats.norm(scale=np.std(diff),loc=np.mean(diff)).ppf([0.025, 0.975])
    ci_lenght = right_bound - left_bound

    p = stats.norm.cdf(x=0,loc= diff.mean(),scale = diff.std())
    pval  = min(p,1-p) * 2

    return {'bayes': {'ttest': pval,
                        'mwtest': None,
                        'effect': effect,
                        'left_ci': left_bound,
                        'right_ci': right_bound,
                        'ci_lenght': ci_lenght,
                        'relative_uplift':uplift}}
            

def generate_tests(control_views:np.ndarray = [],
                    control_clicks:np.ndarray = [],
                    threatment_views:np.ndarray = [],
                    threatment_clicks:np.ndarray = []):
    result = {}
    result.update(raw_ctr_test(control_views,control_clicks,threatment_views,threatment_clicks))
    result.update(rctr_test(control_views,control_clicks,threatment_views,threatment_clicks))
    result.update(rlctr_test (control_views,control_clicks,threatment_views,threatment_clicks))
    result.update(bucketing(control_views,control_clicks,threatment_views,threatment_clicks))
    result.update(bootstrap_poisson(control_views,control_clicks,threatment_views,threatment_clicks))
    result.update(bayes(control_views,control_clicks,threatment_views,threatment_clicks))
    result.update(linearization(control_views,control_clicks,threatment_views,threatment_clicks))
    
    return pd.DataFrame(result)


def experiment(success_rate=.1,uplift=.1,N=1000,skew=1,n_exps = 100,beta=100):
    result = pd.DataFrame()
    for _ in tqdm(range(n_exps)):
        AB_c,AB_t = generate_data(success_rate=success_rate,uplift=uplift,N=N,skew=skew,beta=beta)
        AB_result = generate_tests(*AB_c.values.T,*AB_t.values.T).loc['ttest']
        result = result.append(AB_result.rename(N))
    return result
    
def experiment_cum(success_rate=.1,uplift=.1,N=100,skew=1,n_exps = 100,beta=100):
    result = pd.DataFrame()
    ttl_control = pd.DataFrame()
    ttl_threatment =  pd.DataFrame()
    for _ in tqdm(range(n_exps)):
        control_sample,threatment_sample = generate_data(success_rate=success_rate,uplift=uplift,N=N,skew=skew,beta=beta)
        ttl_control = ttl_control.append(control_sample)
        ttl_threatment = ttl_threatment.append(threatment_sample)
        AB_result = generate_tests(*ttl_control.values.T,*ttl_threatment.values.T).loc['ttest']
        result = result.append(AB_result.rename(N*_))
    return result



metrics_dict = {'ttest':"Т-тест",
                'mwtest':'Ранг-Тест',
                'effect':'Прирост метрики',  
                'left_ci':'Левая граница CI',
                'rigth_ci':'Правая граница CI',
                'ci_lenght':'Длинна доверительного интервала',
                'relative_uplift':'Прирост метрики %'}

methods_dict = {'Raw ctr ttest': 'Классика',
                'weighted_ctr': 'Взвешенный',
                'bucketing': 'Бакетинг',
                'bootstrap': 'Бутстрап',
                'bayes': 'Баес',
                'linearization': 'Линеаризация'}






def generate_single_set(N:int=1000,
                        ctr:float=.1,
                        uplift:float=.1,
                        is_ctr=True):

    control,threatment = generate_data(N=N,
                            success_rate=ctr,
                            uplift = uplift,
                            is_ctr=is_ctr)
    
    if ~is_ctr:
        return control.values[:,0],control.values[:,1],threatment.values[:,0],threatment.values[:,1]

    return control, threatment

    

@st.cache
def generate_N_experiments(N:int=1000,
                            ctr:float=.1,
                            uplift:float=.1,
                            N_sets:int=100,
                            is_ctr = False):



    list_of_data = [generate_single_set(N=N,
                                        ctr=ctr,
                                        uplift = uplift,
                                        is_ctr = is_ctr) for _ in range(N_sets)]

    list_of_tests = [generate_tests(*dataset) for dataset in list_of_data]

    return list_of_tests,np.hstack(list_of_data)
