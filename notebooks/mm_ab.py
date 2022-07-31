import numpy as np
import pandas as pd
from scipy import stats as sps
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt 
import warnings
from dataclasses import dataclass
import statsmodels.api as sm
tqdm.pandas()
plt.style.use('ggplot')
pd.set_option('use_inf_as_na', True)
plt.rcParams['figure.figsize'] = (15,6)


warnings.simplefilter('ignore')


# simple ab test


def get_ab_size(mde: float = None,
                std: float = None,
                alpha: float = 0.05,
                beta: float = 0.2,
                n_groups: int = 2,
                n_metrics: int = 1,
                population: int = None):
    """
    
    Расчет кол-ва объектов в группе для АБ теста\n
    params:
    -------------------------
    mde - минимальные детектируемые эффект\n
    std - стандартное отклонение на выборке ( сигма)\n
    alpha - False positive rate\n
    beta - False negative rate\n
    n_groups - кол-во групп в тесте\n
    n_metrics - кол-во проверяемых метрик\n
    population : Кол-во данных в выборке\n
        - Если указано, то высчитывает MDE
        - В ином случае None
    
    """
    n_adj = n_groups * (n_groups - 1) / 2 * n_metrics
    z = sps.norm.isf((alpha / 2) / n_adj) + sps.norm.isf(beta)

    if population:
        return n_groups**.5 * (z * std / population**.5)
    
    return n_groups * (z * std / mde)**2


def generate_data(
                success_rate: float = 0.1,
                uplift: float = 0.1,
                N: int = 5000,
                skew: float = 1.0,
                beta: float = 100.,
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
    views_0 = np.exp(sps.norm(1, skew).rvs(N)).astype(int) + 1
    views_1 = np.exp(sps.norm(1, skew).rvs(N)).astype(int) + 1

    # views are always positive, abs is fixing numerical issues with high skewness
    views_0 = np.absolute(views_0)
    views_1 = np.absolute(views_1)

    alpha_0 = success_rate * beta / (1 - success_rate)
    success_rate_0 = sps.beta(alpha_0, beta).rvs(N)

    alpha_1 = success_rate * (1 + uplift) * beta / (1 - success_rate * (1 + uplift))
    success_rate_1 = sps.beta(alpha_1, beta).rvs(N)

    clicks_0 = sps.binom(n=views_0, p=success_rate_0).rvs()
    clicks_1 = sps.binom(n=views_1, p=success_rate_1).rvs()
    
    if is_ctr:
        return pd.Series(clicks_0 / views_0), pd.Series(clicks_1 / views_1)

    return views_0,clicks_0,views_1,clicks_1



@dataclass
class BaseABMethod:
    ''' 
    Базовый класс для всех методов улучшения подсчета АБ тестов\n
    В себя принимает просмотры и клики\n
    Если метрика - вещественное число, заполняются только параметры кликов\n
    Просмотры автозаполняются еденицами, что позволяет импользовать методы и не для ratio метрик\n
    Метод ```calc_metric``` переопределяется наследуемыми методами и содержит все необходимые преобразования
    '''
    control_views:np.ndarray
    control_clicks:np.ndarray
    threatment_views:np.ndarray
    threatment_clicks:np.ndarray
    def __post_init__(self)->None:

        if self.control_views is None:
            self.control_views = np.ones_like(self.control_clicks)
        if self.threatment_views is None:
            self.threatment_views = np.ones_like(self.threatment_clicks)



    def calc_metric(self,logtransform=False)-> None:
        ''' 
        Основной метод формирования метрик.\n
        Сравнивает классические CTR
        
        '''
        self.control_ctr = self.control_clicks / self.control_views
        self.threatment_ctr = self.threatment_clicks / self.threatment_views
        if logtransform:
            self.control_ctr = np.log1p(self.control_ctr)
            self.threatment_ctr = np.log1p(self.threatment_ctr)
            
        self.uplift = self.threatment_ctr.mean() - self.control_ctr.mean()

    
    def ci_check(self,**kwargs)->dict:
        '''
        Расчет основных результатов
        
        '''
        
        self.calc_metric(**kwargs)
        effect = self.threatment_ctr.mean() - self.control_ctr.mean()
        std_diff = (self.control_ctr.var() / self.control_ctr.shape[0]
                    + self.threatment_ctr.var() / self.threatment_ctr.shape[0])**(1/2)

        left_bound, right_bound = sps.norm(scale=std_diff,loc=effect).ppf([0.025, 0.975])
        ci_lenght = right_bound - left_bound
        
        t = effect / std_diff
        pvalue_tt = 2 * (1 - sps.norm.cdf(np.abs(t)))

        pvalue_mw = sps.mannwhitneyu(self.control_ctr,self.threatment_ctr).pvalue
        
        uplift = effect / self.control_ctr.mean()
        if uplift>1:
            uplift=None

        return {'ttest':pvalue_tt,
                'mwtest':pvalue_mw,
                'uplift':effect,
                'relative_uplift':uplift,
                'ci_left':left_bound,
                'ci_right':right_bound,
                'ci_lenght':ci_lenght
                }



class ABReweight(BaseABMethod):
    ''' 
    Учитываем вес каждого пользователя для подсчета CTR\n
    Чем больше просмотров, тем больший вес имеем конкретный пользователь в общем CTR
    '''
    def calc_metric(self,w8_func=np.sqrt) -> None:
        w8_control = w8_func(self.control_views)
        w8_threatment = w8_func(self.threatment_views)

        self.control_ctr = (self.control_clicks / self.control_views) * w8_control / w8_threatment.mean()
        self.threatment_ctr = (self.threatment_clicks / self.threatment_views) * w8_threatment / w8_threatment.mean()


class ABLinearize(BaseABMethod):
    ''' 
    Переводим ratio метрики в вещественное число\n
    Увеличиваем мощность за счет возможности выхода за рамки (0,1)
    '''
    def calc_metric(self) -> None:
        k = self.control_clicks.sum() / self.control_views.sum()

        self.control_ctr = self.control_clicks - self.control_views * k
        self.threatment_ctr = self.threatment_clicks - self.threatment_views * k



class ABBootstrap(BaseABMethod):
    '''
    Пуассоновский бутстрап среднего
    '''
    def ci_check(self,n_bootstrap=1000):
        bts_shape = np.minimum(self.control_clicks.shape[0],self.threatment_clicks.shape[0])

        self.control_clicks = self.control_clicks[:bts_shape]
        self.threatment_clicks = self.threatment_clicks[:bts_shape]
        
        self.control_views = self.control_views[:bts_shape]
        self.threatment_views = self.threatment_views[:bts_shape]
        
        poisson_bootstraps = sps.poisson(1).rvs((n_bootstrap, bts_shape)).astype(np.int64)

        ctr_control = np.matmul(self.control_clicks, poisson_bootstraps.T) / np.matmul(self.control_views, poisson_bootstraps.T)
        ctr_threatment = np.matmul(self.threatment_clicks, poisson_bootstraps.T) / np.matmul(self.threatment_views, poisson_bootstraps.T)

        ctr_control = np.random.choice(ctr_control,ctr_control.shape[0])
        ctr_threatment = np.random.choice(ctr_threatment,ctr_threatment.shape[0])

        left_bound,right_bound = np.quantile(ctr_threatment - ctr_control,[0.025, 0.975])
        ci_lenght = right_bound - left_bound

        effect = ctr_threatment.mean() - ctr_control.mean()
        
        bts_diff = np.sum(ctr_threatment < ctr_control )

        pvalue_tt = 2 * np.minimum (bts_diff,n_bootstrap - bts_diff) / n_bootstrap

        uplift = effect / ctr_control.mean()
       
            
        return {'ttest': pvalue_tt,
                'mwtest': None,
                'uplift': effect,
                'relative_uplift':uplift,
                'ci_left': left_bound,
                'ci_right': right_bound,
                'ci_lenght': ci_lenght,
                }


class ABBucketing(BaseABMethod):
    '''
    Сглажевание CTR через формирование мета-пользователей\n
    Собираем N человек и усредняем их показатели, получившееся считаем мета-пользователем
    '''
    def calc_metric(self,bucket_size=100) -> None:
        min_range = np.minimum(self.control_views.shape[0],self.threatment_views.shape[0])


        bucket_range = min_range // bucket_size
        
        self.control_ctr = np.zeros(bucket_range)
        self.threatment_ctr = np.zeros(bucket_range)

        for step in range(bucket_range):

            idx_start = bucket_size*step
            idx_stop = bucket_size*(step+1)

            self.control_ctr[step] = self.control_clicks[idx_start:idx_stop].sum() / self.control_views[idx_start:idx_stop].sum()
            self.threatment_ctr[step] = self.threatment_clicks[idx_start:idx_stop].sum() / self.threatment_views[idx_start:idx_stop].sum()




class ABBayes(BaseABMethod):
    '''
    Баесовский подход к тестированию через beta моделирование
    '''

    def plot_density(self):
        beta_control = sps.beta(self.control_clicks.sum()+1,self.control_views.sum()+1)
        beta_threatment = sps.beta(self.threatment_clicks.sum()+1,self.threatment_views.sum()+1)

        beta_df = pd.DataFrame(np.vstack([beta_control.rvs(10000),beta_threatment.rvs(10000)]).T,columns=['control','threatment'])
        g=sns.jointplot(x=beta_df.control, 
                        y=beta_df.threatment,
                        kind='kde',
                        n_levels=15)
        g.ax_joint.plot([beta_df.min().min(), beta_df.max().max()], [beta_df.min().min(), beta_df.max().max()])
        return g

    def ci_check(self) -> dict:
        
        beta_control = sps.beta(self.control_clicks.sum()+1,self.control_views.sum()+1)
        beta_threatment = sps.beta(self.threatment_clicks.sum()+1,self.threatment_views.sum()+1)
        
        effect = beta_threatment.mean() - beta_control.mean()
        
        diff =  beta_threatment.rvs(10000) - beta_control.rvs(10000)

        uplift = effect / beta_control.mean()
        
        diff_theoretical = sps.norm(scale=diff.std(),loc=diff.mean())

        left_bound, right_bound = diff_theoretical.ppf([0.025, 0.975])

        ci_lenght = right_bound - left_bound

        p = diff_theoretical.cdf(x=0)

        pval  = min(p,1-p) * 2

        return {
            'ttest': pval,
            'mwtest': None,
            'uplift': effect,
            'relative_uplift':uplift,
            'ci_left': left_bound,
            'ci_right': right_bound,
            'ci_lenght': ci_lenght,
                }

class ABML(BaseABMethod):

    def prepare_data(self,cov_dict=None):
        self.calc_metric()
        if cov_dict is None:
            cov_dict = {'control':np.array([]),
                        'threatment':np.array([])}
                        
        control_data = np.vstack([  
                                    np.zeros(self.control_views.shape[0]),
                                    *cov_dict['control'].T,
                                    np.ones(self.control_views.shape[0]),
                                    self.control_ctr])
        
        threatment_data = np.vstack([
                                    np.ones(self.threatment_views.shape[0]),
                                    *cov_dict['threatment'].T,
                                    np.ones(self.threatment_views.shape[0]),
                                    self.threatment_ctr])

        data = np.hstack([control_data,threatment_data]).T
        
        target = data[:,-1]
        data = data[:,:-1]

        return data,target


    def ci_check(self,raw=False,**kwargs):
        data,target = self.prepare_data(**kwargs)
        model = sm.OLS(target,data).fit()

        if raw:
            return model.summary()

        return {'ttest': model.pvalues[0],
                'mwtest': None,
                'uplift': model.params[0],
                'relative_uplift':model.params[0] / self.control_ctr.mean(),
                'ci_left': model.conf_int()[0][0],
                'ci_right': model.conf_int()[0][1],
                'ci_lenght': model.conf_int()[0][1] - model.conf_int()[0][0],
                }
                    

class ABCombo(BaseABMethod):

    '''
    Комба подход, когда из всех методов мы выбираем наилучший для конкретной ситуации
    '''

    def ci_check(self, method:str= 'ci_left_lenght') -> dict:
        self.calc_metric()
        result = {}
        result.update({'reweight_sqrt':ABReweight(self.control_views,self.control_clicks,self.threatment_views,self.threatment_clicks).ci_check(w8_func=np.sqrt)})
        result.update({'reweight_log':ABReweight(self.control_views,self.control_clicks,self.threatment_views,self.threatment_clicks).ci_check(w8_func=np.log)})
        result.update({'linearization':ABLinearize (self.control_views,self.control_clicks,self.threatment_views,self.threatment_clicks).ci_check()})
        result.update({'bucketing':ABBucketing(self.control_views,self.control_clicks,self.threatment_views,self.threatment_clicks).ci_check()})
        result.update({'bootstrap':ABBootstrap(self.control_views,self.control_clicks,self.threatment_views,self.threatment_clicks).ci_check()})
        result.update({'bayesian':ABBayes(self.control_views,self.control_clicks,self.threatment_views,self.threatment_clicks).ci_check()})
        
        result = pd.DataFrame(result)
        if method == 'ci_lenght':
            return result.T.sort_values(by='ci_lenght',ascending=True).iloc[0].to_dict()
        if method == 'ci_left':
            return result.T.sort_values(by='ci_left',ascending=False).iloc[0].to_dict()
        if method == 'ci_left_lenght':
            return result.T.assign(ci_left_lenght = abs(self.uplift - result.T['ci_left']) ).sort_values(by='ci_left_lenght',ascending=True).iloc[0].to_dict()
    
                    
    
def generate_tests(control_views:np.ndarray,
                    control_clicks:np.ndarray,
                    threatment_views:np.ndarray,
                    threatment_clicks:np.ndarray):
    result = {}
    result.update({'base':BaseABMethod(control_views,control_clicks,threatment_views,threatment_clicks).ci_check()})
    result.update({'reweight':ABReweight(control_views,control_clicks,threatment_views,threatment_clicks).ci_check()})
    result.update({'linearization':ABLinearize (control_views,control_clicks,threatment_views,threatment_clicks).ci_check()})
    result.update({'bucketing':ABBucketing(control_views,control_clicks,threatment_views,threatment_clicks).ci_check()})
    result.update({'bootstrap':ABBootstrap(control_views,control_clicks,threatment_views,threatment_clicks).ci_check()})
    result.update({'bayesian':ABBayes(control_views,control_clicks,threatment_views,threatment_clicks).ci_check()})
    result.update({'ML':ABML(control_views,control_clicks,threatment_views,threatment_clicks).ci_check()})
    result.update({'Combo':ABCombo(control_views,control_clicks,threatment_views,threatment_clicks).ci_check()})
    
    return pd.DataFrame(result)

