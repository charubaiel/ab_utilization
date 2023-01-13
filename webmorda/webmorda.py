

from ab_funcs import *
import streamlit as st
import plotly.express as px


st.set_page_config(layout="wide",page_title="Моделирование АБшек")


left_bar, center,right_bar = st.columns((3,8,3))







with left_bar:

    base_ctr = st.slider('CTR',0.0,.99,step=0.001,value=.1)
    uplift = st.slider('Relative Uplift in threatment',0.0,.10,step=0.001,value=.05)
    n_experiments = st.number_input('Кол-во повторений экспериментов',min_value=0,max_value = 10000,value=10)
    days = st.slider('Кол-во дней',1,30,step=1,value=11)
    users = st.slider('Кол-во пользователей в день',100,1000,step=100,value=500)
    methods_names = st.multiselect('Методы',methods_dict.values(),default=methods_dict.values())
    metrics_names = st.multiselect('Метрики',metrics_dict.values(),default='Т-тест')
    st.button('Refresh')





params = {'N': int(days*users ),
        'ctr':base_ctr,
        'uplift':uplift,
        'N_sets':n_experiments,
        'is_ctr':False}






result,list_of_data = generate_N_experiments(**params)
control_ctr = list_of_data[1] / list_of_data[0]
threatment_ctr = list_of_data[3] / list_of_data[2]
ctr_df = pd.DataFrame({'control':control_ctr,
                        'threatment':threatment_ctr})


# bootstrap_sample = pd.concat([ctr_df.sample(frac=.5,replace=True).mean() for _ in range(1000)],axis=1).T
ttl_df = pd.concat(result).rename(index=metrics_dict,columns = methods_dict)
show_df = ttl_df.loc[metrics_names,methods_names]




data_hist = px.histogram(ctr_df.where(lambda x: (x>0) & (x <1)).melt(),
                        opacity=0.6,x='value',
                        color='variable',
                        nbins=500,
                        barmode='overlay',
                        title= 'Распределение наших CTR')

score_boxplot = px.box(show_df.melt(ignore_index=False).reset_index(),
                    animation_frame='index',x='variable',y='value',color='variable',
                    orientation = 'v',title='Агрегированное распределение оценок АБ')

ctr_df.index = ctr_df.index // users
agg_df = ctr_df.groupby(level=0).mean()

lineplot = px.line(agg_df,title='Динамика оценок АБ')

bts = pd.concat([ctr_df.sample(users*days).mean() for _ in range(1000)],axis=1)

data_bts = px.histogram(bts.T.melt(),
                        opacity=0.6,x='value',
                        color='variable',
                        nbins=500,
                        barmode='overlay',
                        title= 'Бутстрапированное распределение средних наших CTR')

mde = get_ab_size(population=params["N"]*2,std = ctr_df.std().mean())


with right_bar:
    
    st.header(f'Кол-во пользоватлей :')
    st.header(f'{params["N"]} vs {params["N"]}')
    st.header(f'MDE :')
    st.header(f'{mde:.2%}')
    st.header(f'Реальный эффект :')
    st.header(f"{ctr_df.mean()['threatment'] - ctr_df.mean()['control']:.2%}")

#     views_control = st.text_area('views_control')
#     clicks_control = st.text_area('clicks_control')
#     clicks_threatment = st.text_area('views_threatment')
#     clicks_threatment = st.text_area('clicks_threatment')


with center:
    
    st.write('Средняя величина метрики')
    st.table(show_df.groupby(level=0).mean())
    st.plotly_chart(score_boxplot,use_container_width=True)
    st.plotly_chart(data_hist,use_container_width=True)
    st.plotly_chart(lineplot,use_container_width=True)
    st.plotly_chart(data_bts,use_container_width=True)

    pass



# mde = get_ab_size(population=days * users, std = ab_ctr_control.std())

# st.write(f'То что мы можем найти : {mde:.3%}')
# st.write(f'То что мы имеем : {ab_ctr_threatment.mean() - ab_ctr_control.mean():.3%} ')


