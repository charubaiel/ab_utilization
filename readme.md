# general

Солянка из всевозможных найденных способах улучшения проведения АБ эксперимента

# description

notebooks - файлы с реализациями и проверками всех методов
* `mm_ab.py` - файл с реализацией всех базовых методов готовых к использованию
* `ab_method_tests.ipynb` - файл с синтетической валидацией методов в виде 1000 АА\АБ 
* `bandits.ipynb` - файл с проверкой бандитов thompson sampling (aka Баесовские бандиты)
* `real_case_example.ipynb` - файл с итеративным применением всех методов в рамках одного реального эксперимента в приложении

data - файлы данных для проверок

pyproject.toml - файл с необходимыми зависимостями  
poetry.lock - файл поетри для восстановления окружения

# reproducibility

0) `pip install poetry` # если нет поетри
1) `poetry install` # установка всех необходимых пакетов
2) `poetry shell` # активация виртуального окружения  
2.1 Выбирайте любой любимый редактор для просмотра  
2.2 В редакторе в виде kernel выбираем окружение  
3) `exit` # для выключения окружения


# references 

Общее  
[Авито общая инфа о инфраструктуре](https://habr.com/ru/company/avito/blog/454164/)
[Большой гайд по оптимизации от VK](https://vkteam.medium.com/practitioners-guide-to-statistical-tests-ed2d580ef04f)  
[AB лайфхаки авито  Часть 1 ( тестирование критериев )](https://habr.com/ru/company/avito/blog/571094/)  
[Антипаттерны АБ](https://www.evanmiller.org/how-not-to-run-an-ab-test.html)  

CUPED   
[ AB лайфхаки авито  Часть 2 ( сниженеие дисперсии и джедайские техники )](https://habr.com/ru/company/avito/blog/571096/)  
[ AB лайфхаки авито  Часть 3 ( ML техники и апдейт CUPED )](https://habr.com/ru/company/avito/blog/590105/)  
[CUPED + Стратификация от yandex](https://habr.com/ru/company/yandex/blog/497804/)  

Bayes  

[Пример баесовского АБ (простой)](https://academy.yandex.ru/posts/prostoy-gid-po-bayesovskomu-a-b-testirovaniyu-na-python)  
[Bayies чуть сложнее на примерах](https://medium.com/convoy-tech/the-power-of-bayesian-a-b-testing-f859d2219d5)  
[Более сложная инфа о баесе](https://www.evanmiller.org/bayesian-ab-testing.html)  
[Бандиты и баес на русском (ч1)](https://craftappmobile.com/bayesian-ab-testing-part-1/)  
[Бандиты и баес на русском (ч2)](https://craftappmobile.com/bayesian-ab-testing-part-2/)  
[Бандиты и их реализации на русском (ods)](https://habr.com/ru/company/ods/blog/325416/)
[Бандиты и их реализации на русском (domclick)](https://habr.com/ru/company/domclick/blog/547258/)
[Бандиты и их реализации на русском (domclick) git](https://github.com/WhatIThinkAbout/BabyRobot/tree/master/Multi_Armed_Bandits)

Matching & Causal & ML

[Крутая гит книжка про каузальность](https://matheusfacure.github.io/python-causality-handbook/10-Matching.html)  
[Double Robust & inverse scaling](https://drive.google.com/file/d/1vgwNdBbrSwCaHF7EnJh06Ukpu1mSzIxO/view)
[Матчинг для АБ](https://humboldt-wi.github.io/blog/research/applied_predictive_modeling_19/matching_methods/)  
[Книга по каузальности от биопрофессора (осторожно, это жестко)](https://cdn1.sph.harvard.edu/wp-content/uploads/sites/1268/2021/03/ciwhatif_hernanrobins_30mar21.pdf)  
[Каузальные метода на русском доступно без регистрации и смс](https://koch-kir.medium.com/causal-inference-from-observational-data-%D0%B8%D0%BB%D0%B8-%D0%BA%D0%B0%D0%BA-%D0%BF%D1%80%D0%BE%D0%B2%D0%B5%D1%81%D1%82%D0%B8-%D0%B0-%D0%B2-%D1%82%D0%B5%D1%81%D1%82-%D0%B1%D0%B5%D0%B7-%D0%B0-%D0%B2-%D1%82%D0%B5%D1%81%D1%82%D0%B0-afb84f2579f2)  
[AB как модели](https://lindeloev.github.io/tests-as-linear/)

