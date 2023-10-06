import ets.models as models

import pandas as pd



data = [14,10,6,2,18,8,4,1,16,9,5,3,18,11,4,2,17,9,5,1]
df = models.triple_exponential_smoothing(data, period_length=12, horizon=4, alpha=0.3, beta=0.2, phi=0.9, gamma=0.2)
print(df)
models.kpi(df)