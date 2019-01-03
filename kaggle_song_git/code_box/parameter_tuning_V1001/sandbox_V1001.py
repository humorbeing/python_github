import numpy as np

# while True:
#     print(np.random.uniform(0, 0.1))
model_time = str(55555555555555)
boosting = 'ggbb'
estimate = 0.6677
model_name = '_L_'+boosting
model_name = '['+str(estimate)+']'+model_name
model_name = model_name + '_' + model_time

print(model_name)