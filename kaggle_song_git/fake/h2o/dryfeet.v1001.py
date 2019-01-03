# import h2o
# h2o.init(nthreads=-1)
#
# df = h2o.import_file("../saves/train_set.csv")
# print(df.head(5))
#
# # print(df['target'].summary())
# # print(df['target'].hist())
# #
# # print(df['target'].table())
#
# # print(df['msno'].as)

import h2o
h2o.init()
# h2o.demo("glm")