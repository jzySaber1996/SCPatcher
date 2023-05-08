import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载示例数据集
# df = sns.load_dataset('iris')
# df.head()
df = pd.DataFrame(pd.read_csv("data.csv", encoding='utf8'))
df.head()

# 绘制多个变量的小提琴图
# plot
sns.violinplot(df)
plt.yticks([1,2,3,4,5])
# plt.ylim(0.5,5.5)
plt.ylabel('score')
plt.show()