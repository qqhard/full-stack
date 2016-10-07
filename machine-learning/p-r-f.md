
# 准确率，召回率，F值

正负样本数相当时，可取阈值0.5，计算准确率，召回率和F值的评估，来评估模型的优劣。

对于一个组用于测试的样本，其label已知，分类器会将样本分为正负两部分，样本被分割成四个部分。

![fpr-and-tpr](fpr-and-tpr.png)

准确率 $$precision = \frac{TP}{TP+FP}$$

召回率 $$recall = \frac{TP}{TP+FN}$$

F值 $$F-measure = \frac{precision*recall*2}{precision+recall}$$
