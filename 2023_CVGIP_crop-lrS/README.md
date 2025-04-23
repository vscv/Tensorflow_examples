2022_Leaf_rewrite-for-model-benchmark-forConvert2Py-20230626-CVGIP-lrS.ipynb
2023-06-26

主題：在單一模型下提升辨識效能的嘗試

探討不同LR排程對於提升卷積神經網路在crop疾病辨識的有效性

'投稿CVGIP版本 僅討論不同lr，限定在低功耗裝置，因此僅能用小模型mbnet, efficentNet等，aug僅使用AutoAug一種以示公平。

2023-06-28

原本規劃題目是 單一模型下討論不同的Aug方法對效能的幫助 但要做K-fold 5 x aug 2~4種 訓練時間太久，原本的k-fold沒找到變成要全部重寫。所以改題目，使用先前的數據來寫，遍歷42種模型效能的測試實驗。 Comprehensive Evaluation of 42 State-of-the-Art Deep Neural Networks for Crop Disease Classification Comprehensive Evaluation of State-of-the-Art Deep Neural Networks for Crop Disease Classification
