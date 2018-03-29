#给定负样本为正样本的2倍
spark.sql("CREATE TABLE negative2 AS SELECT * FROM day1617_18_train_data WHERE DAY18_buy=0 ORDER BY RAND() LIMIT 300")
#根据正负例生成新样本
spark.sql("CREATE TABLE train_datatable_02 AS SELECT * FROM positive UNION ALL SELECT * FROM negative2")

assembler=VectorAssembler(inputCols=["t1","t2","t3","t4","t5","t6","t7","t8","t9","t10"],outputCol="features")
#生成数据表保存
spark.sql("CREATE TABLE dataset_train_02 AS SELECT user_id,item_id,2day_view,2day_favor,2day_tocar,2day_buy, \
CASE WHEN day18_buy>0 THEN 1 ELSE 0 END as label,\
2day_view*2 AS t1 , 2day_favor*3 AS t2 , 2day_tocar*4 AS t3 , 2day_buy*1 AS t4 ,\
2day_view*1+2day_favor*2+2day_tocar*3-2day_buy*1 AS t5,\
(2day_view+1)/(2day_view+2day_favor+2day_tocar+2day_buy+1) AS t6,\
(2day_favor+1)/(2day_view+2day_favor+2day_tocar+2day_buy+1) AS t7,\
(2day_tocar+1)/(2day_view+2day_favor+2day_tocar+2day_buy+1) AS t8,\
(2day_buy+1)/(2day_view+2day_favor+2day_tocar+2day_buy+1) AS t9,\
(2day_favor+1)*(2day_tocar+1)-2day_buy*2 AS t10 \
FROM train_datatable_02 WHERE 2day_buy<20")

dataset_train_02=spark.sql("SELECT user_id,item_id,2day_view,2day_favor,2day_tocar,2day_buy, \
CASE WHEN day18_buy>0 THEN 1 ELSE 0 END as label,\
2day_view*2 AS t1 , 2day_favor*3 AS t2 , 2day_tocar*4 AS t3 , 2day_buy*1 AS t4 ,\
2day_view*1+2day_favor*2+2day_tocar*3-2day_buy*1 AS t5,\
(2day_view+1)/(2day_view+2day_favor+2day_tocar+2day_buy+1) AS t6,\
(2day_favor+1)/(2day_view+2day_favor+2day_tocar+2day_buy+1) AS t7,\
(2day_tocar+1)/(2day_view+2day_favor+2day_tocar+2day_buy+1) AS t8,\
(2day_buy+1)/(2day_view+2day_favor+2day_tocar+2day_buy+1) AS t9,\
(2day_favor+1)*(2day_tocar+1)-2day_buy*2 AS t10 \
FROM train_datatable_02 WHERE 2day_buy<20")

output = assembler.transform(dataset_train_02)
train_data=output.select("label","features")
#有了训练数据集，下一步就是要用Spark的MLlib来构建模型，并训练出一个模型来。如何使用spark mllib就不在本文讨论范围内了。
#假设我们已经训练出一个TreeModel。
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(train_data)
featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(train_data)
(trainingData, testData) = train_data.randomSplit([0.7, 0.3])  #这里分一部分作为测试集来测试模型是否可靠
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
TreeModel = pipeline.fit(trainingData)
predictions = model.transform(testData) #根据模型进行测试集预测

#以下代码可以查看模型预测的错误率，实测错误率0.22左右
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

#新添加
#大体概念：DataFrame => Pipeline => A new DataFrame
#Pipeline: 是由若干个Transformers和Estimators连起来的数据处理过程
#Transformer：入：DataFrame => 出： Data Frame
#Estimator：入：DataFrame => 出：Transformer


sIndexer_02 = StringIndexer(inputCol="label", outputCol="indexed02")
si_model_02 = sIndexer_02.fit(train_data)
(trainingData02, testData02) = train_data.randomSplit([0.7, 0.3]) 
td_02 = si_model_02.transform(trainingData02)

#NB不能为负数
from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

#LR
from pyspark.ml.classification import LogisticRegression, GBTClassifier, RandomForestClassifier
model_LR = LogisticRegression(maxIter=5, regParam=0.01)
model_LR = model_LR.fit(train_data)
predict_lr_testData = model_LR.transform(testData)

#计算精度
def computeAcc(data):
	err = data.filter(data['label'] != data['prediction']).count()
	total = data.count()
	acc = float(err)/total
	print err, total, acc
	return acc
#GBT	
gbt = GBTClassifier(maxIter=5, maxDepth=2,labelCol="indexed02")
model_gbt = gbt.fit(td_02)
predict_gbt_testData= model_gbt.transform(testData02)
computeAcc(predict_gbt_testData)

gbt02 = GBTClassifier(maxIter=4, maxDepth=2,labelCol="indexed02")
model_gbt02 = gbt.fit(td_02)
predict_gbt_testData02= model_gbt02.transform(testData02)
computeAcc(predict_gbt_testData02)

#RF
rf = RandomForestClassifier(numTrees=3, maxDepth=2, labelCol="indexedLabel", seed=42)
model_rf= rf.fit(td_02)
predict_rf_testData = model_rf.transform(testData02)
computeAcc(predict_rf_testData)
