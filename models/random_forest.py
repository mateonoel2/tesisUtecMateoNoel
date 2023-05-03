# Import PySpark and the necessary libraries for linear regression
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("PySpark Regression ANN Example").getOrCreate()

# Load data into a PySpark DataFrame using the specified schema

data = spark.read.format("parquet").load("../processed_datasets/*.parquet")
data = data.drop('__null_dask_index__')
data = data.dropDuplicates()

print("Number of rows:", data.count())

# Rename a column
data = data.withColumnRenamed("arrive_time", "label")

# Prepare data for linear regression training
assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
data = assembler.transform(data).select("features", "label")

# Split data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=21)

# Set up the Random Forest regression model
regressor = RandomForestRegressor(featuresCol="features", labelCol="label")

# Train the linear regression model
model = regressor.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using mean squared error and r2
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) = {:.4f}".format(rmse))

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(predictions)
print("R-squared (R2) = {:.4f}".format(r2))

