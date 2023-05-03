# Import PySpark and the necessary libraries for linear regression
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType


spark = SparkSession.builder.appName("PySpark Regression ANN Example").getOrCreate()

# Load data into a PySpark DataFrame using the specified schema

data = spark.read.format("parquet").load("../processed_datasets/*.parquet")
data = data.drop('__null_dask_index__')
data = data.dropDuplicates()

# Rename a column
data = data.withColumnRenamed("arrive_time", "label")

# Prepare data for linear regression training
assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
data = assembler.transform(data).select("features", "label")

# Split data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=12345)

# Set up the linear regression model
regressor = LinearRegression(featuresCol="features", labelCol="label")

# Train the linear regression model
model = regressor.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using mean squared error
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mse")
mse = evaluator.evaluate(predictions)

print("Mean Squared Error = %g" % mse)
