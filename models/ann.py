# Import PySpark and the necessary libraries for linear regression
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("PySpark Regression ANN Example").getOrCreate()

schema = StructType([
    StructField("day_of_week", IntegerType(), True),
    StructField("day_of_month", IntegerType(), True),
    StructField("month", IntegerType(), True),
    StructField("exit_stop", IntegerType(), True),
    StructField("target_stop", IntegerType(), True),
    StructField("distance", DoubleType(), True),
    StructField("exit_time", DoubleType(), True),
    StructField("arrive_time", DoubleType(), True)
])

# Load data into a PySpark DataFrame
data = spark.read.format("csv").option("header", "true").schema(schema).load("../processed_datasets/2014-08-01.csv")

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
