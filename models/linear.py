from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import os
import sys

os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.driver.extraJavaOptions="-XX:-UseGCOverheadLimit" pyspark-shell'

# Define a function to handle errors during model training
def handle_error(e):
    print("Error occurred during model training: ", e)
    sys.exit(1)

# Create a SparkSession
spark = SparkSession.builder \
    .appName("PySpark Linear Regression") \
    .config("spark.network.timeout", "2h") \
    .config("spark.core.connection.ack.wait.timeout", "2h") \
    .config("spark.executor.memory", "100g") \
    .config("spark.driver.memory", "10g") \
    .config("spark.executor.instances", "40") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

try:
    # Load data from Apache Parquet
    data = spark.read.format("parquet").load("../processed_datasets/*.parquet").limit(80000000)
    data = data.drop('__null_dask_index__').dropna().dropDuplicates()

    print("Number of rows:", data.count())

    # Rename a column
    data = data.withColumnRenamed("arrive_time", "label")

    # Prepare data for linear regression training
    assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
    data = assembler.transform(data).select("features", "label")

    # Split data into training and test sets
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=21)

    # Set up the linear regression model
    regressor = LinearRegression(featuresCol="features", labelCol="label")

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

except Exception as e:
    handle_error(e)
finally:
    spark.stop()
