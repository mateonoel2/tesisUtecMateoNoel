from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import abs
import sys

# Define a function to handle errors during model training
def handle_error(e):
    print("Error occurred during model training: ", e)
    sys.exit(1)

if __name__ == '__main__':    
    try:
        # Create a SparkSession
        spark = SparkSession.builder \
        .appName("PySpark Linear Regression") \
        .config("spark.driver.memory", "20g") \
        .config("spark.executor.memory", "90g") \
        .config("spark.executor.instances", "80") \
        .config("spark.executor.cores", "1") \
        .config("spark.driver.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .config("spark.executor.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .getOrCreate()
        
        # Load data from Parquet
        data = spark.read.parquet("../processed_datasets2/dataset")
        
        # Prepare data for regression training
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

        # Add a new column with the absolute difference between label and prediction
        predictions = predictions.withColumn("abs_diff", abs(predictions["label"] - predictions["prediction"]))

        total_count = predictions.count()

        # Calculate the percentage of data within a ranges

        ranges = [(10*60)/86400, (5*60)/86400, 60/86400, 30/86400]

        # Iterate over the ranges and calculate the percentage of data within each range
        for r in ranges:
            within_range_count = predictions.filter(predictions["abs_diff"] <= r).count()
            percent_within_range = within_range_count / total_count * 100
            print("{:.2f}% of data is within a range of {:.0f} seconds".format(percent_within_range, r * 86400))

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
