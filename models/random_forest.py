from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
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
        .config("spark.executor.instances", "1") \
        .config("spark.executor.cores", "80") \
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

        # Set up the Random Forest regression model
        regressor = RandomForestRegressor(featuresCol="features", labelCol="label")

        # Train the Random Forest regression model
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