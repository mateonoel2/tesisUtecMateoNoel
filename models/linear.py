from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import udf
from datetime import datetime, timedelta
import os
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
        .config("spark.local.dir", "./tmp") \
        .config("spark.driver.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .config("spark.executor.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .getOrCreate()

        N = 92
        start_date = datetime(2014, 8, 1)
        end_date = start_date + timedelta(days=N-1)

        date_range = [start_date + timedelta(days=x) for x in range((end_date-start_date).days + 1)]
        date_range_str = [date.strftime("%Y-%m-%d") for date in date_range]

        path = "../processed_datasets/{date}.parquet"

        valid_paths = []
        for date_str in date_range_str:
            file_path = path.format(date=date_str)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > 0:
                    valid_paths.append(file_path)


        print("Number of days to predict:", len(valid_paths))
        
        # Load data from Parquet
        data = spark.read.format("parquet").load(valid_paths)
        data = data.coalesce(1)
        data = data.drop('__null_dask_index__').dropna().dropDuplicates()

        # Get the unique values from exit_stop column
        exit_stops = data.select("exit_stop").distinct().rdd.flatMap(lambda x: x).collect()
        target_stops = data.select("target_stop").distinct().rdd.flatMap(lambda x: x).collect()

        all_stops = list(set(exit_stops + target_stops))
        # Create a mapping from the old values to the new values
        mapping = dict(zip(all_stops, range(1, len(all_stops) + 1)))
        # Create UDFs to apply the mapping to each column
        exit_stop_mapping = udf(lambda exit_stop: mapping[exit_stop])
        target_stop_mapping = udf(lambda target_stop: mapping[target_stop])

        # Apply the mapping to the DataFrame using withColumn
        data = data \
            .withColumn("exit_stop", exit_stop_mapping(data["exit_stop"]).cast("integer")) \
            .withColumn("target_stop", target_stop_mapping(data["target_stop"]).cast("integer"))
        
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
