from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression, GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import abs, asc, col
import sys

# Define a function to handle errors during model training
def handle_error(e):
    print("Error occurred during model training: ", e)
    sys.exit(1)

if __name__ == '__main__':    
    try:
        # Create a SparkSession
        spark = SparkSession.builder \
        .appName("PySpark Linear Regression")\
        .config("spark.executor.instances", "4")\
        .config("spark.executor.cores", "2")\
        .config("spark.driver.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .config("spark.executor.extraJavaOptions", "-XX:-UseGCOverheadLimit") \
        .getOrCreate()
        

        print("7 am, 11 am todos los días")

        # Load data from Parquet
        data = spark.read.parquet("../unNormalized_dataset/dataset").coalesce(1)\
                .orderBy(asc("month"), asc("day_of_month"), asc("exit_time")).cache()
        
        dataOG = data


        #Short distances:
        data = dataOG.select("day_of_week", "exit_time", "distance", "exit_stop", "target_stop", "label")
        print("Short distances:")

        data.show(5)
        print("Data size", data.count())

        # Prepare data for regression training
        assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features_a")
        data = assembler.transform(data).select("features_a", "label") 

        #Scale data
        scaler = StandardScaler(inputCol="features_a", outputCol="features")
        scaler_model = scaler.fit(data)
        data = scaler_model.transform(data).select("features", "label")

        data.show(5, truncate=False)

        # Calculate the split point
        split_point = int(data.count() * 0.9)

        # Split the data in order
        train_data = data.limit(split_point).cache() 
        test_data = data.subtract(train_data).cache()  

        #choose regressor
        regressor = LinearRegression(featuresCol="features", labelCol="label")
        print("LinearRegression")

        # Train the regression model
        model = regressor.fit(train_data)

        # Make predictions on the test data
        predictions = model.transform(test_data)

        # Add a new column with the absolute difference between label and prediction
        predictions = predictions.withColumn("abs_diff", abs(predictions["label"] - predictions["prediction"]))

        total_count = predictions.count()

        # Calculate the percentage of data within a ranges of 10, 5, 1, and 0.5 minutes
        ranges = [(10*60), (5*60), 60, 30]

        # Iterate over the ranges and calculate the percentage of data within each range
        for r in ranges:
            within_range_count = predictions.filter(predictions["abs_diff"] <= r).count()
            percent_within_range = within_range_count / total_count * 100
            print("{:.2f}% of data is within a range of {:.0f} seconds".format(percent_within_range, r))

        # Evaluate the model using mean squared error and r2
        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) = {:.4f}".format(rmse))

        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
        r2 = evaluator.evaluate(predictions)
        print("R-squared (R2) = {:.4f}\n".format(r2))

        #load model
        model.save("../trained_models/linearRegression")

    except Exception as e:
        handle_error(e)
    finally:
        spark.stop()